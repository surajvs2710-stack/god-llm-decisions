#!/usr/bin/env python3
"""
GOD-LEVEL DECISION ENGINE
========================
Fuses two System-1 decision models into one verdict:

  1. LAYA  (local, open-source, Apache 2.0) — Convai Innovations
     Runs on this VM's CPU in float32. Zero marginal cost.
  2. JEV   (TypeSafe AI commercial API, jev-latest -> jev-1.13.0)
     POST https://api.typesafe.ai/v1/systemone, $0.042 / 1M input tokens.

Fusion rule (documented, no magic):
  - Every question is asked to BOTH models with identical wording.
  - choice/score: confidence-weighted average of the two probability
    distributions:  p_fused[i] = (cL*pL[i] + cJ*pJ[i]) / (cL + cJ)
  - noul:           p_fused     = (cL*pL     + cJ*pJ)     / (cL + cJ)
  - If both models pick the same winner -> DECIDED.
  - If they disagree -> DISPUTED -> escalated to the adjudicator LLM
    (local Ollama), which sees both distributions and returns the final
    verdict with rationale. The dispute is preserved in the decision record.

Every decision is written to decisions.log as a JSON decision record
(NOT cryptographically signed — plain JSON audit entries):
verdict, fused distribution, per-model breakdown, agreement status,
latencies, model IDs, timestamp. Nothing is invented: every number in
the decision record comes from a model response or a clock.

Usage:
  god_decide.py demo                        # 3 canonical questions
  god_decide.py decide --state "..." --q questions.json
  god_decide.py status                      # backend health check
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "logs", "decisions.log")
HF_CACHE = os.path.join(BASE_DIR, "hf_cache")
os.environ.setdefault("HF_HOME", HF_CACHE)
# This VM's NO_PROXY carries bare IPv6 entries (::1, fd8b:...) which httpx's
# URLPattern cannot parse -> huggingface_hub crashes. Sanitize once, centrally.
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

# ---------------------------------------------------------------- backends

class LayaBackend:
    """Local Laya inference on CPU. Verified against laya 0.3.6:
    laya.load(repo, subfolder=..., device='cpu') -> Agent;
    agent.system_one(state, {qid: {type, instructions, criteria?}})
    -> {"model": "laya-rl-agent", "answers": {qid: {...}}, "usage": {...}}.
    """

    def __init__(self, repo="convaiinnovations/laya", subfolder="typed-decisions"):
        self.repo = repo
        self.subfolder = subfolder
        self._agent = None
        self.available = False
        self.error = None
        try:
            import laya  # noqa
            self.available = True
        except Exception as e:  # pragma: no cover
            self.error = f"laya package not importable: {e}"

    @property
    def checkpoint(self):
        return f"{self.repo}#{self.subfolder}" if self.subfolder else self.repo

    def load(self):
        """Load weights. Separated from __init__ so import never blocks."""
        import laya

        self._agent = laya.load(self.repo, subfolder=self.subfolder, device="cpu")
        return self

    def predict(self, state, questions):
        """questions: list of dicts {id, type, instructions, criteria?}.
        Returns {qid: {winner, probabilities, confidence, latency_ms}}."""
        qdict = {}
        for q in questions:
            d = {"type": q["type"], "instructions": q["instructions"]}
            if q["type"] in ("choice", "score") and q.get("criteria") is not None:
                d["criteria"] = q["criteria"]
            qdict[q["id"]] = d
        t0 = time.time()
        raw = self._agent.system_one(state, qdict)
        dt_ms = (time.time() - t0) * 1000.0 / max(len(questions), 1)
        model_id = raw.get("model", "laya-rl-agent")
        out = {}
        for q in questions:
            a = raw["answers"][q["id"]]
            if a["type"] == "choice":
                out[q["id"]] = {
                    "winner": a["choice"],
                    "probabilities": dict(a["probabilities"]),
                    "confidence": float(a["confidence"]),
                    "noul": None,
                    "latency_ms": dt_ms,
                    "backend": "laya",
                    "model_id": f"{model_id} ({self.checkpoint})",
                }
            elif a["type"] == "score":
                legend = {str(k): v for k, v in dict(a["legend"]).items()}
                probs_by_label = {legend[k]: float(v)
                                  for k, v in dict(a["probabilities"]).items()}
                winner_label = legend[str(int(round(a["score"])))]
                out[q["id"]] = {
                    "winner": winner_label,
                    "probabilities": probs_by_label,
                    "confidence": float(a["confidence"]),
                    "noul": None,
                    "latency_ms": dt_ms,
                    "backend": "laya",
                    "model_id": f"{model_id} ({self.checkpoint})",
                }
            else:  # noul
                out[q["id"]] = {
                    "winner": None,
                    "probabilities": None,
                    "confidence": float(a["confidence"]),
                    "noul": float(a["noul"]),
                    "latency_ms": dt_ms,
                    "backend": "laya",
                    "model_id": f"{model_id} ({self.checkpoint})",
                }
        return out


class JevBackend:
    """TypeSafe AI Jev via official SDK. Key from TYPESAFE_API_KEY env."""

    ENDPOINT = "https://api.typesafe.ai/v1/systemone"

    def __init__(self):
        self.available = False
        self.error = None
        self._client = None
        try:
            import typesafe_sdk  # noqa
            self._sdk = typesafe_sdk
        except Exception as e:
            self.error = f"typesafe_sdk not importable: {e}"
            return
        if not os.environ.get("TYPESAFE_API_KEY"):
            self.error = "TYPESAFE_API_KEY not set (provide via Secure Vault)"
            return
        self.available = True

    def predict(self, state, questions):
        from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

        sdk_questions = {}
        for q in questions:
            ins = q["instructions"]
            if q["type"] == "choice":
                sdk_questions[q["id"]] = Choice(instructions=ins, criteria=q["criteria"])
            elif q["type"] == "score":
                sdk_questions[q["id"]] = Score(instructions=ins, criteria=q["criteria"])
            elif q["type"] == "noul":
                sdk_questions[q["id"]] = Noul(instructions=ins)
            else:
                raise ValueError(f"unknown question type: {q['type']}")

        t0 = time.time()
        # Per official docs quickstart: plain instantiation, reads
        # TYPESAFE_API_KEY from the environment, defaults to jev-latest.
        client = TypeSafeClient()
        resp = client.system_one(state=state, questions=sdk_questions)
        dt_ms = (time.time() - t0) * 1000.0 / max(len(questions), 1)

        out = {}
        model_id = getattr(resp, "model", "jev-latest")
        for q in questions:
            a = resp.answers[q["id"]]
            if q["type"] == "choice":
                out[q["id"]] = {
                    "winner": a.choice,
                    "probabilities": dict(a.probabilities),
                    "confidence": float(a.confidence),
                    "noul": None,
                    "latency_ms": dt_ms,
                    "backend": "jev",
                    "model_id": model_id,
                }
            elif q["type"] == "score":
                # SDK returns index-keyed probs + legend {idx: label};
                # map back to rubric labels so fusion aligns with Laya.
                legend = {str(k): v for k, v in dict(a.legend).items()}
                probs_by_label = {legend[k]: float(v)
                                  for k, v in dict(a.probabilities).items()}
                winner_label = legend[str(int(a.score))]
                out[q["id"]] = {
                    "winner": winner_label,
                    "probabilities": probs_by_label,
                    "confidence": float(a.confidence),
                    "noul": None,
                    "latency_ms": dt_ms,
                    "backend": "jev",
                    "model_id": model_id,
                }
            else:
                out[q["id"]] = {
                    "winner": None,
                    "probabilities": None,
                    "confidence": float(a.noul) if a.noul >= 0.5 else float(1 - a.noul),
                    "noul": float(a.noul),
                    "latency_ms": dt_ms,
                    "backend": "jev",
                    "model_id": model_id,
                }
        return out


# ---------------------------------------------------------------- fusion

def _fuse_dist(pL, cL, pJ, cJ):
    """Confidence-weighted average of two distributions over the same labels."""
    labels = list(pL.keys())
    denom = cL + cJ
    if denom <= 0:
        denom = 1e-9
    fused = {k: (cL * pL[k] + cJ * pJ.get(k, 0.0)) / denom for k in labels}
    s = sum(fused.values())
    if s > 0:
        fused = {k: v / s for k, v in fused.items()}
    winner = max(fused, key=fused.get)
    return fused, winner


def adjudicate(state, question, laya_ans, jev_ans):
    """Disagreement escalation: local Ollama adjudicator sees both
    distributions and returns the final verdict. Returns (winner, rationale)
    or (None, reason) if the adjudicator is unreachable."""
    prompt = (
        "You are the adjudicator of a decision council. Two calibrated "
        "decision models disagreed. Return ONLY valid JSON: "
        '{"winner": "<exact label>", "rationale": "<one sentence>"}. '
        "The winner MUST be one of the listed labels, exactly as written.\n\n"
        f"STATE: {state}\n"
        f"QUESTION: {question['instructions']}\n"
        f"LABELS: {list(laya_ans['probabilities'].keys())}\n"
        f"LAYA (local open-source) -> winner={laya_ans['winner']} "
        f"probs={laya_ans['probabilities']} confidence={laya_ans['confidence']:.3f}\n"
        f"JEV (TypeSafe commercial) -> winner={jev_ans['winner']} "
        f"probs={jev_ans['probabilities']} confidence={jev_ans['confidence']:.3f}\n"
    )
    try:
        r = subprocess.run(
            ["ollama", "run", "dolphin-phi:2.7b", prompt],
            capture_output=True, text=True, timeout=180,
            env={**os.environ, "PATH": os.path.expanduser("~/workspace/ollama/bin") + ":" + os.environ.get("PATH", "")},
        )
        txt = r.stdout.strip()
        start, end = txt.find("{"), txt.rfind("}")
        if start == -1 or end == -1:
            return None, "adjudicator returned no JSON"
        data = json.loads(txt[start:end + 1])
        labels = list(laya_ans["probabilities"].keys())
        if data.get("winner") not in labels:
            return None, f"adjudicator winner not in labels: {data.get('winner')}"
        return data["winner"], data.get("rationale", "")
    except FileNotFoundError:
        return None, "ollama binary not found"
    except subprocess.TimeoutExpired:
        return None, "adjudicator timed out"
    except Exception as e:
        return None, f"adjudicator error: {e}"


def god_decide(state, questions, laya=None, jev=None, allow_single=False):
    """Run both models, fuse, escalate disputes. Returns the decision record."""
    t_start = time.time()
    results = {"laya": None, "jev": None, "errors": {}}

    if laya and laya.available:
        try:
            results["laya"] = laya.predict(state, questions)
        except Exception as e:
            results["errors"]["laya"] = str(e)
    elif laya:
        results["errors"]["laya"] = laya.error

    if jev and jev.available:
        try:
            results["jev"] = jev.predict(state, questions)
        except Exception as e:
            results["errors"]["jev"] = str(e)
    elif jev:
        results["errors"]["jev"] = jev.error

    live = [k for k in ("laya", "jev") if results[k] is not None]
    if not live:
        raise RuntimeError(f"no backend produced answers: {results['errors']}")
    if len(live) == 1 and not allow_single:
        raise RuntimeError(
            f"only {live[0]} answered; refusing single-model verdict "
            f"(pass allow_single=True to override). errors={results['errors']}"
        )

    fused_questions = {}
    for q in questions:
        qid = q["id"]
        entry = {"question": q, "models": {}}
        for name in live:
            entry["models"][name] = results[name][qid]

        if len(live) == 2:
            L, J = entry["models"]["laya"], entry["models"]["jev"]
            if q["type"] in ("choice", "score"):
                fused_probs, fused_winner = _fuse_dist(
                    L["probabilities"], L["confidence"],
                    J["probabilities"], J["confidence"],
                )
                agree = L["winner"] == J["winner"]
                entry.update({
                    "fused_probabilities": fused_probs,
                    "fused_winner": fused_winner,
                    "fused_confidence": round(
                        (L["confidence"] + J["confidence"]) / 2 if agree
                        else float(fused_probs[fused_winner]), 4),
                    "agreement": "AGREE" if agree else "DISPUTE",
                })
            else:  # noul
                denom = L["confidence"] + J["confidence"] or 1e-9
                fused_p = (L["confidence"] * L["noul"] + J["confidence"] * J["noul"]) / denom
                agree = (L["noul"] >= 0.5) == (J["noul"] >= 0.5)
                entry.update({
                    "fused_noul": round(fused_p, 4),
                    "fused_winner": "TRUE" if fused_p >= 0.5 else "FALSE",
                    "fused_confidence": round(abs(fused_p - 0.5) * 2, 4),
                    "agreement": "AGREE" if agree else "DISPUTE",
                })

            if entry["agreement"] == "DISPUTE" and q["type"] in ("choice", "score"):
                winner, rationale = adjudicate(state, q, L, J)
                entry["adjudication"] = {
                    "winner": winner, "rationale": rationale,
                    "status": "RESOLVED" if winner else "UNRESOLVED",
                }
                if winner:
                    entry["fused_winner"] = winner
                    entry["verdict_source"] = "adjudicator (dispute resolved)"
                else:
                    entry["verdict_source"] = "UNRESOLVED dispute — human review required"
            elif entry["agreement"] == "DISPUTE":
                entry["verdict_source"] = "UNRESOLVED noul dispute — human review required"
            else:
                entry["verdict_source"] = "fused consensus (both models agree)"
        else:
            name = live[0]
            a = entry["models"][name]
            entry.update({
                "fused_winner": a["winner"],
                "fused_probabilities": a["probabilities"],
                "fused_noul": a["noul"],
                "fused_confidence": round(a["confidence"], 4),
                "agreement": "SINGLE-MODEL",
                "verdict_source": f"single model ({name}) — other backend unavailable",
            })

        fused_questions[qid] = entry

    cert = {
        "engine": "GOD-LEVEL DECISION ENGINE v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "backends_live": live,
        "errors": results["errors"],
        "questions": fused_questions,
        "total_latency_ms": round((time.time() - t_start) * 1000.0, 1),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(cert) + "\n")
    return cert


# ---------------------------------------------------------------- CLI

DEMO_QUESTIONS = [
    {"id": "priority", "type": "choice",
     "instructions": "How should this support ticket be prioritized?",
     "criteria": {
         "p1_critical": "Production is down or money is being lost right now.",
         "p2_high": "Major feature broken but a workaround exists.",
         "p3_normal": "Routine request, no time pressure.",
         "p4_low": "Nice-to-have or informational."}},
    {"id": "refund_ok", "type": "noul",
     "instructions": "Does company policy allow an instant refund in this case?"},
    {"id": "tone", "type": "score",
     "instructions": "How urgent does the customer sound?",
     "criteria": ["calm", "concerned", "urgent", "angry"]},
]
DEMO_STATE = ("Customer Acme Corp reports their checkout page returns HTTP 500 "
              "for all users since 09:15 IST. They process ~200 orders/day and "
              "are asking for an immediate fix and a refund of today's fees.")


def cmd_status():
    laya, jev = LayaBackend(), JevBackend()
    print(json.dumps({
        "laya": {"available": laya.available, "error": laya.error,
                 "checkpoint": laya.checkpoint},
        "jev": {"available": jev.available, "error": jev.error,
                "endpoint": JevBackend.ENDPOINT},
        "log": LOG_PATH,
    }, indent=2))


def cmd_demo():
    laya, jev = LayaBackend(), JevBackend()
    if laya.available:
        print("loading Laya weights (float32, CPU)...", flush=True)
        laya.load()
    cert = god_decide(DEMO_STATE, DEMO_QUESTIONS, laya=laya, jev=jev,
                      allow_single=True)
    print(json.dumps(cert, indent=2))


def main():
    ap = argparse.ArgumentParser(description="God-level decision engine: Laya + Jev fused.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="backend health check")
    sub.add_parser("demo", help="run 3 canonical questions through both models")
    d = sub.add_parser("decide", help="decide from a questions JSON file")
    d.add_argument("--state", required=True)
    d.add_argument("--q", required=True, help="path to questions JSON list")
    d.add_argument("--allow-single", action="store_true")
    args = ap.parse_args()

    if args.cmd == "status":
        return cmd_status()
    if args.cmd == "demo":
        return cmd_demo()
    laya, jev = LayaBackend(), JevBackend()
    if laya.available:
        laya.load()
    with open(args.q) as f:
        questions = json.load(f)
    cert = god_decide(args.state, questions, laya=laya, jev=jev,
                      allow_single=args.allow_single)
    print(json.dumps(cert, indent=2))


if __name__ == "__main__":
    main()
