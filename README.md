# ⚖️ GOD-LLM Decisions

**One question. Two minds. One auditable verdict.**

GOD-LLM Decisions fuses two System-1 decision models — **Laya** (open-source, runs locally) and **Jev** (TypeSafe AI's commercial API) — into a single decision engine for typed `choice`, `score`, and `noul` (yes/no) questions. Every question is asked to **both** models with identical wording, their probability distributions are fused with a documented confidence-weighted rule, disagreements are escalated to a local adjudicator, and the whole thing is written to an append-only JSON decision log.

No black box. No single point of failure. Every number in every certificate traces back to a model response or a clock.

---

## 🧭 Table of contents

- [How it works](#-how-it-works)
- [The two models](#-the-two-models)
- [The fusion rule](#-the-fusion-rule)
- [Decision certificates](#-decision-certificates)
- [Quick start](#-quick-start)
- [Usage](#-usage)
- [Project structure](#-project-structure)
- [Costs](#-costs)
- [Honest limitations](#-honest-limitations)
- [License](#-license)

---

## 🧠 How it works

```
                          ┌─────────────────────┐
                          │   your question      │
                          │ (identical wording)  │
                          └────────┬────────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
        ┌───────────────┐  ┌───────────────┐
        │  LAYA         │  │  JEV          │
        │  local CPU    │  │  TypeSafe API │
        │  open source  │  │  commercial   │
        └───────┬───────┘  └───────┬───────┘
                │ probabilities    │ probabilities
                │ + confidence     │ + confidence
                ▼                  ▼
        ┌───────────────────────────────────┐
        │   FUSION (confidence-weighted)     │
        │   p_fused = (cL·pL + cJ·pJ)/(cL+cJ)│
        └───────────────┬───────────────────┘
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
        ✅ AGREE              ⚠️ DISPUTE
        verdict =             escalated to local
        fused winner          adjudicator (Ollama)
                              → final verdict + rationale
                        │
                        ▼
        ┌───────────────────────────────────┐
        │  📜 decision certificate           │
        │  appended to logs/decisions.log   │
        └───────────────────────────────────┘
```

**Design principles**

1. **Two independent witnesses.** One model can be wrong; two models wrong in the same way is rarer.
2. **Confidence-weighted, not vote-counted.** A sure model outweighs an unsure one — mathematically, not by vibes.
3. **Disputes are preserved, not hidden.** When the models disagree, a third (local) adjudicator breaks the tie, and the disagreement stays in the record.
4. **Refuse a one-model verdict by default.** If only one backend answers, the engine refuses unless you explicitly pass `--allow-single`.
5. **Nothing is invented.** Every number in a certificate comes from a model response or a clock.

---

## 🤖 The two models

| | **Laya** | **Jev** |
|---|---|---|
| What | Open-source typed-decision model (Convai Innovations) | TypeSafe AI's commercial System-1 model |
| License | Apache 2.0 | Commercial API |
| Where it runs | Locally, on CPU — zero marginal cost | `POST https://api.typesafe.ai/v1/systemone` |
| Checkpoint | `convaiinnovations/laya` → `typed-decisions` (421M params) | `jev-latest` (resolves to a versioned model, e.g. `jev-1.13.0`) |
| Auth | None needed | `TYPESAFE_API_KEY` env var (Bearer) |
| Package | `laya 0.3.6` | `typesafe-sdk 0.7.1` |

**Dispute adjudicator:** local [Ollama](https://ollama.com) model (`dolphin-phi:2.7b`, ~1.6 GB). It only sees the two models' distributions and picks the final label with a one-sentence rationale.

---

## ➗ The fusion rule

For `choice` and `score` questions, both models return a probability distribution over the same labels. Fusion is a confidence-weighted average, renormalized:

```
p_fused[i] = (c_L · p_L[i] + c_J · p_J[i]) / (c_L + c_J)
```

For `noul` (yes/no) questions, the same weighting applies to P(true):

```
p_fused = (c_L · p_L + c_J · p_J) / (c_L + c_J)
```

- **Agreement** → verdict = fused winner, confidence = mean of the two confidences.
- **Disagreement** (`choice`/`score`) → escalated to the local adjudicator; its verdict and rationale are stored in the certificate.
- **Disagreement** (`noul`) → marked `UNRESOLVED` — routed to human review. A coin-flip question should never be decided by a coin-flip tiebreak.

---

## 📜 Decision certificates

Every decision appends one JSON certificate to `logs/decisions.log`:

```json
{
  "engine": "GOD-LEVEL DECISION ENGINE v1",
  "timestamp_utc": "2026-09-23T...",
  "state": "Customer Acme Corp reports checkout HTTP 500...",
  "backends_live": ["laya", "jev"],
  "questions": {
    "priority": {
      "fused_winner": "p1_critical",
      "fused_probabilities": {"p1_critical": 0.72, "p2_high": 0.18, ...},
      "fused_confidence": 0.81,
      "agreement": "AGREE",
      "verdict_source": "fused consensus (both models agree)",
      "models": {
        "laya": {"winner": "p1_critical", "confidence": 0.77, "latency_ms": 1640, ...},
        "jev":  {"winner": "p1_critical", "confidence": 0.85, "latency_ms": 920, ...}
      }
    }
  }
}
```

Certificates are append-only. (Cryptographic hash-chaining is on the roadmap — today the log is JSON, not signed.)

---

## 🚀 Quick start

### 1. Clone & install

```bash
git clone https://github.com/<you>/god-llm-decisions.git
cd god-llm-decisions
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

This installs `torch`, `laya`, and `typesafe-sdk`. The Laya checkpoint (~800 MB) downloads automatically from Hugging Face on first load into `hf_cache/`.

### 2. Configure Jev (optional but recommended)

```bash
export TYPESAFE_API_KEY="ts_..."   # from the TypeSafe dashboard
```

Without a key, the engine runs on Laya alone (single-model mode) — or refuses, depending on your flags.

### 3. Check backends

```bash
python3 god_decide.py status
```

### 4. Run the demo — three canonical questions, both models, one fused verdict

```bash
python3 god_decide.py demo
```

---

## 💻 Usage

**Ask your own questions:**

```bash
python3 god_decide.py decide --state "…" --q examples/questions.json
```

`questions.json` is a list of typed questions:

```json
[
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
   "criteria": ["calm", "concerned", "urgent", "angry"]}
]
```

**Flags**

| Flag | Meaning |
|---|---|
| `decide --allow-single` | Permit a verdict when only one backend answers (default: refuse) |

**Question types**

| Type | Asks | Returns |
|---|---|---|
| `choice` | pick one of N labeled options | winner label + probability distribution |
| `score` | rate against a rubric | winning rubric label + distribution |
| `noul` | yes/no | P(true) in [0, 1] |

---

## 📁 Project structure

```
god-llm-decisions/
├── god_decide.py        # the engine: backends, fusion, adjudication, CLI
├── install_laya.sh      # one-shot environment setup
├── requirements.txt     # laya, typesafe-sdk, torch
├── examples/
│   └── questions.json   # sample question set for `decide`
├── hf_cache/            # Laya checkpoint (downloaded, git-ignored)
├── logs/
│   └── decisions.log    # append-only decision certificates (git-ignored)
└── README.md
```

---

## 💰 Costs

| Component | Cost |
|---|---|
| Laya inference | Free — runs on your own CPU |
| Jev API | Per TypeSafe pricing (~$0.042 / 1M input tokens, per docs) |
| Adjudicator (Ollama) | Free — local |

---

## ⚠️ Honest limitations

- **Fusion is an ensemble heuristic, not a proof.** Confidence-weighted averaging is principled but uncalibrated for your domain until you calibrate it. Validate on your own labeled data before trusting consequential decisions.
- **Laya's confidence runs low out of the box.** In smoke tests, choice/score confidences were ~0.19–0.29. Treat fused confidence as a signal, not an authorization.
- **The adjudicator is a small local model** (`dolphin-phi:2.7b`). It breaks ties; it must not override safety, financial, or irreversible-action rules.
- **Benchmarks, honestly sourced.** Laya's reported 0.766 figure comes from its repo's task-fine-tuned checkpoint comparisons; base checkpoints score far lower. Compare on your own benchmark before claiming superiority.
- **noul disputes go to a human.** By design.
- **The log is JSON, not yet signed.** Hash-chaining is planned.

---

## 📄 License

MIT — see [LICENSE](LICENSE). Laya itself is Apache 2.0 (Convai Innovations); Jev is a commercial TypeSafe AI service with its own terms.

---

*Built to make fast decisions auditable — because a decision you can't inspect is a decision you can't trust.* ⚖️
