#!/bin/bash
cd ~/workspace/decision-engine
export HF_HOME=~/workspace/decision-engine/hf_cache
export TMPDIR=~/workspace/decision-engine/pip_tmp
export PIP_NO_CACHE_DIR=1
python3 -m venv venv
source venv/bin/activate
echo "[$(date +%T)] Installing torch from PyPI..."
pip install torch || { echo "TORCH FAILED"; exit 1; }
echo "[$(date +%T)] Installing laya..."
pip install laya || { echo "LAYA FAILED"; exit 1; }
echo "[$(date +%T)] Installing typesafe-sdk..."
pip install typesafe-sdk || { echo "TYPESAFE FAILED"; exit 1; }
echo "[$(date +%T)] INSTALL DONE"
