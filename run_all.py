"""
Run every cell behind Tables II and III and print the tables. Results are also saved
to results.json so they can be re-tabulated without re-running.

Usage:  python run_all.py [N]      (N trials per cell, default 30)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scenarios"))
import common  # noqa: F401  (installs the #202 parser patch on import)
import email_agent
import rag_agent
import web_agent
from common import wilson

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
EMAIL_MODELS = ["llama3.1:8b", "qwen2.5:7b"]                 # Mistral never opens bodies
ALL_MODELS = ["llama3.1:8b", "qwen2.5:7b", "mistral:7b"]
DEF_MODELS = ["llama3.1:8b", "qwen2.5:7b"]
CONFIGS = ["baseline", "hardened", "sanitize", "both"]


def cell(p, lo, hi, k, n):
    return f"{p:.0f}% [{lo:.0f}-{hi:.0f}] ({k}/{n})"


def main():
    t0 = time.time()
    R = {"N": N, "table2": {}, "table3_rag": {}, "table3_web": {}}

    print(f"### Table II (baseline ASR), N={N}")
    for m in EMAIL_MODELS:
        c = email_agent.run(m, N)
        R["table2"].setdefault(m, {})["email"] = c
        print(f"  email {m:14} exposed {c['exposed']}/{c['n']}  ASR {cell(*wilson(c['asr'], c['exposed']), c['asr'], c['exposed'])}")
    for m in ALL_MODELS:
        c = rag_agent.run(m, N)
        R["table2"].setdefault(m, {})["rag"] = c
        print(f"  rag   {m:14} retrieved {c['retrieved']}/{c['n']}  ASR {cell(*wilson(c['asr'], c['retrieved']), c['asr'], c['retrieved'])}")
    for m in ALL_MODELS:
        c = web_agent.run(m, N)
        R["table2"].setdefault(m, {})["web"] = c
        print(f"  web   {m:14} exposed {c['exposed']}/{c['n']}  ASR {cell(*wilson(c['asr'], c['exposed']), c['asr'], c['exposed'])}")

    print(f"\n### Table III (defenses), N={N}")
    for m in DEF_MODELS:
        for cfg in CONFIGS:
            c = rag_agent.run(m, N, defense=cfg)
            R["table3_rag"].setdefault(m, {})[cfg] = c
            print(f"  rag  {m:14} {cfg:9} ASR {cell(*wilson(c['asr'], c['retrieved']), c['asr'], c['retrieved'])}")
    for m in DEF_MODELS:
        for cfg in CONFIGS:
            c = web_agent.run(m, N, defense=cfg)
            R["table3_web"].setdefault(m, {})[cfg] = c
            print(f"  web  {m:14} {cfg:9} ASR {cell(*wilson(c['asr'], c['exposed']), c['asr'], c['exposed'])}")

    Path("results.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
    print(f"\nSaved results.json. Total wall time: {(time.time()-t0)/60:.0f} min.")


if __name__ == "__main__":
    main()
