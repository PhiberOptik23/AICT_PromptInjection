# Indirect Prompt Injection in Agentic AI - Evaluation Harness

This repository accompanies the paper *Adversarial Prompt Injection as a Supply Chain
Attack Vector in Agentic AI Systems* (AICT 2026). It contains everything needed to
reproduce the numbers in Tables II and III: the three attack scenarios, the payloads,
the two defenses, and the measurement scripts.

Everything runs locally on a single consumer GPU through [Ollama](https://ollama.com/);
no paid API is involved. The tools the agents call are mocks - a "sent" e-mail is only
logged, nothing leaves the machine - so the harness is safe to run.

## What the experiments show

The three scenarios attack the same agent pattern through different data channels. These
are the measured results (N=30 per cell, temperature 0.7, 95% Wilson intervals; ASR is
conditioned on the payload actually reaching the model).

**Table II - baseline attack success rate (ASR), by scenario**

| Model | (A) E-mail *(action)* | (B) RAG *(answer)* | (C) Web *(answer)* |
|---|---|---|---|
| Llama 3.1 8B | 1% [0-4] (1/122) | 100% [89-100] (30/30) | 12% [4-31] (3/24) |
| Mistral 7B | n/a\* | 84% [65-94] (21/25) | 66% [47-80] (19/29) |
| Qwen2.5 7B | 3% [1-11] (2/62) | 100% [89-100] (30/30) | 100% [88-100] (27/27) |

\*Mistral never opened an e-mail body (0/50 exposed), so it cannot be evaluated as an
agent in the e-mail scenario.

**Table III - effect of defenses (ASR %)**

| Model | Baseline | +Hardened | +Sanitize | +Both |
|---|---|---|---|---|
| *RAG poisoning* | | | | |
| Llama 3.1 8B | 100 | 100 | 100 | 100 |
| Qwen2.5 7B | 100 | 100 | 100 | 100 |
| *Web (concealed HTML)* | | | | |
| Llama 3.1 8B | 9 | 4 | 0 | 0 |
| Qwen2.5 7B | 100 | 100 | 0 | 0 |

The short version: models refuse an injected *action* fairly reliably (e-mail 1-3%), but
they readily adopt injected *content* that shapes their answer (RAG and web, up to 100%).
And the two defenses we test - a hardened "treat this as untrusted data" system prompt and
input sanitization - do **nothing** against RAG poisoning, because a poisoned fact is not
an instruction you can tell the model to ignore. Sanitization only helps in the web case,
where the payload is an HTML comment that can be stripped outright (ASR drops to 0).

## Setup

You need Python 3.10+ and Ollama running locally.

```bash
pip install -r requirements.txt

# pull the three models (~18 GB)
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull mistral:7b
```

On the first RAG run, ChromaDB downloads a small embedding model (~80 MB); after that it
works offline. In the paper, everything ran on an RTX 3060 (6 GB) with 4-bit quantized weights.

## Reproducing the tables

Run everything at once:

```bash
python run_all.py 30      # N=30 per cell; prints Tables II and III, saves results.json
```

Or run a single scenario, which prints ASR with a 95% Wilson interval:

```bash
python scenarios/email_agent.py   # Table II, column A
python scenarios/rag_agent.py     # Table II, column B
python scenarios/web_agent.py     # Table II, column C
```

For the defense results (Table III), call `run(model, n, defense=...)` with `defense`
set to `baseline`, `hardened`, `sanitize`, or `both`. For example, to confirm that no
defense dents RAG poisoning:

```python
from scenarios.rag_agent import run
for d in ["baseline", "hardened", "sanitize", "both"]:
    print(d, run("llama3.1:8b", 30, defense=d))
```

Rates are conditioned on the payload actually reaching the model (the message was read,
or the poisoned document was retrieved); a trial where the agent never ingested the
payload is not counted as "resisted." Numbers will vary run to run - decoding temperature
is 0.7 - but the qualitative picture is stable.

## Layout

```
common.py              pipeline, tool-output formatter, Wilson interval, #202 parser patch
defenses.py            hardened prompts and sanitizers
run_all.py             runs every cell of Tables II and III
scenarios/
  email_agent.py       scenario A
  rag_agent.py         scenario B
  web_agent.py         scenario C
payloads/
  email_injections.txt the five e-mail injection variants
requirements.txt
```

## A couple of honest caveats

- DeepSeek-R1 7B is not included: its distilled checkpoint does not reliably emit tool
  calls through this stack, so it cannot be driven as an agent.
- Mistral 7B often summarizes e-mail from subject lines without opening the bodies, so in
  the e-mail scenario it frequently never sees the payload at all.
- AgentDojo 0.1.35's local-model path can drop a tool call whose JSON is malformed
  (issue #202). We patch the parser in `common.py` so no-argument calls are not lost;
  the remaining dropped calls lower utility but do not inflate ASR.

## Citation

If you use this harness, please cite the AICT 2026 paper.
