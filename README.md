# Indirect Prompt Injection in Agentic AI — Evaluation Harness

This repository accompanies the paper *Adversarial Prompt Injection as a Supply Chain
Attack Vector in Agentic AI Systems* (AICT 2026). It contains everything needed to
reproduce the numbers in Tables II and III: the three attack scenarios, the payloads,
the two defenses, and the measurement scripts.

Everything runs locally on a single consumer GPU through [Ollama](https://ollama.com/);
no paid API is involved. The tools the agents call are mocks - a "sent" e-mail is only
logged, nothing leaves the machine - so the harness is safe to run.

## What the experiments show

The three scenarios attack the same agent pattern through different data channels:

| Scenario | What the injection asks for | Attack success (ASR) |
|---|---|---|
| **A — e-mail** | call `send_email` to the attacker (an *action*) | ~0–14% |
| **B — RAG** | repeat a poisoned fact (an *answer*) | ~100% |
| **C — web** | report a phishing address from a page (an *answer*) | ~50–100% |

The short version: models refuse an injected *action* fairly reliably, but they will
happily adopt injected *content* that shapes their answer. And the two defenses we test
(a hardened "treat this as untrusted data" system prompt, and input sanitization) do
**nothing** against RAG poisoning - because a poisoned fact is not an instruction you
can tell the model to ignore. Sanitization only helps in the web case, where the payload
is an HTML comment that can be stripped outright.

## Setup

You need Python 3.10+ and Ollama running locally.

```bash
pip install -r requirements.txt

# pull the three models (≈18 GB)
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull mistral:7b
```

On the first RAG run, ChromaDB downloads a small embedding model (~80 MB); after that it
works offline. In the paper, everything ran on an RTX 3060 (6 GB) with 4-bit quantized weights.

## Reproducing the tables

Each scenario is runnable on its own and prints ASR with a 95% Wilson interval:

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
    print(d, run("llama3.1:8b", 10, defense=d))
```

Rates are conditioned on the payload actually reaching the model (the message was read,
or the poisoned document was retrieved); a trial where the agent never ingested the
payload is not counted as "resisted." Numbers will vary run to run - decoding temperature
is 0.7 and each cell is only N=10 - but the qualitative picture is stable.

## Layout

```
common.py              pipeline + tool-output formatter + Wilson interval
defenses.py            hardened prompts and sanitizers
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
- AgentDojo 0.1.35's local-model path occasionally drops a tool call whose JSON is
  malformed (issue #202). Those trials show up as a missed action; they lower utility but
  do not inflate ASR.

## Citation

If you use this harness, please cite the AICT 2026 paper.
