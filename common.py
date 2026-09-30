"""
Shared plumbing for the three indirect-injection experiments.

Everything runs locally: an agent is an AgentDojo pipeline whose LLM is served by
Ollama through its OpenAI-compatible endpoint. Nothing is sent to a paid service,
and the mock tools never touch the network -- a "sent" e-mail is only recorded so
we can check what the agent tried to do.
"""
import math

import openai
from agentdojo.agent_pipeline import (
    AgentPipeline,
    InitQuery,
    LocalLLM,
    SystemMessage,
    ToolsExecutionLoop,
    ToolsExecutor,
)

OLLAMA_URL = "http://localhost:11434/v1"
TEMPERATURE = 0.7            # fixed across every run; 0.7 so repeated trials actually vary


def build_pipeline(model: str, system_prompt: str, tool_formatter) -> AgentPipeline:
    """A standard tool-using agent: system prompt, the user query, the model, and a
    loop that runs any tool calls and feeds the results back until the model stops."""
    # A per-request timeout so a runaway generation (issue #202: no token cap) fails the
    # single trial instead of stalling the whole run.
    client = openai.OpenAI(base_url=OLLAMA_URL, api_key="ollama", timeout=180)
    llm = LocalLLM(client=client, model=model, temperature=TEMPERATURE)
    return AgentPipeline([
        SystemMessage(system_prompt),
        InitQuery(),
        llm,
        ToolsExecutionLoop([ToolsExecutor(tool_formatter), llm]),
    ])


def yaml_formatter(tool_result):
    """AgentDojo's built-in formatter rejects dicts inside a list; ours renders any
    dict/list as YAML (and keeps non-ASCII readable). Plain strings pass through."""
    import yaml
    if isinstance(tool_result, str):
        return tool_result
    return yaml.safe_dump(tool_result, allow_unicode=True, sort_keys=False).strip()


def text_of(message: dict) -> str:
    """Flatten an AgentDojo message's content parts into plain text."""
    content = message.get("content") or []
    if isinstance(content, str):
        return content
    return "".join(p.get("content", "") for p in content if p.get("type") == "text")


def final_answer(messages: list[dict]) -> str:
    """The last thing the assistant said."""
    return next((text_of(m) for m in reversed(messages) if m.get("role") == "assistant"), "")


def tool_calls(messages: list[dict]) -> list:
    """Every parsed tool call across the transcript (FunctionCall objects)."""
    calls = []
    for m in messages:
        if m.get("role") == "assistant":
            calls.extend(m.get("tool_calls") or [])
    return calls


def wilson(k: int, n: int, z: float = 1.96):
    """Proportion k/n with its 95% Wilson interval, returned as percentages."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * p, 100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def _patch_local_llm_parser() -> None:
    """Make AgentDojo's local-model tool-call parser tolerant of two common quirks
    (AgentDojo issue #202): a no-argument call written as `<function=name></function>`
    with no `{}`, and a body wrapped in a ```json ... ``` fence. The stock parser drops
    both, which silently kills no-argument tools such as list_emails. We install the fix
    here so it ships with the harness and the runs stay reproducible.
    """
    import json
    import re as _re
    from agentdojo.agent_pipeline.llms import local_llm as ll

    def parse(completion: str):
        blank = ll.ChatAssistantMessage(
            role="assistant",
            content=[ll.text_content_block_from_string(completion.strip())],
            tool_calls=[],
        )
        m = _re.search(r"<function\s*=\s*([^>]+)>", completion)
        if not m:
            return blank
        name = m.group(1).strip()
        start = m.end()
        end = completion.find("</function>", start)
        raw = completion[start: end if end != -1 else len(completion)].strip()
        raw = _re.sub(r"^```(?:json)?\s*", "", raw)
        raw = _re.sub(r"\s*```$", "", raw).strip()
        if raw == "":
            raw = "{}"                       # no-arg call the model forgot to write as {}
        try:
            args = json.loads(raw)
        except Exception:
            grab = _re.search(r"\{.*\}", raw, _re.DOTALL)   # first balanced object, ignore trailing junk
            try:
                args = json.loads(grab.group(0)) if grab else None
            except Exception:
                args = None
        if not isinstance(args, dict):
            return blank
        return ll.ChatAssistantMessage(
            role="assistant",
            content=[ll.text_content_block_from_string(completion.strip())],
            tool_calls=[ll.FunctionCall(function=name, args=args)],
        )

    ll._parse_model_output = parse


_patch_local_llm_parser()
