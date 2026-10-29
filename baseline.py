"""Phase 2 baseline: run ONE legitimate AgentDojo task, with no attack and no memory.

What this script does, in order:
  1. Load the workspace suite and one user task (default: user_task_0).
  2. Build the agent (an LLM + the loop that runs tools) the way AgentDojo itself does it.
  3. Let AgentDojo run the task: it sets up the environment, takes the "before" snapshot,
     runs the agent, and calls the task's own utility check on the "after" state.
  4. Save what happened to results/<run_id>/result.json.

Run it:
  uv run python baseline.py                      # live run, needs OPENAI_API_KEY in .env
  uv run python baseline.py --ground-truth       # no LLM, no cost: replays the task's answer key
"""

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv

from agentdojo.agent_pipeline import AgentPipeline, BasePipelineElement, GroundTruthPipeline, OpenAILLM, PipelineConfig
from agentdojo.models import ModelsEnum
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import get_text_content_as_str

DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"


class RecordingPipeline(BasePipelineElement):
    """Wraps the real agent and keeps a copy of the conversation.

    AgentDojo's run_task_with_pipeline only returns True/False results, so without this
    we could not save the agent's answer or the tool calls it made. It changes nothing
    about how the agent behaves.
    """

    def __init__(self, inner: BasePipelineElement):
        self.inner = inner
        self.name = getattr(inner, "name", None)
        self.messages = []

    def query(self, query, runtime, env, messages=[], extra_args={}):
        result = self.inner.query(query, runtime, env, messages, extra_args)
        self.messages = list(result[3])  # result = (query, runtime, env, messages, extra_args)
        return result


def build_agent(model: str) -> BasePipelineElement:
    """The agent = system message + user prompt + LLM + tool loop (AgentDojo's standard setup, no defense)."""
    if model in [m.value for m in ModelsEnum]:
        llm = model  # a model AgentDojo knows by name
    else:
        import openai  # any other OpenAI model name, e.g. "gpt-4o-mini"

        llm = OpenAILLM(openai.OpenAI(), model)
    config = PipelineConfig(llm=llm, model_id=None, defense=None, system_message_name=None, system_message=None)
    return AgentPipeline.from_config(config)


def summarise_messages(messages) -> tuple[str, list[dict]]:
    """Pull out the agent's final answer and every tool call (with its result) from the conversation."""
    tool_calls, final_answer = [], ""
    for msg in messages:
        if msg["role"] == "tool":
            tool_calls.append(
                {
                    "function": msg["tool_call"].function,
                    "args": msg["tool_call"].args,
                    "error": msg["error"],
                    "result": get_text_content_as_str(msg["content"]),
                }
            )
        elif msg["role"] == "assistant" and msg["content"]:
            final_answer = get_text_content_as_str(msg["content"])
    return final_answer, tool_calls


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one legitimate AgentDojo task (no attack, no memory).")
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--suite", default="workspace")
    parser.add_argument("--task", default="user_task_0")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ground-truth", action="store_true", help="No LLM: replay the task's answer key (offline check).")
    args = parser.parse_args()

    load_dotenv()  # puts OPENAI_API_KEY from .env into the environment; the key itself is never logged

    suite = get_suite(args.benchmark_version, args.suite)
    task = suite.get_user_task_by_id(args.task)

    if args.ground_truth:
        agent = RecordingPipeline(GroundTruthPipeline(task))
        mode, model = "ground_truth_no_llm", None
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("No OPENAI_API_KEY found. Add it to .env in the project folder, or use --ground-truth.")
        agent = RecordingPipeline(build_agent(args.model))
        mode, model = "live", args.model

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    started = datetime.now(timezone.utc)
    error = None
    utility = None
    try:
        # injection_task=None and injections={} -> every placeholder gets its harmless default text.
        utility, _ = suite.run_task_with_pipeline(agent, task, injection_task=None, injections={})
    if error is not None:
        raise SystemExit(1)
    finished = datetime.now(timezone.utc)

    final_answer, tool_calls = summarise_messages(agent.messages)
    result = {
        "run_id": run_id,
        "mode": mode,
        "started_utc": started.isoformat(),
        "duration_s": round((finished - started).total_seconds(), 2),
        "agentdojo_version": version("agentdojo"),
        "benchmark_version": args.benchmark_version,
        "suite": args.suite,
        "task_id": args.task,
        "task_prompt": task.PROMPT,
        "model": model,
        "attack": None,
        "memory": None,
        "final_answer": final_answer,
        "tool_calls": tool_calls,
        "utility": utility,
        "error": error,
    }

    out_dir = Path("results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))

    print(f"mode:       {mode}")
    print(f"task:       {args.suite}/{args.task} ({args.benchmark_version})")
    print(f"tool calls: {[c['function'] for c in tool_calls]}")
    print(f"utility:    {utility}")
    if error:
        print(f"ERROR:      {error}")
    print(f"saved:      {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()