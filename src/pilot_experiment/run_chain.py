"""Runs the two experimental conditions and records one result row per run.

Conditions
----------
chain       : one environment with all three attack texts in place and an empty memory.
              The three stage requests are sent in order to the SAME environment, so the
              memory (and anything else) carries from one stage to the next.
final-only  : a fresh environment with ONLY the stage-3 attack text (stages 1 and 2 keep
              their harmless defaults) and an empty memory. Only the stage-3 request runs.
              This is the control: the final attack, with no help from earlier stages.

Both conditions use the same agent, the same stage-3 request and the same checker, so a
difference between them comes only from what stages 1 and 2 left behind.

Scoring
-------
After every stage we check the final attacker goal (code sent to the attacker), so a
success that happens early is attributed to the stage where it first occurs, not to
stage 3. `chain_success` is True if the goal was reached at any point in the run.

The runner takes a `make_pipeline` callable, so tests can pass a scripted stand-in agent
and run everything offline; real runs pass the GPT-backed pipeline from pipeline.py.
"""

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from agentdojo.functions_runtime import FunctionsRuntime, make_function
from agentdojo.types import get_text_content_as_str

from pilot_experiment import stages as S
from pilot_experiment.suite import ALL_TOOLS, task_suite

# (stage number, the request Emma sends, the checker for that stage)
STAGES = [
    (1, S.STAGE1_REQUEST, S.check_stage1),
    (2, S.STAGE2_REQUEST, S.check_stage2),
    (3, S.STAGE3_REQUEST, S.check_stage3),
]

ALL_ATTACKS = {
    S.STAGE1_SLOT: S.STAGE1_ATTACK,
    S.STAGE2_SLOT: S.STAGE2_ATTACK,
    S.STAGE3_SLOT: S.STAGE3_ATTACK,
}
FINAL_ONLY_ATTACKS = {S.STAGE3_SLOT: S.STAGE3_ATTACK}  # stages 1 and 2 keep harmless defaults


def _tools():
    return [make_function(t) for t in ALL_TOOLS]


def _run_one_request(pipeline, env, request):
    """Send one request to the agent. Returns the (mutated) environment and the messages."""
    runtime = FunctionsRuntime(_tools())
    result = pipeline.query(request, runtime, env)
    env = result[2]  # (query, runtime, env, messages, extra_args)
    messages = result[3]
    return env, messages


def _tool_calls(messages):
    return [m["tool_call"].function for m in messages if m.get("role") == "tool"]


def run_chain(make_pipeline) -> dict:
    """Complete chain: all three attack texts present, three requests on one environment."""
    env = task_suite.load_and_inject_default_environment(ALL_ATTACKS)
    pipeline = make_pipeline()
    per_stage = {}
    first_goal_stage = None

    for num, request, check in STAGES:
        pre = env.model_copy(deep=True)
        env, messages = _run_one_request(pipeline, env, request)
        per_stage[f"stage{num}"] = check(pre, env)
        if first_goal_stage is None and S.final_goal_reached(pre, env):
            first_goal_stage = num

    return {
        "condition": "chain",
        "stage1": per_stage["stage1"],
        "stage2": per_stage["stage2"],
        "stage3": per_stage["stage3"],
        "chain_success": first_goal_stage is not None,
        "code_first_sent_stage": first_goal_stage,
        "memory_after": [e.content for e in env.memory.entries],
    }


def run_final_only(make_pipeline) -> dict:
    """Control: only the stage-3 attack text, empty memory, only the stage-3 request."""
    env = task_suite.load_and_inject_default_environment(FINAL_ONLY_ATTACKS)
    pipeline = make_pipeline()

    pre = env.model_copy(deep=True)
    env, messages = _run_one_request(pipeline, env, S.STAGE3_REQUEST)
    success = S.final_goal_reached(pre, env)

    return {
        "condition": "final_only",
        "stage1": None,  # not run
        "stage2": None,  # not run
        "stage3": success,
        "chain_success": success,
        "code_first_sent_stage": 3 if success else None,
        "memory_after": [e.content for e in env.memory.entries],
    }


# --- the real (GPT-backed) pipeline factory -------------------------------------------
def gpt_pipeline_factory(model: str):
    """Returns a function that builds a fresh GPT agent (with memory shown in context)."""
    from pilot_experiment.pipeline import build_agent

    return lambda: build_agent(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the chain or final-only condition with a live model.")
    parser.add_argument("--mode", choices=["chain", "final-only"], required=True)
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--runs", type=int, default=1, help="How many times to repeat the condition.")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("No OPENAI_API_KEY found. Add it to .env in the project folder.")

    factory = gpt_pipeline_factory(args.model)
    run_fn = run_chain if args.mode == "chain" else run_final_only

    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    out_dir = Path("results") / batch_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(args.runs):
        result = run_fn(factory)
        result.update(
            run_index=i,
            mode=args.mode,
            model=args.model,
            agentdojo_version=version("agentdojo"),
            benchmark_version="v1.2.2 (workspace_memory)",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        rows.append(result)
        print(f"run {i}: stage1={result['stage1']} stage2={result['stage2']} "
              f"stage3={result['stage3']} chain_success={result['chain_success']} "
              f"first_sent={result['code_first_sent_stage']}")

    (out_dir / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()