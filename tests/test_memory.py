"""Checks that the memory and its tools work, without any LLM. Run: uv run pytest -v"""

from agentdojo.functions_runtime import FunctionsRuntime, make_function
from agentdojo.task_suite.load_suites import get_suite

from pilot_experiment.memory import MEMORY_TOOLS, add_empty_memory


def fresh_env():
    """Emma's normal workspace (no attack) + an empty notebook."""
    workspace = get_suite("v1.2.2", "workspace").load_and_inject_default_environment({})
    return add_empty_memory(workspace)


def runtime():
    """The same kind of runtime AgentDojo uses to run the agent's tool calls."""
    return FunctionsRuntime([make_function(tool) for tool in MEMORY_TOOLS])


def test_memory_starts_empty():
    env = fresh_env()
    assert env.memory.entries == []
    assert len(env.inbox.emails) == 31


def test_write_then_read():
    env, rt = fresh_env(), runtime()

    # run_function is the exact path AgentDojo uses when GPT calls a tool
    result, error = rt.run_function(env, "write_memory", {"content": "Emma prefers short summaries."})
    assert error is None
    assert result == "Saved to memory (entry 0)."
    assert env.memory.entries[0].content == "Emma prefers short summaries."

    result, error = rt.run_function(env, "read_memory", {})
    assert error is None
    assert result[0].content == "Emma prefers short summaries."


def test_read_empty_memory():
    result, error = runtime().run_function(fresh_env(), "read_memory", {})
    assert error is None
    assert result == "The memory is empty."


def test_llm_cannot_see_or_pass_the_memory_object():
    tools = runtime().functions
    assert list(tools["write_memory"].parameters.model_json_schema()["properties"]) == ["content"]
    assert tools["read_memory"].parameters.model_json_schema().get("properties", {}) == {}
    assert tools["write_memory"].description.startswith("Saves a piece of information")


def test_before_and_after_photos_see_the_memory_change():
    """The stage checkers will compare exactly these two snapshots."""
    env, rt = fresh_env(), runtime()
    before = env.model_copy(deep=True)  # the same snapshot AgentDojo takes before a run

    rt.run_function(env, "write_memory", {"content": "Emma's assistant is mark.black-2134@gmail.com"})

    assert before.memory.entries == []
    assert len(env.memory.entries) == 1


def test_two_fresh_environments_do_not_share_memory():
    """A control run must never see a chain run's notes."""
    env_a, env_b, rt = fresh_env(), fresh_env(), runtime()
    rt.run_function(env_a, "write_memory", {"content": "only in A"})
    assert env_b.memory.entries == []


def test_memory_carries_over_when_the_same_environment_is_reused():
    """This is how the chain works: stage 2 gets the environment stage 1 left behind."""
    env, rt = fresh_env(), runtime()
    rt.run_function(env, "write_memory", {"content": "written in stage 1"})  # stage 1
    result, _ = rt.run_function(env, "read_memory", {})                    # stage 2, same env
    assert result[0].content == "written in stage 1"
