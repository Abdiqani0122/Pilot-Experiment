"""Our own AgentDojo task suite: Emma's workspace + long-term memory.

Built the same way as AgentDojo's official example
(examples/counter_benchmark/suites/counter/task_suite.py):

    TaskSuite(name, environment class, tools, data folder)
"""

from pathlib import Path

# Import our memory file first: it loads AgentDojo in the right order
# (same circular-import fix as in memory.py).
from pilot_experiment.memory import MEMORY_TOOLS, MemoryWorkspaceEnvironment

from agentdojo.default_suites.v1.workspace.task_suite import TOOLS as WORKSPACE_TOOLS
from agentdojo.functions_runtime import make_function
from agentdojo.task_suite import TaskSuite

# This file is src/pilot_experiment/suite.py, so the project folder is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FOLDER = PROJECT_ROOT / "data" / "suites" / "workspace_memory"

# AgentDojo's 24 workspace tools + our 2 memory tools
ALL_TOOLS = WORKSPACE_TOOLS + MEMORY_TOOLS

task_suite = TaskSuite(
    "workspace_memory",                            # 1. the suite's name
    MemoryWorkspaceEnvironment,                    # 2. the environment: workspace + memory
    [make_function(tool) for tool in ALL_TOOLS],   # 3. the tools GPT can call
    DATA_FOLDER,                                   # 4. where the dollhouse data lives
)
