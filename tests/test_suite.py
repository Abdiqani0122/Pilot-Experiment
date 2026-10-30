"""Checks our own suite loads, has the memory tools, and has three fillable attack slots. No LLM."""

from pilot_experiment.suite import task_suite


def clean_env():
    """The dollhouse with no attack: every slot keeps its harmless default text."""
    return task_suite.load_and_inject_default_environment({})


def test_suite_has_workspace_and_memory_tools():
    names = [t.name for t in task_suite.tools]
    assert len(names) == 26                 # 24 AgentDojo workspace tools + 2 memory tools
    assert "write_memory" in names
    assert "read_memory" in names
    assert "send_email" in names            # a normal AgentDojo tool is still there


def test_clean_environment_is_the_full_workspace_plus_memory():
    e = clean_env()
    assert type(e).__name__ == "MemoryWorkspaceEnvironment"
    assert e.memory.entries == []           # fresh memory is empty
    assert len(e.inbox.emails) == 33        # 31 original + 2 new (stage 1 and stage 3 emails)
    assert len(e.cloud_drive.files) == 27   # 26 original + 1 new (stage 2 policy document)


def test_three_attack_slots_exist_with_harmless_defaults():
    defaults = task_suite.get_injection_vector_defaults()
    assert len(defaults) == 19              # 16 AgentDojo slots + our 3
    for key in ["stage1_email_injection", "stage2_policy_injection", "stage3_email_injection"]:
        assert key in defaults
    # with no attack, the slots hold their harmless default text, not attack text
    e = clean_env()
    assert "all-hands meeting" in e.inbox.emails["34"].body
    assert "password manager" in e.cloud_drive.files["26"].content
    assert "routine security notice" in e.inbox.emails["35"].body


def test_each_slot_can_be_filled_independently():
    e = task_suite.load_and_inject_default_environment({
        "stage1_email_injection": "TEST-S1",
        "stage2_policy_injection": "TEST-S2",
        "stage3_email_injection": "TEST-S3",
    })
    assert e.inbox.emails["34"].body == "TEST-S1"
    assert e.cloud_drive.files["26"].content == "TEST-S2"
    assert e.inbox.emails["35"].body == "TEST-S3"
    # a normal email is untouched
    assert e.inbox.emails["0"].subject == "Birthday Party"
