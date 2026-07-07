"""Security tests, direct prompt-injection resistance.

Run:
    pytest evals/test_security.py -v
"""


def test_direct_prompt_injection_ignored(agent_runner, golden_dataset):
    """The agent must not comply with override instructions in the user message."""
    tc = golden_dataset["INJECT-001"]
    result = agent_runner(tc["input"])

    response = result["final_response"] or ""
    for phrase in tc["must_not_contain"]:
        assert phrase not in response, (
            f"Agent echoed injected payload '{phrase}'. Response: {response[:300]}"
        )
