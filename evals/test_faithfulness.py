"""Faithfulness tests — researcher must ground answers in the local KB.

Run:
    pytest evals/test_faithfulness.py -v
"""


def test_researcher_grounds_in_knowledge_base(agent_runner, golden_dataset):
    """Researcher should call search_knowledge_base and reflect retrieved facts."""
    tc = golden_dataset["FAITH-001"]
    result = agent_runner(tc["input"])

    tool_names = [c["name"] for c in result["tool_calls"]]
    assert "search_knowledge_base" in tool_names, (
        "Researcher must consult the knowledge base before answering. "
        f"Tools called: {tool_names}"
    )

    response_lower = result["final_response"].lower()
    matched = [kw for kw in tc["expected_keywords"] if kw.lower() in response_lower]
    assert matched, (
        f"Response missing all expected keywords {tc['expected_keywords']}. "
        f"Response: {result['final_response'][:300]}"
    )
