from cortex.api.app import checkpoint_serializer
from cortex.workflow import Intent


def test_current_workflow_intent_round_trip():
    serializer = checkpoint_serializer()

    encoded = serializer.dumps_typed(Intent.KNOWLEDGE_QUERY)

    assert serializer.loads_typed(encoded) is Intent.KNOWLEDGE_QUERY


def test_legacy_workflow_intent_round_trip(monkeypatch):
    serializer = checkpoint_serializer()
    with monkeypatch.context() as patch:
        patch.setattr(Intent, "__module__", "cortex.workflow")
        encoded = serializer.dumps_typed(Intent.KNOWLEDGE_QUERY)

    assert serializer.loads_typed(encoded) is Intent.KNOWLEDGE_QUERY
