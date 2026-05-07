import pytest
from pydantic import ValidationError

from data_structure.schemas import RecoveryStep


def test_recovery_step_contains_react_style_fields_and_is_schema_validated():
    step = RecoveryStep(
        iteration=1,
        reasoning="too few docs",
        action="refine_query",
        action_input={"previous_query": "민법"},
        observation={"refined_query": "민법 손해배상"},
        evidence_delta=0,
        next_query="민법 손해배상",
        selected_collection="law_civil",
        source="llm",
    ).model_dump(mode="json")

    for key in [
        "iteration",
        "reasoning",
        "action",
        "action_input",
        "observation",
        "evidence_delta",
        "next_query",
        "selected_collection",
        "source",
    ]:
        assert key in step
    assert step["action"] == "refine_query"
    assert RecoveryStep.model_validate(step).action == "refine_query"


def test_recovery_step_rejects_unknown_action():
    with pytest.raises(ValidationError):
        RecoveryStep(
            iteration=1,
            reasoning="invalid action should fail",
            action="random_action",
            action_input={},
            observation={},
        )
