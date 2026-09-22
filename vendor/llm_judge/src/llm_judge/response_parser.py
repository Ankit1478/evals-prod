"""Step 6: turn an untrusted raw model response into a trusted result."""

from typing import Dict, Type, Union

from pydantic import BaseModel, ValidationError

from .azure_client import RawJudgeResponse
from .contracts import (
    BinaryEvaluationResult,
    EvaluationInput,
    EvaluationMode,
    EvaluationResult,
    PairwiseEvaluationResult,
)


ParsedJudgeResult = Union[
    BinaryEvaluationResult,
    PairwiseEvaluationResult,
    EvaluationResult,
]


class JudgeResponseValidationError(ValueError):
    """Raised when a judge response cannot be trusted by the application."""


class JudgeRefusalError(JudgeResponseValidationError):
    """Raised when a judge model refuses to evaluate a case."""


_RESULT_MODEL_BY_MODE: Dict[EvaluationMode, Type[BaseModel]] = {
    EvaluationMode.BINARY: BinaryEvaluationResult,
    EvaluationMode.PAIRWISE: PairwiseEvaluationResult,
    EvaluationMode.SCORE: EvaluationResult,
}


def parse_judge_response(
    response: RawJudgeResponse,
    evaluation_input: EvaluationInput,
) -> ParsedJudgeResult:
    """Validate judge JSON using the Pydantic class for the requested mode.

    Raw model text is untrusted. This function checks the mode, refusal state,
    JSON structure, field types and values, and case identity before returning a
    result that the rest of the application may safely use.
    """

    if response.mode != evaluation_input.mode:
        raise JudgeResponseValidationError(
            "Judge response mode does not match the evaluation input"
        )

    if response.refusal:
        # Do not treat a refusal as FAIL: the model did not actually judge the case.
        raise JudgeRefusalError("Judge refused to evaluate the case")

    if not response.content:
        raise JudgeResponseValidationError("Judge response has no JSON content")

    result_model = _RESULT_MODEL_BY_MODE[evaluation_input.mode]
    try:
        parsed = result_model.model_validate_json(response.content)
    except (ValidationError, ValueError) as error:
        # Keep raw model output out of the public error message because it may
        # contain user data. The original error is still available via chaining.
        raise JudgeResponseValidationError(
            f"Judge returned invalid {evaluation_input.mode.value} result JSON"
        ) from error

    if parsed.case_id != evaluation_input.case_id:
        raise JudgeResponseValidationError(
            "Judge response case_id does not match the evaluation input"
        )

    return parsed  # type: ignore[return-value]
