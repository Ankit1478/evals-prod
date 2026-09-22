"""Step 7: evaluate one case with both Terra and Luna judges."""

from enum import Enum
from typing import Dict, List, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .azure_client import AzureJudgeClient, RawJudgeResponse, TokenUsage
from .contracts import (
    CRITERIA,
    TASK_DEFINITION,
    Criterion,
    Decision,
    EvaluationInput,
    EvaluationMode,
    EvaluationResult,
)
from .prompt_builder import JudgePrompt, build_judge_prompt
from .response_parser import ParsedJudgeResult, parse_judge_response
from .settings import AzureJudgeSettings


class JudgeModel(str, Enum):
    """The two Azure deployments allowed to act as judges in this project."""

    TERRA = "gpt-5.6-terra"
    LUNA = "gpt-5.6-luna"


class JudgeTransport(Protocol):
    """Small interface shared by the real Azure client and test fakes."""

    settings: AzureJudgeSettings

    def evaluate(self, prompt: JudgePrompt) -> RawJudgeResponse:
        ...


class ModelJudgment(BaseModel):
    """One model's validated judgment, kept separately for auditability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: JudgeModel
    deployment: str
    resolved_model: Optional[str] = None
    response_id: Optional[str] = None
    request_id: Optional[str] = None
    usage: TokenUsage
    result: ParsedJudgeResult


class TwoModelJudgeResult(BaseModel):
    """Combined result without hiding disagreement between Terra and Luna."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    mode: EvaluationMode
    judgments: List[ModelJudgment] = Field(min_length=2, max_length=2)
    agreement: bool
    aggregate_decision: Optional[str] = None
    average_scores: Optional[Dict[Criterion, float]] = None
    average_weighted_score: Optional[float] = None
    requires_human_review: bool

    @model_validator(mode="after")
    def require_one_result_from_each_model(self) -> "TwoModelJudgeResult":
        models = [judgment.model for judgment in self.judgments]
        if set(models) != {JudgeModel.TERRA, JudgeModel.LUNA}:
            raise ValueError("Result requires exactly one Terra and one Luna judgment")
        if any(
            judgment.result.case_id != self.case_id for judgment in self.judgments
        ):
            raise ValueError("Every model judgment must match the combined case_id")
        return self


class TwoModelJudge:
    """Run the same blinded prompt through Terra and Luna and aggregate safely."""

    def __init__(
        self,
        terra_client: JudgeTransport,
        luna_client: JudgeTransport,
    ) -> None:
        expected = {
            JudgeModel.TERRA: terra_client,
            JudgeModel.LUNA: luna_client,
        }
        for model, client in expected.items():
            if client.settings.deployment != model.value:
                raise ValueError(
                    f"{model.name} client must use deployment {model.value}"
                )
        self._clients = expected

    @classmethod
    def from_settings(cls, settings: AzureJudgeSettings) -> "TwoModelJudge":
        """Create both clients from one endpoint/key configuration."""

        terra_settings = settings.model_copy(
            update={"deployment": JudgeModel.TERRA.value}
        )
        luna_settings = settings.model_copy(
            update={"deployment": JudgeModel.LUNA.value}
        )
        return cls(
            terra_client=AzureJudgeClient(terra_settings),
            luna_client=AzureJudgeClient(luna_settings),
        )

    def evaluate(
        self,
        evaluation_input: EvaluationInput,
        *,
        include_examples: bool = True,
        example_limit: int = 3,
    ) -> TwoModelJudgeResult:
        """Judge one case twice, validate each output, then combine the decisions."""

        # Both models receive the exact same prompt, so judge model is the only
        # intentional variable in their comparison.
        prompt = build_judge_prompt(
            evaluation_input,
            include_examples=include_examples,
            example_limit=example_limit,
        )
        judgments = []
        for model in (JudgeModel.TERRA, JudgeModel.LUNA):
            judgments.append(self.evaluate_prompt(model, prompt, evaluation_input))

        return _aggregate(evaluation_input, judgments)

    def evaluate_prompt(
        self,
        model: JudgeModel,
        prompt: JudgePrompt,
        evaluation_input: EvaluationInput,
    ) -> ModelJudgment:
        """Evaluate one already-built prompt with one selected judge model.

        Step 10 uses this method so every repeat receives the exact same prompt
        object. It also keeps provider access and response validation in one place.
        """

        if prompt.mode != evaluation_input.mode:
            raise ValueError("Prompt mode must match the evaluation input")
        raw_response = self._clients[model].evaluate(prompt)
        parsed_result = parse_judge_response(raw_response, evaluation_input)
        return ModelJudgment(
            model=model,
            deployment=raw_response.deployment,
            resolved_model=raw_response.model,
            response_id=raw_response.response_id,
            request_id=raw_response.request_id,
            usage=raw_response.usage,
            result=parsed_result,
        )


def _aggregate(
    evaluation_input: EvaluationInput,
    judgments: List[ModelJudgment],
) -> TwoModelJudgeResult:
    """Combine two validated results while making ties visible."""

    decisions = [judgment.result.decision.value for judgment in judgments]
    agreement = decisions[0] == decisions[1]

    if evaluation_input.mode != EvaluationMode.SCORE:
        # Two different votes form a 1-1 tie, not a majority decision.
        return TwoModelJudgeResult(
            case_id=evaluation_input.case_id,
            mode=evaluation_input.mode,
            judgments=judgments,
            agreement=agreement,
            aggregate_decision=decisions[0] if agreement else None,
            requires_human_review=not agreement,
        )

    score_results = [judgment.result for judgment in judgments]
    if not all(isinstance(result, EvaluationResult) for result in score_results):
        raise TypeError("Score mode requires EvaluationResult from both judges")

    average_scores = {
        criterion: round(
            sum(
                next(
                    item.score
                    for item in result.scores
                    if item.criterion == criterion
                )
                for result in score_results
            )
            / len(score_results),
            2,
        )
        for criterion in Criterion
    }
    weighted_score = round(
        sum(
            average_scores[criterion] * definition.weight
            for criterion, definition in CRITERIA.items()
        ),
        2,
    )
    critical_scores_pass = all(
        average_scores[criterion] >= TASK_DEFINITION.minimum_critical_score
        for criterion in (Criterion.CORRECTNESS, Criterion.RELEVANCE)
    )
    aggregate_decision = (
        Decision.PASS.value
        if critical_scores_pass
        and weighted_score >= TASK_DEFINITION.pass_threshold
        else Decision.FAIL.value
    )
    return TwoModelJudgeResult(
        case_id=evaluation_input.case_id,
        mode=evaluation_input.mode,
        judgments=judgments,
        agreement=agreement,
        aggregate_decision=aggregate_decision,
        average_scores=average_scores,
        average_weighted_score=weighted_score,
        requires_human_review=not agreement,
    )
