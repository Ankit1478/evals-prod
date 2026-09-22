"""Step 1: define exactly what the LLM judge evaluates.

This module contains no model or Azure integration. It is the stable contract that
future prompts, API calls, datasets, and metrics will use.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
# Enums keep values consistent across datasets, prompts, API responses, and
# reports. For example, using EvaluationMode.PAIRWISE prevents spelling variants
# such as "pair-wise" from entering stored evaluation data.


class Criterion(str, Enum):
    """Independent dimensions used to judge answer quality."""

    CORRECTNESS = "correctness"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"


class Decision(str, Enum):
    """Final decision for pointwise and binary evaluations."""

    PASS = "PASS"
    FAIL = "FAIL"


class EvaluationMode(str, Enum):
    """Evaluation formats highlighted by the survey's quick-practice flow."""

    SCORE = "score"
    BINARY = "binary"
    PAIRWISE = "pairwise"


class PairwiseDecision(str, Enum):
    """Result of comparing two anonymous candidates."""

    A_WINS = "A_WINS"
    B_WINS = "B_WINS"
    TIE = "TIE"


class ReferencePolicy(str, Enum):
    """Controls whether a judge may or must use a reference answer."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    REFERENCE_FREE = "reference_free"


class TaskComplexity(str, Enum):
    """Reasoning capabilities that may be needed from the judge model."""

    FACTUAL_VERIFICATION = "factual_verification"
    INSTRUCTION_FOLLOWING = "instruction_following"
    MULTI_STEP_REASONING = "multi_step_reasoning"
    SUBJECTIVE_PREFERENCE = "subjective_preference"


class ReliabilityMetric(str, Enum):
    """Measurements used later to decide whether the judge is trustworthy."""

    EXACT_HUMAN_AGREEMENT = "exact_human_agreement"
    COHENS_KAPPA = "cohens_kappa"
    SCORE_CORRELATION = "score_correlation"
    REPEAT_CONSISTENCY = "repeat_consistency"
    POSITION_FLIP_RATE = "position_flip_rate"


class ExampleLabel(str, Enum):
    """Human-labelled case types needed to calibrate and challenge the judge."""

    GOOD = "good"
    BAD = "bad"
    BORDERLINE = "borderline"
    PAIRWISE_TIE = "pairwise_tie"


class CriterionDefinition(BaseModel):
    """Plain-language meaning and business weight of one quality dimension."""

    # frozen=True prevents accidental configuration changes while an evaluation
    # run is in progress.
    model_config = ConfigDict(frozen=True)

    description: str
    weight: float = Field(gt=0, le=1)


# Weights add to 1.0. Correctness receives the largest weight because a polished
# but factually wrong answer should not pass merely through good presentation.
CRITERIA: Dict[Criterion, CriterionDefinition] = {
    Criterion.CORRECTNESS: CriterionDefinition(
        description=(
            "The answer is factually consistent with the supplied reference "
            "and contains no material errors."
        ),
        weight=0.40,
    ),
    Criterion.RELEVANCE: CriterionDefinition(
        description="The answer directly addresses the user's question.",
        weight=0.25,
    ),
    Criterion.COMPLETENESS: CriterionDefinition(
        description="The answer covers the important information needed by the user.",
        weight=0.20,
    ),
    Criterion.CLARITY: CriterionDefinition(
        description="The answer is understandable, precise, and well organized.",
        weight=0.15,
    ),
}


class TaskDefinition(BaseModel):
    """Stable policy describing what this judge is allowed and expected to do.

    This is project configuration, not an individual evaluation request. A single
    request is represented later by ``EvaluationInput``.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    objective: str
    score_min: int
    score_max: int
    pass_threshold: float
    minimum_critical_score: int
    supported_modes: List[EvaluationMode]
    preferred_mode: EvaluationMode
    supported_reference_policies: List[ReferencePolicy]
    default_reference_policy: ReferencePolicy
    complexity: List[TaskComplexity]
    human_evaluation_protocol: List[str]
    reliability_metrics: List[ReliabilityMetric]
    reliability_thresholds: Dict[ReliabilityMetric, float]
    excluded_dimensions: List[str]

    @model_validator(mode="after")
    def validate_task_configuration(self) -> "TaskDefinition":
        """Reject internally contradictory configuration during application startup."""

        if not self.score_min <= self.pass_threshold <= self.score_max:
            raise ValueError("Pass threshold must be inside the scoring range")
        if not self.score_min <= self.minimum_critical_score <= self.score_max:
            raise ValueError("Critical score must be inside the scoring range")
        if self.preferred_mode not in self.supported_modes:
            raise ValueError("Preferred mode must be one of the supported modes")
        if self.default_reference_policy not in self.supported_reference_policies:
            raise ValueError("Default reference policy must be supported")
        return self


# InputOrderPolicy controls how Answer A and Answer B are shown to the judge to
# prevent bias.
#
#   - blind_candidate_identity: Hide which model produced each answer.
#   - randomize_initial_order: Randomly decide which answer appears first.
#   - evaluate_swapped_order: Evaluate again after swapping A and B.
#
# Example:
#   First test:  A = GPT answer, B = Terra answer
#   Second test: A = Terra answer, B = GPT answer
#
# If the winner changes incorrectly, the judge may have position bias.
# frozen=True means the policy cannot accidentally change during an evaluation.
class InputOrderPolicy(BaseModel):
    """Controls identity and position effects during pairwise evaluation.

    Evaluating A/B and then B/A helps reveal whether the judge simply prefers the
    first or second position instead of the better answer.
    """

    model_config = ConfigDict(frozen=True)

    blind_candidate_identity: bool
    randomize_initial_order: bool
    evaluate_swapped_order: bool

# We are using this reliable exam policy to test whether the judge is performing correctly or not. One human marks all the answers, and then we will perform the judge on the same dataset and check whether the judge is performing accurately or not. After that, we will push to production. 
class ReliableExamplePolicy(BaseModel):
    """Minimum kinds of human-reviewed examples needed before calibration.

    Borderline and tie cases are required because only using obvious good/bad
    examples makes a judge appear more reliable than it will be in production.
    """

    model_config = ConfigDict(frozen=True)

    required_labels: List[ExampleLabel]
    minimum_examples_per_label: int = Field(ge=1)


# This is the actual task configuration used by the project. Pairwise comparison
# is preferred for relative quality judgments, while score and binary modes remain
# available for diagnostics and simple gates.
TASK_DEFINITION = TaskDefinition(
    name="answer_quality",
    objective=(
        "Evaluate one AI-generated answer or compare two answers against a user "
        "question and the evidence permitted by the reference policy."
    ),
    score_min=1,
    score_max=5,
    pass_threshold=3.5,
    minimum_critical_score=3,
    supported_modes=[
        EvaluationMode.SCORE,
        EvaluationMode.BINARY,
        EvaluationMode.PAIRWISE,
    ],
    preferred_mode=EvaluationMode.PAIRWISE,
    supported_reference_policies=[
        ReferencePolicy.REQUIRED,
        ReferencePolicy.OPTIONAL,
        ReferencePolicy.REFERENCE_FREE,
    ],
    default_reference_policy=ReferencePolicy.REQUIRED,
    complexity=[
        TaskComplexity.FACTUAL_VERIFICATION,
        TaskComplexity.INSTRUCTION_FOLLOWING,
        TaskComplexity.MULTI_STEP_REASONING,
    ],
    human_evaluation_protocol=[
        "Review the question and only the evidence allowed by the reference policy.",
        "Judge each rubric dimension independently before making a final decision.",
        "Keep candidate identities hidden and do not infer which model produced them.",
        "For pairwise cases, select a winner only when the quality difference is meaningful.",
        "Record a tie when neither answer is meaningfully better.",
        "Resolve human-label disagreements through an independent adjudicator.",
    ],
    reliability_metrics=[
        ReliabilityMetric.EXACT_HUMAN_AGREEMENT,
        ReliabilityMetric.COHENS_KAPPA,
        ReliabilityMetric.SCORE_CORRELATION,
        ReliabilityMetric.REPEAT_CONSISTENCY,
        ReliabilityMetric.POSITION_FLIP_RATE,
    ],
    # A universal threshold would be arbitrary. We will set these values only
    # after measuring performance against a human-labelled baseline dataset.
    reliability_thresholds={},
    excluded_dimensions=[
        "personal writing-style preference",
        "answer length by itself",
        "model identity",
        "generation cost",
        "response latency",
    ],
)


# Do not expose provider/model names as candidate labels. Only the neutral labels
# A and B will be shown to the judge.
INPUT_ORDER_POLICY = InputOrderPolicy(
    blind_candidate_identity=True,
    randomize_initial_order=True,
    evaluate_swapped_order=True,
)


# At least two examples of each class avoids calibrating from a single anecdote.
RELIABLE_EXAMPLE_POLICY = ReliableExamplePolicy(
    required_labels=[
        ExampleLabel.GOOD,
        ExampleLabel.BAD,
        ExampleLabel.BORDERLINE,
        ExampleLabel.PAIRWISE_TIE,
    ],
    minimum_examples_per_label=2,
)


class EvaluationInput(BaseModel):
    """One evaluation case and all evidence the judge is permitted to use.

    ``candidate_answer`` is candidate A in pairwise mode. ``candidate_b`` is only
    provided for pairwise mode. The neutral field names help keep model identity
    hidden from the judge.
    """

    case_id: str = Field(min_length=1)
    mode: EvaluationMode = EvaluationMode.SCORE
    reference_policy: ReferencePolicy = ReferencePolicy.REQUIRED
    question: str = Field(min_length=1)
    reference_answer: Optional[str] = Field(default=None, min_length=1)
    candidate_answer: str = Field(min_length=1)
    candidate_b: Optional[str] = Field(default=None, min_length=1)
    context: Optional[str] = None

    @model_validator(mode="after")
    def validate_mode_and_reference_policy(self) -> "EvaluationInput":
        """Make invalid mode/reference combinations impossible to send to a model."""

        if self.reference_policy == ReferencePolicy.REQUIRED and not self.reference_answer:
            raise ValueError("A reference answer is required by the selected policy")
        if self.reference_policy == ReferencePolicy.REFERENCE_FREE and self.reference_answer:
            raise ValueError("Reference-free evaluation must not include a reference answer")
        if self.mode == EvaluationMode.PAIRWISE and not self.candidate_b:
            raise ValueError("Pairwise evaluation requires candidate_b")
        if self.mode != EvaluationMode.PAIRWISE and self.candidate_b:
            raise ValueError("candidate_b is only valid for pairwise evaluation")
        return self


class CriterionScore(BaseModel):
    """A score plus short, observable evidence for that score.

    Evidence is deliberately short: we need an auditable justification, not the
    model's private chain of thought.
    """

    model_config = ConfigDict(extra="forbid")

    criterion: Criterion
    score: int = Field(ge=1, le=5)
    evidence: str = Field(min_length=1, max_length=500)


class EvaluationResult(BaseModel):
    """Validated pointwise output with an application-owned final decision.

    The model supplies dimension scores and evidence. Python computes the final
    decision so the result cannot say PASS while its scores require FAIL.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    scores: List[CriterionScore] = Field(min_length=4, max_length=4)
    summary: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def require_every_criterion_once(self) -> "EvaluationResult":
        """Require one—and only one—score for every configured dimension."""

        actual = [item.criterion for item in self.scores]
        expected = set(Criterion)
        if len(set(actual)) != len(actual):
            raise ValueError("Each criterion must appear exactly once; duplicate found")
        if set(actual) != expected:
            missing = sorted(item.value for item in expected - set(actual))
            raise ValueError(f"Scores are missing criteria: {missing}")
        return self

    @computed_field
    @property
    def weighted_score(self) -> float:
        """Combine the four dimension scores using the weights in ``CRITERIA``."""

        score_by_criterion = {item.criterion: item.score for item in self.scores}
        total = sum(
            score_by_criterion[criterion] * definition.weight
            for criterion, definition in CRITERIA.items()
        )
        return round(total, 2)

    @computed_field
    @property
    def decision(self) -> Decision:
        """Apply both the overall threshold and critical-dimension safety gates."""

        score_by_criterion = {item.criterion: item.score for item in self.scores}
        # A high clarity/completeness score cannot compensate for an incorrect or
        # irrelevant answer. Both critical criteria must independently score >= 3.
        critical_scores_pass = all(
            score_by_criterion[criterion] >= TASK_DEFINITION.minimum_critical_score
            for criterion in (Criterion.CORRECTNESS, Criterion.RELEVANCE)
        )
        if critical_scores_pass and self.weighted_score >= TASK_DEFINITION.pass_threshold:
            return Decision.PASS
        return Decision.FAIL


class BinaryEvaluationResult(BaseModel):
    """Structured result for a direct Yes/No or Pass/Fail evaluation.

    Use this when the evaluation question has a clear binary outcome and detailed
    dimension scores would add little value.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    decision: Decision
    evidence: str = Field(min_length=1, max_length=500)


class PairwiseEvaluationResult(BaseModel):
    """Structured result for relative comparison with an explicit tie option.

    A tie is a real outcome, not an error: forcing a winner can create false
    preferences when both answers have equivalent quality.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    decision: PairwiseDecision
    evidence: str = Field(min_length=1, max_length=500)
