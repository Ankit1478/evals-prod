import tempfile
import unittest
from pathlib import Path

from llm_judge.azure_client import TokenUsage
from llm_judge.contracts import (
    BinaryEvaluationResult,
    Criterion,
    CriterionScore,
    EvaluationMode,
    EvaluationResult,
)
from llm_judge.dataset import ReviewStatus
from llm_judge.dataset_runner import CaseRunResult, CaseRunStatus
from llm_judge.multi_judge import JudgeModel, ModelJudgment, TwoModelJudgeResult
from llm_judge.reliability import (
    calculate_reliability,
    cohens_kappa,
    load_case_results,
    pearson_correlation,
)


def binary_judgment(case_id: str, model: JudgeModel, decision: str) -> ModelJudgment:
    return ModelJudgment(
        model=model,
        deployment=model.value,
        usage=TokenUsage(total_tokens=10),
        result=BinaryEvaluationResult(
            case_id=case_id,
            decision=decision,
            evidence="Evidence.",
        ),
    )


def completed_binary(
    case_id: str,
    human: str,
    terra: str,
    luna: str,
    aggregate=None,
    *,
    review_status: ReviewStatus = ReviewStatus.HUMAN_REVIEWED,
) -> CaseRunResult:
    agreement = terra == luna
    if aggregate is None and agreement:
        aggregate = terra
    judge_result = TwoModelJudgeResult(
        case_id=case_id,
        mode=EvaluationMode.BINARY,
        judgments=[
            binary_judgment(case_id, JudgeModel.TERRA, terra),
            binary_judgment(case_id, JudgeModel.LUNA, luna),
        ],
        agreement=agreement,
        aggregate_decision=aggregate,
        requires_human_review=not agreement,
    )
    return CaseRunResult(
        case_id=case_id,
        mode=EvaluationMode.BINARY,
        review_status=review_status,
        human_decision=human,
        status=CaseRunStatus.COMPLETED,
        judge_result=judge_result,
        matches_human=aggregate == human if aggregate is not None else None,
        requires_human_review=not agreement,
    )


def failed_case(case_id: str) -> CaseRunResult:
    return CaseRunResult(
        case_id=case_id,
        mode=EvaluationMode.BINARY,
        review_status=ReviewStatus.HUMAN_REVIEWED,
        human_decision="PASS",
        status=CaseRunStatus.ERROR,
        requires_human_review=True,
        error_type="JudgeClientError",
        error_message="Case evaluation failed; inspect controlled logs",
    )


def score_judgment(
    case_id: str,
    model: JudgeModel,
    values,
) -> ModelJudgment:
    return ModelJudgment(
        model=model,
        deployment=model.value,
        usage=TokenUsage(total_tokens=10),
        result=EvaluationResult(
            case_id=case_id,
            scores=[
                CriterionScore(
                    criterion=criterion,
                    score=score,
                    evidence="Evidence.",
                )
                for criterion, score in zip(Criterion, values)
            ],
            summary="Summary.",
        ),
    )


def completed_score(case_id: str) -> CaseRunResult:
    human_values = [1, 2, 3, 4]
    terra_values = [1, 2, 3, 4]
    luna_values = [4, 3, 2, 1]
    average_values = [2.5, 2.5, 2.5, 2.5]
    human_scores = {
        criterion: score for criterion, score in zip(Criterion, human_values)
    }
    judge_result = TwoModelJudgeResult(
        case_id=case_id,
        mode=EvaluationMode.SCORE,
        judgments=[
            score_judgment(case_id, JudgeModel.TERRA, terra_values),
            score_judgment(case_id, JudgeModel.LUNA, luna_values),
        ],
        agreement=True,
        aggregate_decision="FAIL",
        average_scores={
            criterion: score for criterion, score in zip(Criterion, average_values)
        },
        average_weighted_score=2.5,
        requires_human_review=False,
    )
    return CaseRunResult(
        case_id=case_id,
        mode=EvaluationMode.SCORE,
        review_status=ReviewStatus.HUMAN_REVIEWED,
        human_decision="FAIL",
        human_scores=human_scores,
        status=CaseRunStatus.COMPLETED,
        judge_result=judge_result,
        matches_human=True,
        requires_human_review=False,
    )


class ReliabilityTests(unittest.TestCase):
    def test_cohens_kappa_corrects_for_chance(self) -> None:
        value = cohens_kappa(
            ["PASS", "PASS", "FAIL", "FAIL"],
            ["PASS", "FAIL", "FAIL", "FAIL"],
        )

        self.assertEqual(value, 0.5)
        self.assertEqual(
            cohens_kappa(["PASS", "FAIL"], ["PASS", "FAIL"]),
            1.0,
        )
        self.assertIsNone(cohens_kappa(["PASS", "PASS"], ["PASS", "PASS"]))

    def test_pearson_correlation_handles_direction_and_constant_values(self) -> None:
        self.assertEqual(pearson_correlation([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertEqual(pearson_correlation([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertIsNone(pearson_correlation([1, 1, 1], [2, 3, 4]))

    def test_report_calculates_model_human_and_failure_metrics(self) -> None:
        results = [
            completed_binary("one", "PASS", "PASS", "PASS"),
            completed_binary("two", "FAIL", "FAIL", "PASS"),
            failed_case("three"),
        ]

        report = calculate_reliability(results)

        self.assertEqual(report.total_cases, 3)
        self.assertEqual(report.completed_cases, 2)
        self.assertEqual(report.failure_rate, 0.3333)
        self.assertEqual(report.human_review_rate, 0.6667)
        self.assertEqual(report.model_disagreements, 1)
        self.assertEqual(report.model_disagreement_rate, 0.5)
        self.assertEqual(report.terra_vs_human.agreement_rate, 1.0)
        self.assertEqual(report.terra_vs_human.cohens_kappa, 1.0)
        self.assertEqual(report.luna_vs_human.agreement_rate, 0.5)
        self.assertEqual(report.luna_vs_human.cohens_kappa, 0.0)
        self.assertEqual(report.aggregate_vs_human.evaluated_cases, 1)
        self.assertEqual(report.aggregate_vs_human.abstentions, 1)
        self.assertEqual(report.terra_vs_luna.agreement_rate, 0.5)
        self.assertEqual(report.by_mode[EvaluationMode.BINARY].total_cases, 3)
        self.assertEqual(report.by_mode[EvaluationMode.BINARY].completed_cases, 2)
        self.assertTrue(any("Fewer than 30" in item for item in report.warnings))

    def test_score_report_calculates_each_models_correlation(self) -> None:
        report = calculate_reliability([completed_score("score-one")])

        self.assertEqual(report.score_correlation.score_cases, 1)
        self.assertEqual(report.score_correlation.score_pairs, 4)
        self.assertEqual(report.score_correlation.terra_pearson, 1.0)
        self.assertEqual(report.score_correlation.luna_pearson, -1.0)
        self.assertIsNone(report.score_correlation.aggregate_pearson)
        self.assertTrue(
            any("Score correlation is undefined" in item for item in report.warnings)
        )

    def test_draft_results_add_production_warning(self) -> None:
        result = completed_binary(
            "draft",
            "PASS",
            "PASS",
            "PASS",
            review_status=ReviewStatus.DRAFT,
        )

        report = calculate_reliability([result])

        self.assertTrue(any("Draft human labels" in item for item in report.warnings))

    def test_jsonl_loader_validates_lines_and_duplicate_case_ids(self) -> None:
        result = completed_binary("duplicate", "PASS", "PASS", "PASS")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(
                result.model_dump_json() + "\n" + result.model_dump_json() + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate case IDs"):
                load_case_results(path)


if __name__ == "__main__":
    unittest.main()
