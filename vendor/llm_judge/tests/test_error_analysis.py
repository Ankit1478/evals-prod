import unittest

from llm_judge.azure_client import TokenUsage
from llm_judge.contracts import (
    BinaryEvaluationResult,
    EvaluationMode,
    PairwiseEvaluationResult,
)
from llm_judge.dataset import ReviewStatus
from llm_judge.dataset_runner import CaseRunResult, CaseRunStatus
from llm_judge.error_analysis import AnalysisTarget, calculate_error_analysis
from llm_judge.multi_judge import JudgeModel, ModelJudgment, TwoModelJudgeResult


def judgment(case_id, mode, model, decision):
    result_type = (
        PairwiseEvaluationResult
        if mode == EvaluationMode.PAIRWISE
        else BinaryEvaluationResult
    )
    return ModelJudgment(
        model=model,
        deployment=model.value,
        usage=TokenUsage(total_tokens=10),
        result=result_type(
            case_id=case_id,
            decision=decision,
            evidence="Evidence.",
        ),
    )


def completed_case(
    case_id,
    human,
    terra,
    luna,
    *,
    mode=EvaluationMode.BINARY,
    review_status=ReviewStatus.HUMAN_REVIEWED,
):
    agreement = terra == luna
    aggregate = terra if agreement else None
    combined = TwoModelJudgeResult(
        case_id=case_id,
        mode=mode,
        judgments=[
            judgment(case_id, mode, JudgeModel.TERRA, terra),
            judgment(case_id, mode, JudgeModel.LUNA, luna),
        ],
        agreement=agreement,
        aggregate_decision=aggregate,
        requires_human_review=not agreement,
    )
    return CaseRunResult(
        case_id=case_id,
        mode=mode,
        review_status=review_status,
        human_decision=human,
        status=CaseRunStatus.COMPLETED,
        judge_result=combined,
        matches_human=aggregate == human if aggregate else None,
        requires_human_review=not agreement,
    )


class ErrorAnalysisTests(unittest.TestCase):
    def test_binary_confusion_matrix_and_rates(self):
        results = [
            completed_case("tp", "PASS", "PASS", "PASS"),
            completed_case("fn", "PASS", "FAIL", "PASS"),
            completed_case("fp", "FAIL", "PASS", "FAIL"),
            completed_case("tn", "FAIL", "FAIL", "FAIL"),
        ]

        report = calculate_error_analysis(results, bootstrap_iterations=100)
        terra = report.analyses[AnalysisTarget.TERRA].pass_fail_overall

        self.assertEqual(terra.confusion_matrix.true_positive, 1)
        self.assertEqual(terra.confusion_matrix.true_negative, 1)
        self.assertEqual(terra.confusion_matrix.false_positive, 1)
        self.assertEqual(terra.confusion_matrix.false_negative, 1)
        self.assertEqual(terra.accuracy.value, 0.5)
        self.assertEqual(terra.precision.value, 0.5)
        self.assertEqual(terra.recall.value, 0.5)
        self.assertEqual(terra.f1.value, 0.5)
        self.assertEqual(terra.false_pass_rate.value, 0.5)
        self.assertEqual(terra.false_fail_rate.value, 0.5)
        self.assertEqual(terra.false_pass_rate.denominator, 2)
        self.assertEqual(terra.false_fail_rate.denominator, 2)
        self.assertEqual(terra.false_pass_case_ids, ["fp"])
        self.assertEqual(terra.false_fail_case_ids, ["fn"])

        luna = report.analyses[AnalysisTarget.LUNA].pass_fail_overall
        self.assertEqual(luna.accuracy.value, 1.0)
        self.assertEqual(luna.false_pass_rate.value, 0.0)
        self.assertEqual(luna.false_fail_rate.value, 0.0)

    def test_aggregate_disagreement_is_an_abstention(self):
        results = [
            completed_case("agree", "PASS", "PASS", "PASS"),
            completed_case("split", "FAIL", "PASS", "FAIL"),
        ]

        report = calculate_error_analysis(results, bootstrap_iterations=20)
        aggregate = report.analyses[AnalysisTarget.AGGREGATE].pass_fail_overall

        self.assertEqual(aggregate.available_cases, 2)
        self.assertEqual(aggregate.evaluated_cases, 1)
        self.assertEqual(aggregate.abstentions, 1)
        self.assertEqual(aggregate.accuracy.denominator, 1)

    def test_pairwise_matrix_label_metrics_and_mismatch_ids(self):
        results = [
            completed_case(
                "a-correct",
                "A_WINS",
                "A_WINS",
                "A_WINS",
                mode=EvaluationMode.PAIRWISE,
            ),
            completed_case(
                "b-as-tie",
                "B_WINS",
                "TIE",
                "B_WINS",
                mode=EvaluationMode.PAIRWISE,
            ),
            completed_case(
                "tie-correct",
                "TIE",
                "TIE",
                "TIE",
                mode=EvaluationMode.PAIRWISE,
            ),
        ]

        report = calculate_error_analysis(results, bootstrap_iterations=100)
        terra = report.analyses[AnalysisTarget.TERRA].pairwise

        self.assertEqual(terra.confusion_matrix["A_WINS"]["A_WINS"], 1)
        self.assertEqual(terra.confusion_matrix["B_WINS"]["TIE"], 1)
        self.assertEqual(terra.accuracy.value, 0.6667)
        self.assertEqual(terra.by_label["TIE"].precision.value, 0.5)
        self.assertEqual(terra.by_label["TIE"].recall.value, 1.0)
        self.assertEqual(terra.mismatch_case_ids, ["b-as-tie"])

    def test_bootstrap_is_deterministic_and_contains_observed_accuracy(self):
        results = [
            completed_case("one", "PASS", "PASS", "PASS"),
            completed_case("two", "PASS", "FAIL", "FAIL"),
            completed_case("three", "FAIL", "FAIL", "FAIL"),
            completed_case("four", "FAIL", "FAIL", "FAIL"),
        ]

        first = calculate_error_analysis(
            results, bootstrap_iterations=200, random_seed=7
        )
        second = calculate_error_analysis(
            results, bootstrap_iterations=200, random_seed=7
        )
        first_accuracy = first.analyses[AnalysisTarget.TERRA].pass_fail_overall.accuracy
        second_accuracy = second.analyses[AnalysisTarget.TERRA].pass_fail_overall.accuracy

        self.assertEqual(first_accuracy, second_accuracy)
        self.assertLessEqual(first_accuracy.ci_lower, first_accuracy.value)
        self.assertGreaterEqual(first_accuracy.ci_upper, first_accuracy.value)

    def test_missing_class_makes_rate_undefined_and_adds_warnings(self):
        results = [
            completed_case(
                "draft-pass",
                "PASS",
                "PASS",
                "PASS",
                review_status=ReviewStatus.DRAFT,
            )
        ]

        report = calculate_error_analysis(results, bootstrap_iterations=20)
        terra = report.analyses[AnalysisTarget.TERRA].pass_fail_overall

        self.assertIsNone(terra.false_pass_rate.value)
        self.assertEqual(terra.false_pass_rate.denominator, 0)
        self.assertEqual(terra.false_pass_rate.bootstrap_samples, 0)
        self.assertTrue(any("Draft human labels" in warning for warning in report.warnings))
        self.assertTrue(any("class coverage" in warning for warning in report.warnings))
        self.assertTrue(any("Fewer than 30" in warning for warning in report.warnings))

    def test_rejects_invalid_statistical_settings(self):
        result = completed_case("one", "PASS", "PASS", "PASS")

        with self.assertRaisesRegex(ValueError, "bootstrap_iterations"):
            calculate_error_analysis([result], bootstrap_iterations=0)
        with self.assertRaisesRegex(ValueError, "confidence_level"):
            calculate_error_analysis([result], confidence_level=1.0)


if __name__ == "__main__":
    unittest.main()
