import tempfile
import unittest
from datetime import date
from pathlib import Path

from llm_judge.contracts import EvaluationMode
from llm_judge.error_analysis import (
    AnalysisTarget,
    BinaryClassificationAnalysis,
    BinaryConfusionMatrix,
    ErrorAnalysisReport,
    JudgeErrorAnalysis,
    MetricEstimate,
    PairwiseClassificationAnalysis,
)
from llm_judge.multi_judge import JudgeModel
from llm_judge.production_gate import (
    GateDecision,
    ProductionThresholds,
    evaluate_production_gate,
    load_thresholds,
)
from llm_judge.reliability import (
    AgreementMetrics,
    ReliabilityReport,
    ScoreCorrelationMetrics,
)
from llm_judge.stability import ModelStabilitySummary, StabilityReport
from llm_judge.rubric import ACTIVE_RUBRIC
from llm_judge.rubric_approval import ApprovalStatus, validate_rubric_approval
from test_rubric_approval import valid_approval


def valid_rubric_validation():
    return validate_rubric_approval(
        ACTIVE_RUBRIC,
        valid_approval(),
        evaluated_on=date(2026, 9, 7),
    )


def estimate(value, lower=None, upper=None, denominator=100):
    return MetricEstimate(
        value=value,
        ci_lower=value if lower is None else lower,
        ci_upper=value if upper is None else upper,
        denominator=denominator,
        denominator_description="test cases",
        bootstrap_samples=100,
    )


def binary_analysis():
    return BinaryClassificationAnalysis(
        available_cases=100,
        evaluated_cases=100,
        abstentions=0,
        confusion_matrix=BinaryConfusionMatrix(
            true_positive=49,
            true_negative=49,
            false_positive=1,
            false_negative=1,
        ),
        accuracy=estimate(0.98, lower=0.95, upper=1.0),
        precision=estimate(0.98),
        recall=estimate(0.98),
        f1=estimate(0.98),
        false_pass_rate=estimate(0.02, lower=0.0, upper=0.04, denominator=50),
        false_fail_rate=estimate(0.02, lower=0.0, upper=0.04, denominator=50),
        false_pass_case_ids=["fp"],
        false_fail_case_ids=["fn"],
    )


def pairwise_analysis():
    return PairwiseClassificationAnalysis(
        available_cases=100,
        evaluated_cases=100,
        abstentions=0,
        confusion_matrix={},
        accuracy=estimate(0.98, lower=0.95, upper=1.0),
        macro_f1=estimate(0.98),
        by_label={},
        mismatch_case_ids=["pair-mismatch"],
    )


def error_report():
    analyses = {
        target: JudgeErrorAnalysis(
            target=target,
            pass_fail_overall=binary_analysis(),
            pass_fail_by_mode={
                EvaluationMode.BINARY: binary_analysis(),
                EvaluationMode.SCORE: binary_analysis(),
            },
            pairwise=pairwise_analysis(),
        )
        for target in AnalysisTarget
    }
    return ErrorAnalysisReport(
        total_cases=100,
        completed_cases=100,
        failed_cases=0,
        confidence_level=0.95,
        bootstrap_iterations=100,
        random_seed=42,
        analyses=analyses,
        warnings=[],
    )


def reliability_report():
    agreement = AgreementMetrics(
        available_cases=100,
        evaluated_cases=100,
        abstentions=0,
        exact_matches=98,
        agreement_rate=0.98,
        cohens_kappa=0.95,
    )
    return ReliabilityReport(
        total_cases=100,
        completed_cases=100,
        failed_cases=0,
        failure_rate=0.0,
        human_review_required=2,
        human_review_rate=0.02,
        model_disagreements=2,
        model_disagreement_rate=0.02,
        terra_vs_human=agreement,
        luna_vs_human=agreement,
        aggregate_vs_human=agreement,
        terra_vs_luna=agreement,
        by_mode={},
        score_correlation=ScoreCorrelationMetrics(
            score_cases=30,
            score_pairs=120,
            terra_pearson=0.95,
            luna_pearson=0.94,
            aggregate_pearson=0.96,
        ),
        warnings=[],
    )


def stability_summary(model):
    return ModelStabilitySummary(
        model=model,
        evaluated_cases=100,
        total_calls=300,
        failed_calls=0,
        failure_rate=0.0,
        repeat_comparable_cases=100,
        stable_cases=98,
        unstable_cases=2,
        insufficient_repeat_cases=0,
        mean_repeat_consistency=0.98,
        median_repeat_consistency=1.0,
        position_cases=30,
        position_comparisons=90,
        position_mismatches=1,
        position_flip_rate=0.0111,
        first_position_preference_pairs=0,
        second_position_preference_pairs=0,
        tie_relevant_pairs=30,
        tie_consistency_rate=1.0,
        unstable_case_ids=["unstable"],
        position_biased_case_ids=["position"],
        insufficient_repeat_case_ids=[],
    )


def stability_report():
    return StabilityReport(
        dataset_cases=100,
        repeat_count=3,
        total_calls=600,
        failed_calls=0,
        failure_rate=0.0,
        per_model={
            JudgeModel.TERRA: stability_summary(JudgeModel.TERRA),
            JudgeModel.LUNA: stability_summary(JudgeModel.LUNA),
        },
        results=[],
        warnings=[],
    )


class ProductionGateTests(unittest.TestCase):
    def test_gate_passes_when_every_requirement_passes(self):
        report = evaluate_production_gate(
            reliability_report(),
            error_report(),
            stability_report(),
            draft_cases=0,
            matching_case_sets=True,
            rubric_approval=valid_rubric_validation(),
        )

        self.assertEqual(report.decision, GateDecision.PASSED)
        self.assertEqual(report.failed_checks, 0)
        self.assertEqual(report.passed_checks, report.total_checks)

    def test_gate_fails_and_names_the_specific_failed_checks(self):
        errors = error_report()
        aggregate = errors.analyses[AnalysisTarget.AGGREGATE]
        risky_binary = aggregate.pass_fail_overall.model_copy(
            update={
                "accuracy": estimate(0.88, lower=0.84, upper=0.91),
                "false_pass_rate": estimate(
                    0.08, lower=0.04, upper=0.12, denominator=50
                ),
            }
        )
        errors = errors.model_copy(
            update={
                "analyses": {
                    **errors.analyses,
                    AnalysisTarget.AGGREGATE: aggregate.model_copy(
                        update={"pass_fail_overall": risky_binary}
                    ),
                }
            }
        )

        report = evaluate_production_gate(
            reliability_report(),
            errors,
            stability_report(),
            draft_cases=3,
            matching_case_sets=False,
            rubric_approval=valid_rubric_validation(),
        )

        self.assertEqual(report.decision, GateDecision.FAILED)
        self.assertIn("matching_case_sets", report.failed_check_ids)
        self.assertIn("no_draft_labels", report.failed_check_ids)
        self.assertIn("pass_fail_accuracy_lower_bound", report.failed_check_ids)
        self.assertIn("false_pass_rate_upper_bound", report.failed_check_ids)

    def test_missing_measurement_fails_safely(self):
        reliability = reliability_report().model_copy(
            update={"model_disagreement_rate": None}
        )

        report = evaluate_production_gate(
            reliability,
            error_report(),
            stability_report(),
            draft_cases=0,
            matching_case_sets=True,
            rubric_approval=valid_rubric_validation(),
        )
        check = next(
            item for item in report.checks if item.check_id == "model_disagreement_rate"
        )

        self.assertFalse(check.passed)
        self.assertIsNone(check.observed)
        self.assertIn("unavailable", check.explanation)

    def test_missing_rubric_approval_fails_even_when_metrics_pass(self):
        report = evaluate_production_gate(
            reliability_report(),
            error_report(),
            stability_report(),
            draft_cases=0,
            matching_case_sets=True,
        )

        self.assertEqual(report.decision, GateDecision.FAILED)
        self.assertIn("rubric_approval", report.failed_check_ids)

    def test_invalid_rubric_approval_fails_even_when_metrics_pass(self):
        draft_approval = valid_approval().model_copy(
            update={"status": ApprovalStatus.DRAFT}
        )
        invalid_validation = validate_rubric_approval(
            ACTIVE_RUBRIC,
            draft_approval,
            evaluated_on=date(2026, 9, 7),
        )

        report = evaluate_production_gate(
            reliability_report(),
            error_report(),
            stability_report(),
            draft_cases=0,
            matching_case_sets=True,
            rubric_approval=invalid_validation,
        )

        self.assertEqual(report.decision, GateDecision.FAILED)
        self.assertIn("rubric_approval", report.failed_check_ids)

    def test_custom_threshold_file_is_validated(self):
        custom = ProductionThresholds(minimum_completed_cases=250)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thresholds.json"
            path.write_text(custom.model_dump_json(indent=2), encoding="utf-8")

            loaded = load_thresholds(path)

        self.assertEqual(loaded.minimum_completed_cases, 250)
        self.assertEqual(loaded.version, "1.0.0")


if __name__ == "__main__":
    unittest.main()
