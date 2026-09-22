import tempfile
import unittest
from pathlib import Path

from llm_judge.azure_client import TokenUsage
from llm_judge.calibration import (
    CalibrationDecision,
    MetricChange,
    compare_calibration_runs,
    load_split_manifest,
    save_dataset_split,
    split_evaluation_dataset,
    verify_heldout_run,
)
from llm_judge.contracts import (
    BinaryEvaluationResult,
    EvaluationMode,
    ReferencePolicy,
)
from llm_judge.dataset import EvaluationCase, EvaluationDataset, ReviewStatus, load_jsonl
from llm_judge.dataset_runner import CaseRunResult, CaseRunStatus
from llm_judge.multi_judge import JudgeModel, ModelJudgment, TwoModelJudgeResult
from llm_judge.rubric import ExampleKind


def evaluation_case(case_id, expected):
    return EvaluationCase(
        case_id=case_id,
        case_kind=ExampleKind.GOOD if expected == "PASS" else ExampleKind.BAD,
        mode=EvaluationMode.BINARY,
        reference_policy=ReferencePolicy.REQUIRED,
        question=f"Question for {case_id}?",
        reference_answer="Reference.",
        candidate_answer="Candidate.",
        expected_binary_decision=expected,
        review_status=ReviewStatus.HUMAN_REVIEWED,
        reviewer_count=1,
        review_notes="Reviewed calibration label.",
    )


def judgment(case_id, model, decision):
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


def run_result(case_id, human, prediction):
    judged = TwoModelJudgeResult(
        case_id=case_id,
        mode=EvaluationMode.BINARY,
        judgments=[
            judgment(case_id, JudgeModel.TERRA, prediction),
            judgment(case_id, JudgeModel.LUNA, prediction),
        ],
        agreement=True,
        aggregate_decision=prediction,
        requires_human_review=False,
    )
    return CaseRunResult(
        case_id=case_id,
        mode=EvaluationMode.BINARY,
        review_status=ReviewStatus.HUMAN_REVIEWED,
        human_decision=human,
        status=CaseRunStatus.COMPLETED,
        judge_result=judged,
        matches_human=human == prediction,
        requires_human_review=False,
    )


def four_case_dataset():
    return EvaluationDataset(
        cases=[
            evaluation_case("pass-one", "PASS"),
            evaluation_case("pass-two", "PASS"),
            evaluation_case("fail-one", "FAIL"),
            evaluation_case("fail-two", "FAIL"),
        ]
    )


class DatasetSplitTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_stratified_by_mode(self):
        dataset = load_jsonl("datasets/evaluation_cases.example.jsonl")

        first = split_evaluation_dataset(dataset, allow_drafts=True, random_seed=7)
        second = split_evaluation_dataset(dataset, allow_drafts=True, random_seed=7)
        calibration_ids = set(first.manifest.calibration_case_ids)
        heldout_ids = set(first.manifest.heldout_case_ids)

        self.assertEqual(first, second)
        self.assertFalse(calibration_ids & heldout_ids)
        self.assertEqual(
            calibration_ids | heldout_ids,
            {case.case_id for case in dataset.cases},
        )
        self.assertTrue(
            all(first.manifest.calibration_by_mode[mode] > 0 for mode in EvaluationMode)
        )
        self.assertTrue(
            all(first.manifest.heldout_by_mode[mode] > 0 for mode in EvaluationMode)
        )

    def test_split_rejects_drafts_unless_explicitly_allowed(self):
        dataset = load_jsonl("datasets/evaluation_cases.example.jsonl")

        with self.assertRaisesRegex(ValueError, "Draft cases"):
            split_evaluation_dataset(dataset)

    def test_saved_split_round_trips_and_refuses_overwrite(self):
        split = split_evaluation_dataset(
            four_case_dataset(),
            allow_drafts=True,
            calibration_fraction=0.5,
        )
        with tempfile.TemporaryDirectory() as directory:
            calibration = Path(directory) / "calibration.jsonl"
            heldout = Path(directory) / "heldout.jsonl"
            manifest = Path(directory) / "manifest.json"
            save_dataset_split(
                split,
                calibration_path=calibration,
                heldout_path=heldout,
                manifest_path=manifest,
            )

            self.assertEqual(load_jsonl(calibration), split.calibration)
            self.assertEqual(load_jsonl(heldout), split.heldout)
            self.assertEqual(load_split_manifest(manifest), split.manifest)
            with self.assertRaises(FileExistsError):
                save_dataset_split(
                    split,
                    calibration_path=calibration,
                    heldout_path=heldout,
                    manifest_path=manifest,
                )


class CalibrationComparisonTests(unittest.TestCase):
    def setUp(self):
        self.split = split_evaluation_dataset(
            four_case_dataset(),
            calibration_fraction=0.5,
            random_seed=3,
            allow_drafts=True,
        )
        # Build results from exactly the IDs selected by the deterministic split.
        labels = {
            case.case_id: case.expected_binary_decision.value
            for case in self.split.calibration.cases
        }
        self.candidate = [
            run_result(case_id, human, human) for case_id, human in labels.items()
        ]
        self.baseline = [
            run_result(
                case_id,
                human,
                "FAIL" if human == "PASS" else "PASS",
            )
            for case_id, human in labels.items()
        ]

    def compare(self, baseline=None, candidate=None, reviewed_by="developer"):
        return compare_calibration_runs(
            baseline or self.baseline,
            candidate or self.candidate,
            self.split.manifest,
            baseline_configuration="rubric-v2",
            candidate_configuration="rubric-v3",
            change_summary="Clarify that short correct answers must not be penalized.",
            reviewed_by=reviewed_by,
            bootstrap_iterations=20,
        )

    def test_improved_reviewed_candidate_is_accepted_for_heldout(self):
        report = self.compare()

        self.assertEqual(report.decision, CalibrationDecision.ACCEPTED)
        self.assertIn("aggregate_human_agreement", report.improved_metrics)
        self.assertTrue(report.fixed_false_pass_case_ids)
        self.assertTrue(report.fixed_false_fail_case_ids)
        self.assertFalse(report.new_false_pass_case_ids)
        self.assertFalse(report.regressed_metrics)

    def test_improvement_still_requires_developer_review(self):
        report = self.compare(reviewed_by=None)

        self.assertEqual(report.decision, CalibrationDecision.NEEDS_DEVELOPER_REVIEW)

    def test_regression_is_rejected_and_new_error_ids_are_visible(self):
        report = self.compare(baseline=self.candidate, candidate=self.baseline)

        self.assertEqual(report.decision, CalibrationDecision.REJECTED)
        self.assertTrue(report.regressed_metrics)
        self.assertTrue(report.new_false_pass_case_ids)
        self.assertTrue(report.new_false_fail_case_ids)
        agreement = next(
            item
            for item in report.comparisons
            if item.metric == "aggregate_human_agreement"
        )
        self.assertEqual(agreement.outcome, MetricChange.REGRESSED)

    def test_changed_human_label_is_rejected(self):
        changed = list(self.candidate)
        changed[0] = changed[0].model_copy(update={"human_decision": "CHANGED"})

        with self.assertRaisesRegex(ValueError, "Human labels"):
            self.compare(candidate=changed)

    def test_unexpected_case_id_is_rejected_as_contamination(self):
        contaminated = list(self.candidate)
        contaminated[0] = contaminated[0].model_copy(update={"case_id": "heldout-id"})

        with self.assertRaisesRegex(ValueError, "do not match the split"):
            self.compare(candidate=contaminated)

    def test_heldout_verification_requires_exact_heldout_ids(self):
        comparison = self.compare()
        results = [
            run_result(
                case.case_id,
                case.expected_binary_decision.value,
                case.expected_binary_decision.value,
            )
            for case in self.split.heldout.cases
        ]

        report = verify_heldout_run(
            results,
            self.split.manifest,
            comparison,
            configuration_version="rubric-v3",
            bootstrap_iterations=20,
        )

        self.assertEqual(report.heldout_cases, len(results))
        self.assertEqual(report.configuration_version, "rubric-v3")
        self.assertEqual(report.calibration_reviewed_by, "developer")
        with self.assertRaisesRegex(ValueError, "do not match the split"):
            verify_heldout_run(
                results[:-1],
                self.split.manifest,
                comparison,
                configuration_version="rubric-v3",
                bootstrap_iterations=20,
            )

    def test_heldout_rejects_unapproved_or_different_configuration(self):
        results = [
            run_result(
                case.case_id,
                case.expected_binary_decision.value,
                case.expected_binary_decision.value,
            )
            for case in self.split.heldout.cases
        ]
        unreviewed = self.compare(reviewed_by=None)

        with self.assertRaisesRegex(ValueError, "accepted calibration"):
            verify_heldout_run(
                results,
                self.split.manifest,
                unreviewed,
                configuration_version="rubric-v3",
                bootstrap_iterations=20,
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            verify_heldout_run(
                results,
                self.split.manifest,
                self.compare(),
                configuration_version="different-version",
                bootstrap_iterations=20,
            )


if __name__ == "__main__":
    unittest.main()
