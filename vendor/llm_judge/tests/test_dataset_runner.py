import tempfile
import unittest
from pathlib import Path

from llm_judge.azure_client import TokenUsage
from llm_judge.contracts import (
    BinaryEvaluationResult,
    CriterionScore,
    EvaluationMode,
    EvaluationResult,
    PairwiseEvaluationResult,
)
from llm_judge.dataset import load_jsonl
from llm_judge.dataset_runner import CaseRunStatus, DatasetRunner
from llm_judge.multi_judge import (
    JudgeModel,
    ModelJudgment,
    TwoModelJudgeResult,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "datasets" / "evaluation_cases.example.jsonl"


class FakeTwoModelJudge:
    def __init__(self, fail_case_ids=None):
        self.fail_case_ids = set(fail_case_ids or [])
        self.seen_case_ids = []

    def evaluate(self, case, *, include_examples=True, example_limit=3):
        self.seen_case_ids.append(case.case_id)
        if case.case_id in self.fail_case_ids:
            raise RuntimeError("sensitive provider detail must not be saved")

        if case.mode == EvaluationMode.BINARY:
            parsed = BinaryEvaluationResult(
                case_id=case.case_id,
                decision=case.expected_binary_decision,
                evidence="Matches the human answer key.",
            )
            human_decision = case.expected_binary_decision.value
        elif case.mode == EvaluationMode.PAIRWISE:
            parsed = PairwiseEvaluationResult(
                case_id=case.case_id,
                decision=case.expected_pairwise_decision,
                evidence="Matches the human answer key.",
            )
            human_decision = case.expected_pairwise_decision.value
        else:
            parsed = EvaluationResult(
                case_id=case.case_id,
                scores=[
                    CriterionScore(
                        criterion=criterion,
                        score=score,
                        evidence="Matches the human answer key.",
                    )
                    for criterion, score in case.expected_scores.items()
                ],
                summary="Matches the expected scores.",
            )
            human_decision = parsed.decision.value

        judgments = [
            ModelJudgment(
                model=model,
                deployment=model.value,
                resolved_model=f"{model.value}-test",
                usage=TokenUsage(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                ),
                result=parsed,
            )
            for model in (JudgeModel.TERRA, JudgeModel.LUNA)
        ]
        return TwoModelJudgeResult(
            case_id=case.case_id,
            mode=case.mode,
            judgments=judgments,
            agreement=True,
            aggregate_decision=human_decision,
            requires_human_review=False,
        )


class DatasetRunnerTests(unittest.TestCase):
    def test_production_mode_rejects_draft_dataset_before_model_calls(self) -> None:
        dataset = load_jsonl(DATASET_PATH)
        judge = FakeTwoModelJudge()

        with self.assertRaisesRegex(ValueError, "Draft cases"):
            DatasetRunner(judge).run(dataset)

        self.assertEqual(judge.seen_case_ids, [])

    def test_learning_mode_runs_all_cases_and_writes_incremental_jsonl(self) -> None:
        dataset = load_jsonl(DATASET_PATH)
        judge = FakeTwoModelJudge()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results" / "run.jsonl"
            report = DatasetRunner(judge).run(
                dataset,
                output_path=output,
                allow_drafts=True,
            )
            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(report.dataset_cases, 12)
        self.assertEqual(report.completed_cases, 12)
        self.assertEqual(report.failed_cases, 0)
        self.assertEqual(report.matches_human, 12)
        self.assertEqual(len(lines), 12)
        self.assertIn('"total_tokens":15', lines[0])

    def test_one_failure_is_recorded_and_does_not_stop_later_cases(self) -> None:
        dataset = load_jsonl(DATASET_PATH)
        failed_id = dataset.cases[1].case_id
        judge = FakeTwoModelJudge(fail_case_ids={failed_id})

        report = DatasetRunner(judge).run(dataset, allow_drafts=True)
        failed = next(item for item in report.results if item.case_id == failed_id)

        self.assertEqual(report.completed_cases, 11)
        self.assertEqual(report.failed_cases, 1)
        self.assertEqual(len(judge.seen_case_ids), 12)
        self.assertEqual(failed.status, CaseRunStatus.ERROR)
        self.assertEqual(failed.error_type, "RuntimeError")
        self.assertNotIn("sensitive provider detail", failed.error_message)
        self.assertTrue(failed.requires_human_review)

    def test_existing_output_requires_explicit_overwrite(self) -> None:
        dataset = load_jsonl(DATASET_PATH)
        judge = FakeTwoModelJudge()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.jsonl"
            output.write_text("keep me\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                DatasetRunner(judge).run(
                    dataset,
                    output_path=output,
                    allow_drafts=True,
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")

        self.assertEqual(judge.seen_case_ids, [])


if __name__ == "__main__":
    unittest.main()
