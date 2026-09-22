import tempfile
import unittest
from datetime import date
from pathlib import Path

from llm_judge.rubric import ACTIVE_RUBRIC
from llm_judge.rubric_approval import (
    ApprovalStatus,
    ReviewDecision,
    ReviewerAttestation,
    ReviewerRole,
    RubricApproval,
    RubricValidationEvidence,
    load_rubric_approval,
    rubric_fingerprint,
    validate_rubric_approval,
)


def valid_approval():
    return RubricApproval(
        approval_id="answer-quality-v2-production",
        rubric_name=ACTIVE_RUBRIC.name,
        rubric_version=ACTIVE_RUBRIC.version,
        rubric_fingerprint=rubric_fingerprint(ACTIVE_RUBRIC),
        status=ApprovalStatus.APPROVED,
        intended_use="Evaluate text answers using the documented answer-quality task.",
        prohibited_uses=["Medical diagnosis without a specialized rubric."],
        known_limitations=["Requires representative human-reviewed test cases."],
        validation_evidence=RubricValidationEvidence(
            evidence_id="human-validation-2026-01",
            dataset_fingerprint="a" * 64,
            report_location="reports/rubric_validation_2026_01.json",
            report_fingerprint="b" * 64,
            human_reviewed_cases=100,
            passed=True,
            coverage_notes="Covered good, bad, borderline, and pairwise tie cases.",
        ),
        reviewer_attestations=[
            ReviewerAttestation(
                role=ReviewerRole.DOMAIN_EXPERT,
                reviewer_id="domain-owner",
                decision=ReviewDecision.APPROVED,
                reviewed_on=date(2026, 1, 10),
                notes="Criteria and anchors match domain expectations.",
            ),
            ReviewerAttestation(
                role=ReviewerRole.PRODUCT_OWNER,
                reviewer_id="product-owner",
                decision=ReviewDecision.APPROVED,
                reviewed_on=date(2026, 1, 11),
                notes="Intended use and limitations are acceptable.",
            ),
        ],
        approved_on=date(2026, 1, 12),
        expires_on=date(2027, 1, 12),
        decision_notes="Approved for the documented answer-quality use case.",
    )


class RubricApprovalTests(unittest.TestCase):
    def test_valid_human_approval_matches_the_exact_active_rubric(self):
        validation = validate_rubric_approval(
            ACTIVE_RUBRIC,
            valid_approval(),
            evaluated_on=date(2026, 9, 7),
        )

        self.assertTrue(validation.valid_for_production)
        self.assertEqual(validation.failed_check_ids, [])
        self.assertTrue(all(check.passed for check in validation.checks))

    def test_one_rubric_edit_invalidates_the_old_fingerprint(self):
        changed = ACTIVE_RUBRIC.model_copy(
            update={
                "judge_instructions": ACTIVE_RUBRIC.judge_instructions
                + ["New semantic scoring instruction."],
            }
        )

        validation = validate_rubric_approval(
            changed,
            valid_approval(),
            evaluated_on=date(2026, 9, 7),
        )

        self.assertFalse(validation.valid_for_production)
        self.assertIn("rubric_fingerprint", validation.failed_check_ids)

    def test_expired_approval_fails(self):
        approval = valid_approval().model_copy(
            update={"expires_on": date(2026, 8, 1)}
        )

        validation = validate_rubric_approval(
            ACTIVE_RUBRIC,
            approval,
            evaluated_on=date(2026, 9, 7),
        )

        self.assertFalse(validation.valid_for_production)
        self.assertIn("approval_dates", validation.failed_check_ids)

    def test_missing_role_and_same_person_for_both_roles_fail(self):
        approval = valid_approval()
        missing = approval.model_copy(
            update={"reviewer_attestations": approval.reviewer_attestations[:1]}
        )
        missing_validation = validate_rubric_approval(
            ACTIVE_RUBRIC,
            missing,
            evaluated_on=date(2026, 9, 7),
        )
        same_person_reviews = [
            item.model_copy(update={"reviewer_id": "same-person"})
            for item in approval.reviewer_attestations
        ]
        same_person = approval.model_copy(
            update={"reviewer_attestations": same_person_reviews}
        )
        same_validation = validate_rubric_approval(
            ACTIVE_RUBRIC,
            same_person,
            evaluated_on=date(2026, 9, 7),
        )

        self.assertIn("required_reviewer_roles", missing_validation.failed_check_ids)
        self.assertIn("separation_of_duties", same_validation.failed_check_ids)

    def test_review_after_final_approval_fails(self):
        approval = valid_approval()
        late_product_review = approval.reviewer_attestations[1].model_copy(
            update={"reviewed_on": date(2026, 1, 13)}
        )
        approval = approval.model_copy(
            update={
                "reviewer_attestations": [
                    approval.reviewer_attestations[0],
                    late_product_review,
                ]
            }
        )

        validation = validate_rubric_approval(
            ACTIVE_RUBRIC,
            approval,
            evaluated_on=date(2026, 9, 7),
        )

        self.assertFalse(validation.valid_for_production)
        self.assertIn("review_dates", validation.failed_check_ids)

    def test_approval_json_round_trips_with_strict_validation(self):
        approval = valid_approval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "approval.json"
            path.write_text(approval.model_dump_json(indent=2), encoding="utf-8")

            loaded = load_rubric_approval(path)

        self.assertEqual(loaded, approval)


if __name__ == "__main__":
    unittest.main()
