"""Governance controls proving that the rubric itself is approved for use."""

import argparse
import hashlib
import json
from datetime import date
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .rubric import ACTIVE_RUBRIC, EvaluationRubric


class ApprovalStatus(str, Enum):
    """Lifecycle state chosen by the rubric owners."""

    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"
    RETIRED = "retired"


class ReviewDecision(str, Enum):
    """One reviewer's explicit decision."""

    APPROVED = "approved"
    REJECTED = "rejected"


class ReviewerRole(str, Enum):
    """Independent perspectives required before production use."""

    DOMAIN_EXPERT = "domain_expert"
    PRODUCT_OWNER = "product_owner"
    SAFETY_OWNER = "safety_owner"


class ReviewerAttestation(BaseModel):
    """Named, dated evidence that a person reviewed the rubric semantics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: ReviewerRole
    reviewer_id: str = Field(min_length=1)
    decision: ReviewDecision
    reviewed_on: date
    notes: str = Field(min_length=1)


class RubricValidationEvidence(BaseModel):
    """Human test evidence supporting the rubric approval decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str = Field(min_length=1)
    dataset_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    report_location: str = Field(min_length=1)
    report_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    human_reviewed_cases: int = Field(ge=0)
    passed: bool
    coverage_notes: str = Field(min_length=1)


class RubricApproval(BaseModel):
    """Versioned human-governance artifact for one exact rubric document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    approval_id: str = Field(min_length=1)
    rubric_name: str = Field(min_length=1)
    rubric_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    rubric_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    status: ApprovalStatus
    intended_use: str = Field(min_length=1)
    prohibited_uses: List[str] = Field(min_length=1)
    known_limitations: List[str] = Field(min_length=1)
    validation_evidence: RubricValidationEvidence
    reviewer_attestations: List[ReviewerAttestation]
    approved_on: Optional[date] = None
    expires_on: Optional[date] = None
    decision_notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_reviewer_roles(self) -> "RubricApproval":
        roles = [review.role for review in self.reviewer_attestations]
        if len(set(roles)) != len(roles):
            raise ValueError("A rubric approval cannot repeat a reviewer role")
        return self


class RubricGovernancePolicy(BaseModel):
    """Controls that every production rubric approval must satisfy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = "1.0.0"
    required_roles: List[ReviewerRole] = Field(
        default_factory=lambda: [
            ReviewerRole.DOMAIN_EXPERT,
            ReviewerRole.PRODUCT_OWNER,
        ]
    )
    minimum_human_reviewed_cases: int = Field(default=30, ge=1)
    require_separate_reviewers: bool = True
    require_expiration: bool = True


DEFAULT_RUBRIC_GOVERNANCE_POLICY = RubricGovernancePolicy()


class RubricGovernanceCheck(BaseModel):
    """One readable control result used in audits and the production gate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    check_id: str
    passed: bool
    explanation: str


class RubricApprovalValidation(BaseModel):
    """Complete validation result for one approval and one active rubric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    approval_id: str
    rubric_name: str
    rubric_version: str
    expected_fingerprint: str
    supplied_fingerprint: str
    evaluated_on: date
    policy_version: str
    valid_for_production: bool
    checks: List[RubricGovernanceCheck]
    failed_check_ids: List[str]


def rubric_fingerprint(rubric: EvaluationRubric) -> str:
    """Hash the complete rubric so any semantic edit invalidates old approval."""

    canonical = json.dumps(
        rubric.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _governance_check(
    check_id: str,
    passed: bool,
    success: str,
    failure: str,
) -> RubricGovernanceCheck:
    return RubricGovernanceCheck(
        check_id=check_id,
        passed=passed,
        explanation=success if passed else failure,
    )


def validate_rubric_approval(
    rubric: EvaluationRubric,
    approval: RubricApproval,
    *,
    policy: RubricGovernancePolicy = DEFAULT_RUBRIC_GOVERNANCE_POLICY,
    evaluated_on: Optional[date] = None,
) -> RubricApprovalValidation:
    """Validate identity, evidence, independent approval, and expiration."""

    today = evaluated_on or date.today()
    expected_fingerprint = rubric_fingerprint(rubric)
    attestations = {item.role: item for item in approval.reviewer_attestations}
    required_attestations = [attestations.get(role) for role in policy.required_roles]
    approved_required = all(
        item is not None and item.decision == ReviewDecision.APPROVED
        for item in required_attestations
    )
    reviewer_ids = [
        item.reviewer_id for item in required_attestations if item is not None
    ]
    separate_reviewers = (
        not policy.require_separate_reviewers
        or len(reviewer_ids) == len(policy.required_roles)
        and len(set(reviewer_ids)) == len(reviewer_ids)
    )
    valid_dates = (
        approval.approved_on is not None
        and approval.approved_on <= today
        and (
            not policy.require_expiration
            or approval.expires_on is not None
            and approval.approved_on < approval.expires_on
            and today <= approval.expires_on
        )
    )
    valid_review_dates = (
        approval.approved_on is not None
        and all(
            item is not None
            and item.reviewed_on <= approval.approved_on
            and item.reviewed_on <= today
            for item in required_attestations
        )
    )
    checks = [
        _governance_check(
            "approval_status",
            approval.status == ApprovalStatus.APPROVED,
            "Approval status is approved.",
            f"Approval status is {approval.status.value}, not approved.",
        ),
        _governance_check(
            "rubric_name",
            approval.rubric_name == rubric.name,
            "Approval rubric name matches the active rubric.",
            "Approval rubric name does not match the active rubric.",
        ),
        _governance_check(
            "rubric_version",
            approval.rubric_version == rubric.version,
            "Approval version matches the active rubric.",
            "Approval version does not match the active rubric.",
        ),
        _governance_check(
            "rubric_fingerprint",
            approval.rubric_fingerprint == expected_fingerprint,
            "Approval fingerprint matches every active rubric field.",
            "Active rubric content changed after this approval was created.",
        ),
        _governance_check(
            "validation_evidence_passed",
            approval.validation_evidence.passed,
            "Human rubric-validation evidence passed.",
            "Human rubric-validation evidence did not pass.",
        ),
        _governance_check(
            "minimum_human_reviewed_cases",
            approval.validation_evidence.human_reviewed_cases
            >= policy.minimum_human_reviewed_cases,
            "Human validation case count meets the governance policy.",
            (
                "Human validation case count is below the required "
                f"{policy.minimum_human_reviewed_cases}."
            ),
        ),
        _governance_check(
            "required_reviewer_roles",
            approved_required,
            "Every required reviewer role approved the rubric.",
            "A required reviewer role is missing or did not approve the rubric.",
        ),
        _governance_check(
            "separation_of_duties",
            separate_reviewers,
            "Required roles were completed by separate reviewers.",
            "Required roles must be completed by different reviewers.",
        ),
        _governance_check(
            "review_dates",
            valid_review_dates,
            "Required reviews occurred before the final approval decision.",
            "A required review is missing, future-dated, or later than approval.",
        ),
        _governance_check(
            "approval_dates",
            valid_dates,
            "Approval is active and unexpired.",
            "Approval dates are missing, future-dated, invalid, or expired.",
        ),
    ]
    failed = [check.check_id for check in checks if not check.passed]
    return RubricApprovalValidation(
        approval_id=approval.approval_id,
        rubric_name=rubric.name,
        rubric_version=rubric.version,
        expected_fingerprint=expected_fingerprint,
        supplied_fingerprint=approval.rubric_fingerprint,
        evaluated_on=today,
        policy_version=policy.version,
        valid_for_production=not failed,
        checks=checks,
        failed_check_ids=failed,
    )


def load_rubric_approval(path: Union[str, Path]) -> RubricApproval:
    """Load an approval artifact with strict Pydantic field validation."""

    return RubricApproval.model_validate_json(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    """Validate an approval artifact against the exact active rubric."""

    parser = argparse.ArgumentParser(
        description="Validate human approval of the active evaluation rubric"
    )
    parser.add_argument("--approval", required=True)
    parser.add_argument("--evaluated-on", type=date.fromisoformat)
    args = parser.parse_args()
    result = validate_rubric_approval(
        ACTIVE_RUBRIC,
        load_rubric_approval(args.approval),
        evaluated_on=args.evaluated_on,
    )
    print(result.model_dump_json(indent=2))
    if not result.valid_for_production:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
