"""API-independent claim validation and conservative symbolic adjudication.

This is a research scaffold, not a trained extractor or a truth oracle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
import json
import re
from pathlib import Path
from typing import Any, Iterable, Literal


Decision = Literal["accept", "reject", "abstain"]


@dataclass
class Article:
    article_id: str
    text: str
    published_at: str
    source: str | None = None
    url: str | None = None
    author: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateClaim:
    claim_id: str
    article_id: str
    subject: str
    relation: str
    object: str
    subject_type: str
    object_type: str
    event_time: str
    evidence: str
    value: float | None = None
    unit: str | None = None
    polarity: str | None = None
    modality: str | None = None
    raw_output: dict[str, Any] = field(default_factory=dict)


@dataclass
class Verification:
    claim_id: str
    decision: Decision
    reason_codes: list[str]
    evidence_supported: bool | None
    evidence_span: str | None
    scores: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _valid_iso_date(value: str) -> bool:
    try:
        date.fromisoformat(value[:10])
        return True
    except (TypeError, ValueError):
        return False


def validate_article(article: Article) -> list[str]:
    errors = []
    if not article.article_id.strip():
        errors.append("missing_article_id")
    if not article.text.strip():
        errors.append("missing_article_text")
    if not _valid_iso_date(article.published_at):
        errors.append("invalid_publication_date")
    return errors


def validate_claim(claim: CandidateClaim) -> list[str]:
    errors = []
    for name in ("claim_id", "article_id", "subject", "relation", "object", "evidence"):
        if not getattr(claim, name).strip():
            errors.append(f"missing_{name}")
    if not _valid_iso_date(claim.event_time):
        errors.append("invalid_event_date")
    if claim.value is not None and claim.unit is None:
        errors.append("numeric_value_missing_unit")
    return errors


def find_evidence(article_text: str, evidence: str) -> str | None:
    """Return the exact source span iff literal evidence occurs in the article."""
    if not evidence.strip():
        return None
    match = re.search(re.escape(evidence.strip()), article_text, flags=re.IGNORECASE)
    return match.group(0) if match else None


def verify_claim(
    claim: CandidateClaim,
    article: Article,
    known_entity_types: dict[str, str] | None = None,
    allowed_relations: Iterable[str] | None = None,
) -> Verification:
    reasons: list[str] = []
    notes: list[str] = []
    errs = validate_claim(claim)
    if errs:
        return Verification(claim.claim_id, "abstain", errs, None, None)
    article_errs = validate_article(article)
    if article_errs:
        return Verification(claim.claim_id, "abstain", article_errs, None, None)
    if claim.article_id != article.article_id:
        return Verification(claim.claim_id, "reject", ["article_id_mismatch"], False, None)

    span = find_evidence(article.text, claim.evidence)
    if span is None:
        reasons.append("evidence_not_found_verbatim")

    if not _valid_iso_date(claim.event_time):
        reasons.append("invalid_event_date")
    elif claim.event_time[:10] > article.published_at[:10]:
        reasons.append("event_after_publication")

    if allowed_relations is not None and claim.relation not in set(allowed_relations):
        reasons.append("relation_outside_schema")

    if known_entity_types is not None:
        for entity, predicted_type in ((claim.subject, claim.subject_type), (claim.object, claim.object_type)):
            expected = known_entity_types.get(entity)
            if expected is not None and expected != predicted_type:
                reasons.append(f"entity_type_conflict:{entity}")

    # A missing verbatim span rejects only the evidence link; date/type conflicts are hard rule failures.
    hard = [r for r in reasons if r != "evidence_not_found_verbatim"]
    if hard:
        decision: Decision = "reject"
    elif span is None:
        decision = "reject"
    else:
        # Literal evidence occurrence is only a grounding proxy. It does not entail the full claim.
        notes.append("verbatim_span_found; semantic entailment and objective truth remain unverified")
        decision = "abstain"
    return Verification(claim.claim_id, decision, reasons, span is not None, span, notes=notes)


def adjudicate(claim: CandidateClaim, verification: Verification, policy: str = "conservative") -> Verification:
    """Apply a configurable policy; default requires external semantic review to accept."""
    if policy == "grounded_literal" and verification.decision == "abstain" and verification.evidence_supported:
        verification.decision = "accept"
        verification.notes.append("accepted by grounded_literal policy; literal match is not truth verification")
    elif policy != "conservative" and policy != "grounded_literal":
        verification.notes.append(f"unknown policy {policy!r}; conservative decision retained")
    return verification


def export_verified_quadruples(
    claims: Iterable[CandidateClaim],
    verifications: Iterable[Verification],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    time_to_id: dict[str, int],
    output_path: str | Path,
    provenance_path: str | Path,
) -> int:
    """Write accepted claims in FinDKG's 4-column TSV format plus provenance JSONL."""
    verification_by_id = {v.claim_id: v for v in verifications}
    out = Path(output_path)
    provenance = Path(provenance_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    provenance.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8", newline="") as graph_f, provenance.open("w", encoding="utf-8") as prov_f:
        for claim in claims:
            result = verification_by_id.get(claim.claim_id)
            if result is None or result.decision != "accept":
                continue
            try:
                h, r, t = entity_to_id[claim.subject], relation_to_id[claim.relation], entity_to_id[claim.object]
                time_id = time_to_id[claim.event_time[:10]]
            except KeyError:
                continue
            graph_f.write(f"{h}\t{r}\t{t}\t{time_id}\t0\n")
            prov_f.write(json.dumps({"claim": asdict(claim), "verification": asdict(result)}, ensure_ascii=False) + "\n")
            written += 1
    return written


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
