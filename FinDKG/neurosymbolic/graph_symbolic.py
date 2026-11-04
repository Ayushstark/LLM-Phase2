"""Symbolic consistency checks for existing FinDKG temporal quadruples.

The engine audits dataset structure and schema consistency. It cannot determine
whether a graph fact is supported by its original news source or objectively true.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Quad:
    head: int
    relation: int
    tail: int
    time: int
    extra: str = "0"


@dataclass
class AuditRecord:
    split: str
    row_index: int
    quadruple: list[int]
    decision: str
    reason_codes: list[str]


def read_tsv_quads(path: str | Path) -> list[Quad]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for idx, line in enumerate(stream, 1):
            cols = line.rstrip("\r\n").split("\t")
            if len(cols) < 4:
                raise ValueError(f"{path}:{idx}: expected at least four tab-separated columns")
            try:
                rows.append(Quad(*(int(value) for value in cols[:4]), extra=cols[4] if len(cols) > 4 else "0"))
            except ValueError as exc:
                raise ValueError(f"{path}:{idx}: invalid integer in quadruple") from exc
    return rows


def read_entity_types(path: str | Path) -> dict[int, int]:
    """Read entity2id, taking type ID from the fourth column as the DKG loader does."""
    result: dict[int, int] = {}
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for idx, line in enumerate(stream, 1):
            cols = line.rstrip("\r\n").split("\t")
            if len(cols) < 4:
                raise ValueError(f"{path}:{idx}: expected four entity mapping columns")
            entity_id, type_id = int(cols[1]), int(cols[3])
            if entity_id in result:
                raise ValueError(f"{path}:{idx}: duplicate entity ID {entity_id}")
            result[entity_id] = type_id
    return result


def audit_dataset(dataset_dir: str | Path) -> tuple[list[AuditRecord], dict[str, Any]]:
    """Apply deterministic schema, entity-type, and temporal split constraints.

    Findings are marked accept/reject/abstain; abstain means the available graph
    metadata cannot settle the condition (for example, an unseen entity type).
    """
    root = Path(dataset_dir)
    stat = [int(x) for x in (root / "stat.txt").read_text(encoding="utf-8").split()[:3]]
    num_entities, num_relations, num_times = stat
    entity_types = read_entity_types(root / "entity2id.txt")
    splits = {name: read_tsv_quads(root / filename) for name, filename in (
        ("train", "train.txt"), ("valid", "valid.txt"), ("test", "test.txt"))}
    bounds = {
        "train": (min(q.time for q in splits["train"]), max(q.time for q in splits["train"])),
        "valid": (min(q.time for q in splits["valid"]), max(q.time for q in splits["valid"])),
        "test": (min(q.time for q in splits["test"]), max(q.time for q in splits["test"])),
    }
    records: list[AuditRecord] = []
    all_qsets: dict[str, set[tuple[int, int, int, int]]] = {}
    for split, rows in splits.items():
        accepted: set[tuple[int, int, int, int]] = set()
        seen_quads: set[tuple[int, int, int, int]] = set()
        for row_index, q in enumerate(rows, 1):
            reasons = []
            if not (0 <= q.head < num_entities and 0 <= q.tail < num_entities):
                reasons.append("entity_id_out_of_range")
            if not 0 <= q.relation < num_relations:
                reasons.append("relation_id_out_of_range")
            if not 0 <= q.time < num_times:
                reasons.append("time_id_out_of_range")
            if q.head not in entity_types or q.tail not in entity_types:
                reasons.append("entity_missing_from_type_mapping")
            pair = (q.head, q.relation, q.tail, q.time)
            if pair not in seen_quads:
                accepted.add(pair)
                seen_quads.add(pair)
            lo, hi = bounds[split]
            if not lo <= q.time <= hi:
                reasons.append("timestamp_outside_split_range")
            records.append(AuditRecord(split, row_index, [q.head, q.relation, q.tail, q.time],
                                       "reject" if reasons else "accept", reasons))
        all_qsets[split] = accepted

    split_order = ["train", "valid", "test"]
    temporal_findings = []
    for left, right in zip(split_order, split_order[1:]):
        if bounds[left][1] >= bounds[right][0]:
            temporal_findings.append({"rule": "chronological_splits_disjoint", "decision": "reject", "left": left, "right": right})
        else:
            temporal_findings.append({"rule": "chronological_splits_disjoint", "decision": "accept", "left": left, "right": right})

    cross_split = []
    for i, left in enumerate(split_order):
        for right in split_order[i + 1:]:
            overlap = len(all_qsets[left] & all_qsets[right])
            cross_split.append({"rule": "no_exact_quadruple_overlap", "decision": "reject" if overlap else "accept",
                                "left": left, "right": right, "overlap_count": overlap})

    type_counts = Counter(entity_types.values())
    repeated_quadruple_counts = {}
    repeated_fact_counts = {}
    for split, rows in splits.items():
        exact_counts = Counter((q.head, q.relation, q.tail, q.time) for q in rows)
        fact_times: dict[tuple[int, int, int], set[int]] = defaultdict(set)
        for q in rows:
            fact_times[(q.head, q.relation, q.tail)].add(q.time)
        repeated_quadruple_counts[split] = sum(count - 1 for count in exact_counts.values() if count > 1)
        repeated_fact_counts[split] = sum(len(times) - 1 for times in fact_times.values() if len(times) > 1)
    report = {
        "dataset": root.name,
        "rule_scope": "graph schema and temporal consistency only; no source evidence or objective truth available",
        "declared_counts": {"entities": num_entities, "relations": num_relations, "time_points": num_times},
        "observed_counts": {"entity_mapping_rows": len(entity_types), "type_ids": len(type_counts),
                            "type_entity_counts": dict(sorted(type_counts.items()))},
        "splits": {name: {"rows": len(rows), "time_min": bounds[name][0], "time_max": bounds[name][1],
                          "accepted": sum(r.decision == "accept" for r in records if r.split == name),
                          "rejected": sum(r.decision == "reject" for r in records if r.split == name)}
                   for name, rows in splits.items()},
        "temporal_findings": temporal_findings,
        "cross_split_findings": cross_split,
        "recurrence_statistics": {
            "exact_repeated_rows_within_split": repeated_quadruple_counts,
            "same_fact_repeated_at_later_time_rows_within_split": repeated_fact_counts,
            "interpretation": "descriptive recurrence signals; retained as valid data, never silently removed",
        },
        "rule_counts": dict(Counter(code for record in records for code in record.reason_codes)),
        "limitations": ["No raw news text or source provenance is present.",
                        "No claim-level semantic entailment or factual verification is possible.",
                        "Graph-level structural consistency is not a proxy for extraction accuracy."],
    }
    return records, report


def write_audit(dataset_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    records, report = audit_dataset(dataset_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "symbolic_audit.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record)) + "\n")
    (target / "symbolic_audit_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_filtered_dataset(dataset_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Apply hard schema checks and an early-train-learned relation/type signature rule.

    The first 80% of train time points define observed (relation, head type,
    tail type) signatures. Only the final 20% of train is constrained; the
    validation and test splits are copied unchanged.
    """
    root, target = Path(dataset_dir), Path(output_dir)
    records, report = audit_dataset(root)
    target.mkdir(parents=True, exist_ok=True)
    decisions = {(r.split, r.row_index): r.decision for r in records}
    entity_types = read_entity_types(root / "entity2id.txt")
    train_rows = read_tsv_quads(root / "train.txt")
    train_times = sorted({row.time for row in train_rows})
    if len(train_times) < 2:
        raise ValueError("Need at least two training time points to fit and apply temporal symbolic rules")
    fit_count = max(1, int(len(train_times) * 0.8))
    fit_times = set(train_times[:fit_count])
    application_times = set(train_times[fit_count:])
    signatures = Counter(
        (row.relation, entity_types[row.head], entity_types[row.tail])
        for row in train_rows if row.time in fit_times and row.head in entity_types and row.tail in entity_types
    )
    min_support = 1
    known_signatures = {signature for signature, support in signatures.items() if support >= min_support}
    rejection_records = []
    for split, filename in (("train", "train.txt"), ("valid", "valid.txt"), ("test", "test.txt")):
        source_lines = (root / filename).read_text(encoding="utf-8").splitlines()
        keep = []
        for index, line in enumerate(source_lines, 1):
            if decisions[(split, index)] != "accept":
                rejection_records.append({"split": split, "row_index": index, "decision": "reject", "reason_codes": ["hard_graph_constraint"]})
                continue
            if split == "train":
                row = train_rows[index - 1]
                if row.time in application_times:
                    signature = (row.relation, entity_types[row.head], entity_types[row.tail])
                    if signature not in known_signatures:
                        rejection_records.append({
                            "split": split, "row_index": index,
                            "quadruple": [row.head, row.relation, row.tail, row.time],
                            "decision": "reject", "reason_codes": ["relation_entity_type_signature_unseen_in_early_train"],
                            "signature": list(signature),
                        })
                        continue
            keep.append(line)
        (target / filename).write_text("\n".join(keep) + "\n", encoding="utf-8")
    for filename in ("stat.txt", "entity2id.txt", "relation2id.txt", "time2id.txt"):
        src = root / filename
        if src.exists():
            (target / filename).write_bytes(src.read_bytes())
    with (target / "rejected_rows_audit.jsonl").open("w", encoding="utf-8") as stream:
        for record in rejection_records:
            stream.write(json.dumps(record) + "\n")
    report["export"] = {
        "accepted_rows_only": True,
        "recurrence_rows_preserved": True,
        "symbolic_rule": "relation-specific (head entity type, tail entity type) signature observed in early training",
        "fit_training_timepoints": [min(fit_times), max(fit_times)],
        "application_training_timepoints": [min(application_times), max(application_times)] if application_times else [],
        "fit_signature_count": len(known_signatures),
        "minimum_signature_support": min_support,
        "filtered_by_symbolic_rule": sum("relation_entity_type_signature_unseen_in_early_train" in item.get("reason_codes", []) for item in rejection_records),
        "rejected_rows": len(rejection_records),
        "split_files": ["train.txt", "valid.txt", "test.txt"],
        "note": "validation and test are copied unchanged; symbolic constraints are fitted/applied within train only",
    }
    (target / "symbolic_filter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--export-filtered", action="store_true")
    args = parser.parse_args()
    result = (write_filtered_dataset if args.export_filtered else write_audit)(args.dataset_dir, args.output_dir)
    print(json.dumps(result, indent=2))
