import tempfile
import unittest
from pathlib import Path

from neurosymbolic import Article, CandidateClaim, verify_claim, adjudicate, export_verified_quadruples


class ClaimVerificationTests(unittest.TestCase):
    def setUp(self):
        self.article = Article("a1", "Acme announced a merger with Beta on 2024-01-02.", "2024-01-03", "Example")
        self.claim = CandidateClaim(
            "c1", "a1", "Acme", "Merge", "Beta", "COMP", "COMP", "2024-01-02",
            "Acme announced a merger with Beta on 2024-01-02."
        )

    def test_verbatim_evidence_abstains_by_default(self):
        result = verify_claim(self.claim, self.article)
        self.assertEqual(result.decision, "abstain")
        self.assertTrue(result.evidence_supported)
        self.assertEqual(adjudicate(self.claim, result).decision, "abstain")

    def test_missing_evidence_rejected(self):
        self.claim.evidence = "Acme acquired Beta"
        result = verify_claim(self.claim, self.article)
        self.assertEqual(result.decision, "reject")
        self.assertIn("evidence_not_found_verbatim", result.reason_codes)

    def test_future_event_date_rejected(self):
        self.claim.event_time = "2024-01-04"
        result = verify_claim(self.claim, self.article)
        self.assertEqual(result.decision, "reject")
        self.assertIn("event_after_publication", result.reason_codes)

    def test_type_conflict_rejected(self):
        result = verify_claim(self.claim, self.article, {"Acme": "PERSON"})
        self.assertEqual(result.decision, "reject")
        self.assertTrue(any(reason.startswith("entity_type_conflict") for reason in result.reason_codes))

    def test_graph_export_keeps_provenance(self):
        verification = verify_claim(self.claim, self.article)
        verification.decision = "accept"  # export contract test only; no factual claim implied
        with tempfile.TemporaryDirectory() as td:
            graph = Path(td) / "train.txt"
            provenance = Path(td) / "audit.jsonl"
            count = export_verified_quadruples(
                [self.claim], [verification], {"Acme": 0, "Beta": 1}, {"Merge": 2}, {"2024-01-02": 0}, graph, provenance
            )
            self.assertEqual(count, 1)
            self.assertEqual(graph.read_text(encoding="utf-8"), "0\t2\t1\t0\t0\n")
            self.assertIn('"claim_id": "c1"', provenance.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
