import tempfile
import unittest
from pathlib import Path

from graph_symbolic import audit_dataset, write_audit, write_filtered_dataset


class GraphSymbolicTests(unittest.TestCase):
    def test_audit_accepts_clean_temporal_fixture(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "fixture"
            root.mkdir()
            (root / "stat.txt").write_text("2\t1\t3\n", encoding="utf-8")
            (root / "entity2id.txt").write_text("A\t0\tCOMP\t0\nB\t1\tCOMP\t0\n", encoding="utf-8")
            (root / "train.txt").write_text("0\t0\t1\t0\t0\n", encoding="utf-8")
            (root / "valid.txt").write_text("1\t0\t0\t1\t0\n", encoding="utf-8")
            (root / "test.txt").write_text("0\t0\t1\t2\t0\n", encoding="utf-8")
            records, report = audit_dataset(root)
            self.assertEqual([r.decision for r in records], ["accept", "accept", "accept"])
            self.assertTrue(all(f["decision"] == "accept" for f in report["temporal_findings"]))
            self.assertEqual(report["splits"]["train"]["rows"], 1)

    def test_audit_retains_recurrence_and_rejects_out_of_range(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "fixture"
            root.mkdir()
            (root / "stat.txt").write_text("2\t1\t3\n", encoding="utf-8")
            (root / "entity2id.txt").write_text("A\t0\tCOMP\t0\nB\t1\tCOMP\t0\n", encoding="utf-8")
            (root / "train.txt").write_text("0\t0\t1\t0\t0\n0\t0\t1\t0\t0\n9\t0\t1\t4\t0\n", encoding="utf-8")
            (root / "valid.txt").write_text("1\t0\t0\t1\t0\n", encoding="utf-8")
            (root / "test.txt").write_text("0\t0\t1\t2\t0\n", encoding="utf-8")
            records, report = audit_dataset(root)
            self.assertEqual(records[1].decision, "accept")
            self.assertIn("entity_id_out_of_range", records[2].reason_codes)
            self.assertIn("time_id_out_of_range", records[2].reason_codes)
            self.assertEqual(report["recurrence_statistics"]["exact_repeated_rows_within_split"]["train"], 1)

    def test_audit_files_are_written(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "d"
            root.mkdir()
            (root / "stat.txt").write_text("1\t1\t3\n", encoding="utf-8")
            (root / "entity2id.txt").write_text("A\t0\tCOMP\t0\n", encoding="utf-8")
            (root / "train.txt").write_text("0\t0\t0\t0\t0\n", encoding="utf-8")
            (root / "valid.txt").write_text("0\t0\t0\t1\t0\n", encoding="utf-8")
            (root / "test.txt").write_text("0\t0\t0\t2\t0\n", encoding="utf-8")
            out = base / "out"
            write_audit(root, out)
            self.assertTrue((out / "symbolic_audit.jsonl").exists())
            self.assertTrue((out / "symbolic_audit_summary.json").exists())

    def test_symbolic_rule_fits_early_train_and_leaves_eval_splits_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "d"
            root.mkdir()
            (root / "stat.txt").write_text("3\t1\t6\n", encoding="utf-8")
            (root / "entity2id.txt").write_text(
                "A\t0\tCOMP\t0\nB\t1\tCOMP\t0\nP\t2\tPERSON\t1\n", encoding="utf-8"
            )
            (root / "train.txt").write_text(
                "0\t0\t1\t0\t0\n0\t0\t1\t1\t0\n0\t0\t1\t2\t0\n"
                "0\t0\t1\t3\t0\n0\t0\t2\t3\t0\n", encoding="utf-8"
            )
            (root / "valid.txt").write_text("0\t0\t2\t4\t0\n", encoding="utf-8")
            (root / "test.txt").write_text("2\t0\t0\t5\t0\n", encoding="utf-8")
            out = base / "filtered"
            report = write_filtered_dataset(root, out)
            self.assertEqual(len((out / "valid.txt").read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(len((out / "test.txt").read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(report["export"]["fit_training_timepoints"], [0, 2])
            self.assertEqual(report["export"]["filtered_by_symbolic_rule"], 1)


if __name__ == "__main__":
    unittest.main()
