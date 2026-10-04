"""tools/experiment_registry.py: سجلّ الفشل وقاعدة «لا إعادة بلا سبب» — والسجلّ الحقيقي وكل بطاقات المستودع صالحة (CI).
    python -m unittest tests.test_experiment_registry -v
"""
import csv
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import experiment_registry as er  # noqa: E402


def _row(**kw):
    r = {c: "" for c in er.COLUMNS}
    r.update(id="F-0001", source="study H1", date="2026-10-04", claim="momentum direction on 1h", mechanism="momentum",
             timeframe="1h; 4h", target="direction", model_class="any", failure_level="economics",
             evidence="net -8bp, t=-3", verified_cause="gross edge < cost", cause_test="zero-cost rerun ~0bp",
             invariants="gross < 14bp RT", reopen_if="R1: maker round-trip <= 2bp | R2: new non-OHLCV information",
             status="closed")
    r.update(kw)
    return r


def _card(**meta):
    lines = ["---"] + [f"{k}: {v}" for k, v in meta.items()] + ["---", "# card"]
    f = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8")
    f.write("\n".join(lines))
    f.close()
    return f.name


SCOPE = dict(mechanism="momentum; trend", timeframe="1h", target="direction", model_class="cnn")


class RegistryTests(unittest.TestCase):
    def test_real_registry_and_all_cards_are_valid(self):
        self.assertEqual(er.main(["validate"]), 0)

    def test_schema_errors(self):
        self.assertEqual(er.validate_registry([_row()], er.COLUMNS), [])
        bad = [_row(id="X1"), _row(status="closed", verified_cause=""), _row(id="F-0002", reopen_if="whenever"),
               _row(id="F-0003", failure_level="vibes"), _row(id="F-0004", mechanism=""),
               _row(id="F-0005", supersedes="F-0099")]
        msgs = " | ".join(er.validate_registry(bad))
        for needle in ("F-0001", "verified_cause", "reopen_if", "failure_level", "'mechanism'", "supersedes"):
            self.assertIn(needle, msgs)

    def test_covered_card_must_reopen_or_explain(self):
        rows = [_row()]
        self.assertTrue(any("F-0001" in e for e in er.check_card(_card(**SCOPE), rows)))           # silent repeat
        self.assertEqual(er.check_card(_card(**SCOPE, reopens="F-0001:R1"), rows), [])           # named condition
        self.assertTrue(er.check_card(_card(**SCOPE, reopens="F-0001:R9"), rows))                 # unknown condition
        self.assertTrue(er.check_card(_card(**SCOPE, reopens="F-0001"), rows))                    # no condition
        self.assertEqual(er.check_card(_card(**SCOPE, not_covered_by="F-0001=target is volatility size, not sign"),
                                       rows), [])
        self.assertTrue(er.check_card(_card(**SCOPE, not_covered_by="F-0001=new"), rows))         # no real reason

    def test_scope_overlap_and_supersede(self):
        rows = [_row()]
        self.assertEqual(er.covering(dict(SCOPE, timeframe="1d"), rows), [])                       # timeframe differs
        self.assertEqual(er.covering(dict(SCOPE, target="volatility"), rows), [])                  # target differs
        self.assertEqual(len(er.covering(dict(SCOPE, model_class="logistic"), rows)), 1)           # 'any' matches
        rows.append(_row(id="F-0002", supersedes="F-0001", reopen_if="R1: x"))
        self.assertEqual([r["id"] for r in er.covering(SCOPE, rows)], ["F-0002"])
        self.assertEqual(er.covering(SCOPE, [_row(status="reopened")]), [])

    def test_card_needs_front_matter_and_scope(self):
        f = _card(mechanism="momentum")
        self.assertTrue(er.check_card(f, []))
        g = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
        g.write("# no front matter")
        g.close()
        with self.assertRaises(ValueError):
            er.parse_card(g.name)

    def test_registry_header_is_exact(self):
        with open(er.REGISTRY, newline="", encoding="utf-8") as f:
            self.assertEqual(next(csv.reader(f)), er.COLUMNS)


if __name__ == "__main__":
    unittest.main()
