"""tools/touched_ledger.py: the touched-data ledger (docs/research/plan/lock/touched.csv) is built from master, covers every registry
universe (F-0001..F-0079) and README universe, and the committed file is exactly what the generator produces.
    python -m unittest tests.test_touched_ledger -v
"""
import csv
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_lock as bl  # noqa: E402
import touched_ledger as tl  # noqa: E402

TEN = [c + "USDT" for c in "ADA BCH BNB BTC DOGE ETH SOL TRX XRP ZEC".split()]
COMMITTED = os.path.join(ROOT, "docs", "research", "plan", "lock", "touched.csv")


def registry_ids():
    with open(os.path.join(ROOT, "docs", "research", "failure_registry.csv"), encoding="utf-8", newline="") as f:
        return [r["id"] for r in csv.DictReader(f)]


class CoverageTests(unittest.TestCase):
    def test_every_registry_row_up_to_the_cutoff_is_assigned_to_one_group(self):
        ids = [i for i in registry_ids() if i <= tl.LAST_REGISTRY_ID]
        self.assertEqual(len(ids), 79)
        mapped = [i for g in tl.GROUPS for i in g["registry_ids"]]
        self.assertEqual(sorted(mapped), sorted(set(mapped)), "a registry id is in two groups")
        self.assertEqual(set(ids) - set(mapped), set())
        self.assertEqual(set(mapped) - set(ids), set())

    def test_unmapped_registry_row_raises(self):
        with self.assertRaises(ValueError):
            tl.check_registry(registry_ids() + [], mapped_ids={"F-0001"})

    def test_rows_after_the_cutoff_are_not_required(self):
        tl.check_registry(registry_ids() + ["F-0080", "F-0123"])


class UniverseTests(unittest.TestCase):
    def test_lists_are_the_repo_lists(self):
        g = {x["key"]: x for x in tl.GROUPS}
        self.assertEqual(len(g["hourly_83"]["symbols"]), 83)
        self.assertEqual(len(set(g["hourly_83"]["symbols"])), 83)
        frozen = json.load(open(os.path.join(ROOT, "docs", "research", "scripts", "entry_range_eval", "results", "frozen.json")))
        names = set()
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(k); walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
            elif isinstance(o, str) and o.endswith("USDT"):
                names.add(o)
        walk(frozen)
        self.assertEqual(names - set(g["hourly_83"]["symbols"]), set())        # the 81 coins of the frozen eval are inside the 83
        self.assertEqual(len(g["perps_195"]["symbols"]), 195)
        self.assertEqual(len(g["listings"]["symbols"]), 413)
        pending = open(os.path.join(ROOT, "research", "edge_discovery", "data_tools", "newlist_pending.txt")).read().split()
        self.assertEqual(set(g["listings"]["symbols"]), set(pending))
        self.assertEqual(g["speirsy10"]["symbols"], sorted(TEN))
        self.assertEqual(g["survey_14"]["symbols"][:2], ["AAVEUSDT", "ADAUSDT"])
        self.assertEqual(len(g["survey_14"]["symbols"]), 14)
        self.assertEqual(sorted(g["h002_5"]["symbols"]), ["DRIFTUSDT", "ILVUSDT", "PYTHUSDT", "RPLUSDT", "SOLUSDT"])

    def test_every_symbol_looks_like_a_binance_symbol(self):
        for g in tl.GROUPS:
            for s in g.get("symbols", []):
                self.assertEqual(s, s.upper(), s)
                self.assertTrue(s.endswith("USDT"), s)


class GeneratedRowsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = tl.generate()
        cls.tmp = tempfile.mkdtemp()
        cls.path = os.path.join(cls.tmp, "touched.csv")
        tl.write_csv(cls.rows, cls.path)

    def test_build_lock_accepts_the_file(self):
        parsed = bl.parse_touched(self.path)
        self.assertEqual(len(parsed), len(self.rows))

    def test_ten_speirsy_coins_at_every_timeframe(self):
        got = {r["symbol_or_rule"]: r for r in self.rows if r["kind"] == "symbol" and r["timeframe"].startswith("all")}
        self.assertEqual(set(got), set(TEN))

    def test_named_but_absent_h18_coins_are_counted(self):
        names = {r["symbol_or_rule"] for r in self.rows if r["kind"] == "symbol"}
        for c in ("LTCUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT"):
            self.assertIn(c, names)

    def test_rules(self):
        rules = {r["symbol_or_rule"] for r in self.rows if r["kind"] == "rule"}
        self.assertEqual(rules, {"folder:history_1d", "unresolved:coinmetrics-44-assets-daily"})
        self.assertFalse(any(r["symbol_or_rule"].startswith("first_bar_before") for r in self.rows))   # D-009 h: proxy dropped

    def test_touch_classes_follow_d009(self):
        by = {}
        for r in self.rows:
            if r["kind"] != "time_seal_read":
                by.setdefault(r["source"], set()).add(r["touch_class"])
        for r in self.rows:
            if r["kind"] == "time_seal_read":
                self.assertEqual(r["touch_class"], "")
            else:
                self.assertIn(r["touch_class"], ("intraday", "daily", "derivs"))
        cls = {g["key"]: g["touch_class"] for g in tl.GROUPS}
        self.assertEqual(cls, {"speirsy10": "intraday", "h002_5": "daily", "survey_14": "daily", "history_1d": "daily",
                               "hourly_83": "intraday", "perps_195": "derivs", "listings": "derivs",
                               "coinmetrics": "daily", "hl_copy": "derivs"})
        for r in self.rows:                                              # rule (a): a sub-daily timeframe read by price studies is intraday
            if r["symbol_or_rule"] in ("LTCUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT") and r["kind"] == "symbol":
                self.assertEqual(r["touch_class"], "intraday")

    def test_only_documented_groups_list_a_subdaily_registry_timeframe_without_being_intraday(self):
        with open(os.path.join(ROOT, "docs", "research", "failure_registry.csv"), encoding="utf-8", newline="") as f:
            tf = {r["id"]: r["timeframe"] for r in csv.DictReader(f)}
        sub = {"15m", "1h", "4h", "5m", "1m"}
        odd = set()
        for g in tl.GROUPS:
            if g["touch_class"] != "intraday" and any(sub & set(tf[i].split(";")) for i in g["registry_ids"]):
                odd.add(g["key"])
        self.assertEqual(odd, {"perps_195", "hl_copy"})                  # hourly metrics/OI files: level B by D-009 (b)

    def test_time_sealed_reads_are_recorded(self):
        reads = [r for r in self.rows if r["kind"] == "time_seal_read"]
        pr15 = [r for r in reads if "PR #15" in r["source"]]
        self.assertEqual(sorted(r["symbol_or_rule"] for r in pr15), sorted(TEN))
        self.assertTrue(all(r["period"] == "2026-10-01..2026-10-08" for r in pr15))
        h07 = [r for r in reads if "h07_forward" in r["source"]]
        self.assertEqual(sorted(r["symbol_or_rule"] for r in h07), sorted(TEN))
        self.assertTrue(all(r["period"] == "2026-10-01..2026-10-03" for r in h07))
        btc = next(r for r in h07 if r["symbol_or_rule"] == "BTCUSDT")
        self.assertIn("ledger.csv", btc["source"])

    def test_every_row_has_a_source_and_no_exact_duplicates(self):
        keys = [(r["symbol_or_rule"], r["kind"], r["source"]) for r in self.rows]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(r["source"].strip() for r in self.rows))

    def test_generation_is_deterministic(self):
        self.assertEqual(tl.generate(), self.rows)


class CommittedFileTests(unittest.TestCase):
    def test_committed_ledger_is_exactly_the_generator_output(self):
        self.assertTrue(os.path.exists(COMMITTED), "docs/research/plan/lock/touched.csv must be committed")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "touched.csv")
            tl.write_csv(tl.generate(), p)
            self.assertEqual(open(COMMITTED, "rb").read(), open(p, "rb").read())


if __name__ == "__main__":
    unittest.main()
