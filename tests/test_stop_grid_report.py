"""tools/stop_grid_report.py on a tiny hand-computed trade table (concentration, leave-one-coin-out, sanity flags, edge cases)
and the runner's --report-only guarantees (never touches test_frozen.csv / verdict.json). SYNTHETIC data only.
    python -m pytest tests/test_stop_grid_report.py -q
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import stop_grid_report as rp  # noqa: E402

T0 = 1_700_000_000_000 // 86_400_000 * 86_400_000


def table(nets_a, nets_b, qv=50_000.0, gap=False, ordering="lb"):
    rows = []
    for coin, nets in (("A", nets_a), ("B", nets_b)):
        for i, n in enumerate(nets):
            ts = T0 + i * 86_400_000 + (3 if coin == "A" else 5) * 3_600_000
            rows.append(dict(coin=coin, ordering=ordering, cost_mult=1.0, net_bp=float(n), gross_bp=float(n) + 10.0, fee_bp=8.0, slip_bp=2.0,
                             entry_ts=ts, exit_ts=ts + 600_000, exit_reason="sl", hold_min=10.0, entry_quote_volume=qv, entry_range_bp=8.0,
                             gap_fill=gap, day=str(pd.Timestamp(ts, unit="ms").date())))
    return pd.DataFrame(rows)


class TestReport(unittest.TestCase):
    def setUp(self):
        self.df = table([100, 50, -10, -10, -10], [20, -20, -20, -20, -20])
        self.rep = rp.build_report(self.df, cost_bp=12.0)
        self.s = self.rep["scopes"]["x1|all|pooled"]

    def test_totals_and_concentration(self):
        s = self.s
        self.assertEqual(s["n"], 10)
        self.assertAlmostEqual(s["net_total"], 60.0)
        c = s["concentration"]
        self.assertAlmostEqual(c["by_net"]["top1"]["sum"], 100.0)
        self.assertAlmostEqual(c["by_net"]["top1"]["share"], 100 / 60)
        self.assertEqual(c["by_net"]["top10"]["k"], 10)
        self.assertAlmostEqual(c["by_net"]["top10"]["share"], 1.0)
        self.assertEqual(c["by_net"]["top1pct"]["k"], 1)          # ceil(10 * 1%) = 1
        self.assertAlmostEqual(c["by_gross"]["top1"]["sum"], 110.0)  # gross = net + 10
        self.assertAlmostEqual(c["after_removing"]["top100"]["net_total_after"], 0.0)
        self.assertIsNone(c["after_removing"]["top100"]["net_mean_after"])
        self.assertAlmostEqual(c["after_removing"]["top1pct"]["net_total_after"], -40.0)
        self.assertAlmostEqual(c["after_removing"]["top1pct"]["net_mean_after"], -40 / 9)
        self.assertAlmostEqual(c["best5_days_share_of_net"], 1.0)   # only 5 days exist

    def test_leave_one_coin_out_and_biggest_coin(self):
        c = self.s["concentration"]
        self.assertAlmostEqual(c["leave_one_coin_out"]["A"]["rest_net_total"], -60.0)
        self.assertAlmostEqual(c["leave_one_coin_out"]["A"]["rest_net_mean"], -12.0)
        self.assertAlmostEqual(c["leave_one_coin_out"]["B"]["rest_net_total"], 120.0)
        self.assertEqual(c["biggest_coin"]["coin"], "A")
        self.assertAlmostEqual(c["biggest_coin"]["share_of_net"], 2.0)
        self.assertAlmostEqual(c["biggest_coin"]["rest_net_total"], -60.0)

    def test_distribution(self):
        d = self.s["distribution"]
        self.assertAlmostEqual(d["win_rate"], 0.3)
        self.assertAlmostEqual(d["avg_win"], 170 / 3)
        self.assertAlmostEqual(d["avg_loss"], -110 / 7)
        self.assertAlmostEqual(d["profit_factor"], 170 / 110)
        self.assertAlmostEqual(d["payoff_ratio"], (170 / 3) / (110 / 7))
        self.assertEqual(d["largest_win"], 100.0)
        self.assertEqual(d["largest_loss"], -20.0)
        self.assertEqual(sum(h["n"] for h in d["histogram"]), 10)
        self.assertGreaterEqual(d["max_loss_streak"], 4)

    def test_equity_and_groups(self):
        e = self.s["equity"]
        self.assertAlmostEqual(e["final"], 60.0)
        self.assertGreater(e["max_drawdown_bp"], 0)
        self.assertEqual(self.s["by_exit_reason"][0]["n"], 10)
        self.assertEqual({r["key"] for r in self.s["by_hour_utc"]}, {"3", "5"})
        self.assertEqual(sum(r["n"] for r in self.s["by_entry_quote_volume"]), 10)
        self.assertEqual(self.s["by_day"]["n_days"], 5)

    def test_scopes_cover_orderings_and_cost(self):
        df = pd.concat([table([1, 2], [3], ordering="lb"), table([5], [6], ordering="ub").assign(cost_mult=2.0)])
        rep = rp.build_report(df)
        for k in ("x1|all|pooled", "x1|lb|A", "x2|ub|pooled", "x2|all|B"):
            self.assertIn(k, rep["scopes"])

    def test_flags(self):
        # thin + gap + gross way above the range + null that earns money
        df = table([30] * 5, [30] * 5, qv=1_000.0, gap=True)
        bar = {"A": {"median_range_bp": 5.0, "median_qv": 1_000.0, "n_days": 0.5}, "B": {"median_range_bp": 5.0, "median_qv": 1_000.0, "n_days": 0.5}}
        nul = pd.DataFrame({"null": ["random_walk"] * 3 + ["sign_flip"] * 3, "net_lb": [20.0, 21.0, 22.0, -13.0, -12.0, -14.0]})
        rep = rp.build_report(df, nulls=nul, cost_bp=12.0, bar_stats=bar, real_net_lb=30.0)
        st = {f["flag"]: f["status"] for f in rep["flags"]}
        self.assertEqual(st["mean_gross_vs_median_bar_range"], "WARN")
        self.assertEqual(st["fills_per_day_vs_median_quote_volume"], "WARN")
        self.assertEqual(st["share_entry_quote_volume_below_20k"], "WARN")
        self.assertEqual(st["share_gap_fills"], "WARN")
        self.assertEqual(st["null_vs_real"], "WARN")
        tab = {r["null"]: r["status"] for r in next(f for f in rep["flags"] if f["flag"] == "null_vs_real")["table"]}
        self.assertEqual(tab, {"random_walk": "WARN", "sign_flip": "PASS"})
        md = rp.render_markdown(rep)
        self.assertIn("WARN", md.split("الملخّص")[0])
        ok = rp.build_report(table([1] * 5, [1] * 5), nulls=nul[nul["null"] == "sign_flip"], cost_bp=12.0,
                             bar_stats={"A": {"median_range_bp": 50.0, "median_qv": 1e6, "n_days": 5}, "B": {"median_range_bp": 50.0, "median_qv": 1e6, "n_days": 5}})
        self.assertTrue(all(f["status"] == "PASS" for f in ok["flags"]), ok["flags"])

    def test_empty_and_single_trade(self):
        empty = table([], [])
        rep = rp.build_report(empty)
        self.assertEqual(rep["n_trades_total"], 0)
        self.assertEqual(rep["flags"][0]["status"], "NA")
        self.assertIn("no_trades", rp.render_markdown(rep))
        one = rp.build_report(table([7], []))
        s = one["scopes"]["x1|all|pooled"]
        self.assertEqual(s["n"], 1)
        self.assertIsNone(s["distribution"]["skew"])
        self.assertIsNone(s["distribution"]["profit_factor"])
        json.dumps(one)
        rp.render_markdown(one)

    def test_json_safe(self):
        json.dumps(self.rep)


class TestRunnerReportOnly(unittest.TestCase):
    def test_smoke_then_report_only_leaves_frozen_files_alone(self):
        runner = os.path.join(ROOT, "research", "stop_grid_1m", "run_stop_grid_1m.py")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            r = subprocess.run([sys.executable, runner, "--smoke", "--out", out], capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
            data = next(l.split(": ", 1)[1].strip() for l in r.stdout.splitlines() if l.startswith("SMOKE DATA DIR"))
            self.assertTrue(os.path.exists(os.path.join(out, "report", "report.md")))
            sha = {f: hashlib.sha256(open(os.path.join(out, f), "rb").read()).hexdigest() for f in ("test_frozen.csv", "verdict.json", "frozen_config.json")}
            r = subprocess.run([sys.executable, runner, "--data", data, "--out", out, "--report-only", "--report-scope", "both", "--log-events"],
                               capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
            for f, h in sha.items():
                self.assertEqual(hashlib.sha256(open(os.path.join(out, f), "rb").read()).hexdigest(), h, f)
            rd = os.path.join(out, "report")
            for f in ("trades.csv.gz", "events.csv.gz", "report.md", "report.json"):
                self.assertTrue(os.path.exists(os.path.join(rd, f)), f)
            tr = pd.read_csv(os.path.join(rd, "trades.csv.gz"))
            self.assertTrue({"ordering", "coin", "exit_reason", "entry_quote_volume", "net_bp", "rebuild_id", "split"} <= set(tr.columns))
            self.assertIn("descriptive re-run", open(os.path.join(rd, "report.md"), encoding="utf-8").read())


if __name__ == "__main__":
    unittest.main()
