"""
PURPOSE:  Guard against look-ahead: runs the end-to-end perturbation audit
          docs/research/audit/r3_00b_no_lookahead_perturbation.py (with its built-in "teeth" check) and requires exit code 0.
TAGS:     look-ahead, leakage, no future, perturbation, 4h context, HOURLY_4H_OVERRIDES, CI guard
PITFALLS: Takes about 2 minutes (synthetic 6-coin universe, builds the dataset several times). The audit script is run
          unchanged as a subprocess; never edit it to make this test pass (its teeth part must FAIL on the leaky config).
"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join("docs", "research", "audit", "r3_00b_no_lookahead_perturbation.py")


class NoLookaheadTests(unittest.TestCase):
    def test_perturbation_audit_passes(self):
        proc = subprocess.run([sys.executable, SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=1800)
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-30:])
        self.assertEqual(proc.returncode, 0, f"{SCRIPT} exited with {proc.returncode}\n--- last output ---\n{tail}")


if __name__ == "__main__":
    unittest.main()
