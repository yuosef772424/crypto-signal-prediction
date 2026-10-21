"""data/windows.py _stale_bars_module under build_dataset's worker threads.

CONFIG['stale_bars'] makes every coin ask for tools/stale_bars_audit.py through _stale_bars_module(). With >1 worker the first
coins ask at the same moment; a module put into sys.modules BEFORE exec_module finished was handed half-built to the other
threads ("module 'stale_bars_audit' has no attribute 'resolve_min_run'", while the thread that loaded it succeeded):
    python -m unittest tests.test_stale_bars_thread_safety -v
"""
import os
import sys
import tempfile
import threading
import unittest

from tests import test_reg_target_scale as H

# the real module has defs after its imports, so a slow import window exists; here the window is made wide on purpose
_SLOW_MODULE = """
import time
time.sleep(0.4)


def resolve_min_run(cfg, timeframe=None):
    return 7
"""


class StaleBarsModuleThreadSafetyTests(unittest.TestCase):
    def setUp(self):
        self._saved_module = sys.modules.pop("stale_bars_audit", None)
        self._tmp = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(self._tmp.name, "tools"))
        with open(os.path.join(self._tmp.name, "tools", "stale_bars_audit.py"), "w", encoding="utf-8") as f:
            f.write(_SLOW_MODULE)
        import data._loader as loader
        self._loader, self._saved_root = loader, loader.REPO_ROOT
        loader.REPO_ROOT = self._tmp.name

    def tearDown(self):
        self._loader.REPO_ROOT = self._saved_root
        sys.modules.pop("stale_bars_audit", None)
        if self._saved_module is not None:
            sys.modules["stale_bars_audit"] = self._saved_module
        self._tmp.cleanup()

    def test_parallel_first_calls_all_get_the_finished_module(self):
        ns = H._ns()
        ns = dict(ns)                                   # own copy: the module cache lives in the namespace
        ns.pop("_STALE_BARS_MODULE", None)
        n_threads = 8
        barrier = threading.Barrier(n_threads)
        results, errors = [None] * n_threads, []

        def worker(i):
            barrier.wait()
            try:
                results[i] = ns["stale_bars_min_run"]({"stale_bars": {"min_run": 3}}, "4h")
            except Exception as e:                      # noqa: BLE001 - the failure under test
                errors.append(repr(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(results, [7] * n_threads)


if __name__ == "__main__":
    unittest.main()
