"""
PURPOSE:  The ONE entry point that runs the self-tests of the shared-namespace packages (data, model, trainer, evaluation) in a
          notebook namespace: `run_self_tests(globals())`. Loading a package (`<pkg>.load_into`) only defines; this runs the tests.
TAGS:     self-tests, run_self_tests, selftest, smoke test, notebook, load_into, _test_, run_pipeline_selftests,
          run_model_selftests, run_trainer_smoke_test, SYNCED_FROM
PITFALLS: Runs what is DEFINED in the namespace: every zero-argument callable named `_test_*` (the small module-level checks of data,
          evaluation and trainer, in definition order), then the entry points in ENTRY_POINTS that are present (the pipeline
          and 1h/4h preset self-tests, the model self-tests, the trainer smoke test, which trains a few tiny epochs). A namespace that holds no
          test at all raises, so a call placed before the loading cell cannot pass silently. The first failing test raises
          its own exception (no summary swallowing). Comment the call out in the notebook only when the code has not changed
          since the last run (SYNCED_FROM.txt shows the commit). signal_eval and discovery already keep their tests in their
          own explicit notebook cells and are not run here.
"""
from __future__ import annotations

import inspect

#: Heavy self-test functions, in run order; each is run only if the loaded packages defined it.
ENTRY_POINTS = ("run_pipeline_selftests", "run_hourly_4h_selftests", "run_model_selftests", "run_trainer_smoke_test")
TEST_PREFIX = "_test_"


def _takes_no_arguments(fn):
    return all(p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
               for p in inspect.signature(fn).parameters.values())


def collect_tests(ns):
    """``[(name, function)]`` to run for namespace ``ns``: the `_test_*` functions, then the ENTRY_POINTS that are defined."""
    inline = [(n, f) for n, f in ns.items() if n.startswith(TEST_PREFIX) and callable(f) and _takes_no_arguments(f)]
    entries = [(n, ns[n]) for n in ENTRY_POINTS if callable(ns.get(n))]
    return inline + entries


def run_self_tests(ns, echo=print):
    """Run every self-test defined in ``ns`` (a notebook's ``globals()``); returns the names run, raises on the first failure."""
    tests = collect_tests(ns)
    if not tests:
        raise RuntimeError("run_self_tests: no self-test is defined in this namespace; call it after the cells that load the "
                           "packages (data / model / trainer / evaluation)")
    names = []
    for name, fn in tests:
        echo(f"self-test: {name}")
        fn()
        names.append(name)
    echo(f"self-tests passed: {len(names)} ({', '.join(names)})")
    return names
