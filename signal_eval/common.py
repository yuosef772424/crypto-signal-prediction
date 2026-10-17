"""
PURPOSE:  Shared imports of the signal-evaluation axis (np, pd, typing names) and the lazy-annotations switch.
TAGS:     imports, shared namespace, typing, from __future__ annotations
PITFALLS: Loaded first: later modules use np/pd/Optional/Dict/... without importing them (except those that re-import). Executed into the one shared signal_eval namespace by signal_eval/_loader.py (never imported on its own): names from other modules resolve at call time.

## ١) استيرادات
"""
from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any, Callable

import numpy as np
import pandas as pd
