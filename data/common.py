"""
PURPOSE:  Shared imports (numpy, pandas, typing, concurrency) and the TypeVars T/R used by every other pipeline module.
TAGS:     imports, shared namespace, typing, TypeVar, from __future__
PITFALLS: Loaded first: every later module relies on these names (np, pd, Optional, deepcopy, Path...) without importing them. The __future__ import applies to this file only (each module is compiled on its own). Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 2) الاستيرادات العامة
"""
from __future__ import annotations

import gc
import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd

T = TypeVar('T')
R = TypeVar('R')
