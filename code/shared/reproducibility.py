"""Utilities for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
	"""Set global random seed for python, numpy and torch (if installed)."""
	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	np.random.seed(seed)

	try:
		import torch

		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)

		if deterministic_torch:
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False
	except Exception:
		# Torch can be optional in some analysis-only scripts.
		pass
