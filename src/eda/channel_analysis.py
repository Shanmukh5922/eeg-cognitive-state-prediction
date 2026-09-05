"""Channel-level amplitude and difference analysis."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def channel_summary(
	data: np.ndarray,
	channel_names: Sequence[str],
) -> list[dict[str, float | str]]:
	"""Summarize amplitude and variability aggregated over all segments."""

	values = np.asarray(data, dtype=float)
	if values.ndim != 3 or len(channel_names) != values.shape[1]:
		raise ValueError("data must be 3-D and channel_names must match channels")
	return [
		{
			"channel": name,
			"mean": float(np.mean(values[:, index, :])),
			"std": float(np.std(values[:, index, :])),
			"rms": float(np.sqrt(np.mean(values[:, index, :] ** 2))),
			"peak_to_peak": float(np.ptp(values[:, index, :])),
		}
		for index, name in enumerate(channel_names)
	]


def channel_difference(
	data: np.ndarray,
	channel_names: Sequence[str],
	first_channel: str,
	second_channel: str,
) -> dict[str, np.ndarray | str]:
	"""Return sample-wise difference and summary for two named channels."""

	values = np.asarray(data, dtype=float)
	names = list(channel_names)
	if values.ndim != 3 or len(names) != values.shape[1]:
		raise ValueError("data must be 3-D and channel_names must match channels")
	if first_channel not in names or second_channel not in names:
		raise ValueError("both requested channels must exist")
	difference = values[:, names.index(first_channel), :] - values[:, names.index(second_channel), :]
	return {
		"first_channel": first_channel,
		"second_channel": second_channel,
		"difference": difference,
		"mean_difference": np.mean(difference, axis=1),
		"difference_std": np.std(difference, axis=1),
	}
