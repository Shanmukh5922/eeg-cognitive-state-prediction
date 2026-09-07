"""Channel correlation and recorded-condition comparisons."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np


def channel_correlation(data: np.ndarray, channel_names: Sequence[str]) -> np.ndarray:
	"""Compute the Pearson correlation matrix across channels."""

	values = np.asarray(data, dtype=float)
	if values.ndim != 3 or len(channel_names) != values.shape[1]:
		raise ValueError("data must be 3-D and channel_names must match channels")
	flattened = values.transpose(1, 0, 2).reshape(values.shape[1], -1)
	return np.corrcoef(flattened)


def condition_summary(
	data: np.ndarray,
	metadata: Sequence[dict[str, Any]],
) -> list[dict[str, float | int | str]]:
	"""Compare mean absolute amplitude and variability for each condition."""

	values = np.asarray(data, dtype=float)
	if values.ndim != 3 or len(metadata) != values.shape[0]:
		raise ValueError("metadata count must match the segment dimension")
	grouped: dict[str, list[np.ndarray]] = defaultdict(list)
	for segment, item in zip(values, metadata):
		grouped[str(item["condition"])].append(segment)
	rows = []
	for condition, segments in sorted(grouped.items()):
		grouped_values = np.stack(segments)
		rows.append(
			{
				"condition": condition,
				"segments": len(segments),
				"mean_absolute_amplitude": float(np.mean(np.abs(grouped_values))),
				"mean_std": float(np.mean(np.std(grouped_values, axis=-1))),
				"mean_rms": float(np.mean(np.sqrt(np.mean(grouped_values**2, axis=-1)))),
			}
		)
	return rows


def resting_vs_cognitive(
	data: np.ndarray,
	metadata: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
	"""Return the same summaries with ``rs`` separated from cognitive trials."""

	values = np.asarray(data, dtype=float)
	if values.ndim != 3 or len(metadata) != values.shape[0]:
		raise ValueError("metadata count must match the segment dimension")
	groups: dict[str, list[np.ndarray]] = {"resting": [], "cognitive": []}
	for segment, item in zip(values, metadata):
		groups["resting" if item.get("condition") == "rs" else "cognitive"].append(segment)
	result: dict[str, dict[str, float | int]] = {}
	for group, segments in groups.items():
		if not segments:
			continue
		grouped_values = np.stack(segments)
		result[group] = {
			"segments": len(segments),
			"mean_absolute_amplitude": float(np.mean(np.abs(grouped_values))),
			"mean_std": float(np.mean(np.std(grouped_values, axis=-1))),
		}
	return result
