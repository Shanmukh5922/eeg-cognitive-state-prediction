"""EDA-informed EEG feature extraction and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.eda.frequency_domain import band_powers
from src.eda.time_domain import summarize_time_domain


@dataclass(frozen=True)
class FeatureSelectionResult:
	"""Selected feature matrix and its metadata for downstream modelling."""

	values: np.ndarray
	feature_names: tuple[str, ...]
	metadata: tuple[dict[str, Any], ...]
	scores: dict[str, float]


def build_feature_matrix(
	data: np.ndarray,
	sampling_frequency: float,
	channel_names: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
	"""Build targeted amplitude, variability, and relative band-power features."""

	summaries = summarize_time_domain(data, channel_names)
	powers, bands = band_powers(data, sampling_frequency, relative=True)
	names: list[str] = []
	columns: list[np.ndarray] = []
	for metric in ("std", "rms", "mean_absolute"):
		for channel_index, channel in enumerate(channel_names):
			names.append(f"{channel}_{metric}")
			columns.append(summaries[metric][:, channel_index])
	for band_index, band in enumerate(bands):
		for channel_index, channel in enumerate(channel_names):
			names.append(f"{channel}_{band}_relative_power")
			columns.append(powers[:, channel_index, band_index])
	return np.column_stack(columns), tuple(names)


def select_features(
	data: np.ndarray,
	metadata: Sequence[dict[str, Any]],
	sampling_frequency: float,
	channel_names: Sequence[str],
	*,
	max_features: int = 30,
	min_variance: float = 0.0,
) -> FeatureSelectionResult:
	"""Select a compact feature set using variance and condition separation.

	Within-condition variability is used to avoid ranking a feature highly only
	because of a single outlier segment. When condition labels are unavailable,
	the total variance is used as the fallback score.
	"""

	values, names = build_feature_matrix(data, sampling_frequency, channel_names)
	if len(metadata) != values.shape[0]:
		raise ValueError("metadata count must match the segment dimension")
	if max_features < 1:
		raise ValueError("max_features must be positive")
	variances = np.var(values, axis=0)
	labels = np.asarray([str(item.get("condition", "")) for item in metadata])
	unique_labels = np.unique(labels)
	scores_array = variances.copy()
	if len(unique_labels) > 1:
		grand_mean = np.mean(values, axis=0)
		between = np.zeros(values.shape[1])
		within = np.zeros(values.shape[1])
		for label in unique_labels:
			group = values[labels == label]
			between += len(group) * (np.mean(group, axis=0) - grand_mean) ** 2
			within += np.sum((group - np.mean(group, axis=0)) ** 2, axis=0)
		scores_array = between / np.maximum(within, np.finfo(float).eps)
	eligible = np.flatnonzero(variances > min_variance)
	if not len(eligible):
		eligible = np.arange(values.shape[1])
	ranked = eligible[np.argsort(scores_array[eligible])[::-1][:max_features]]
	selected_names = tuple(names[index] for index in ranked)
	scores = {names[index]: float(scores_array[index]) for index in ranked}
	return FeatureSelectionResult(values[:, ranked], selected_names, tuple(metadata), scores)
