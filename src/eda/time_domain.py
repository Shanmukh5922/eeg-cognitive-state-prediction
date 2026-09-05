"""Time-domain summaries for Person 1 processed EEG segments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def load_processed_archive(
	archive_path: str | Path,
	*,
	include_anomalous: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
	"""Load a Person 1 ``.npz`` archive and its per-segment metadata."""

	with np.load(archive_path, allow_pickle=False) as archive:
		if "data" not in archive or "metadata" not in archive:
			raise ValueError("Processed archive must contain data and metadata")
		data = np.asarray(archive["data"], dtype=float)
		metadata = [json.loads(str(item)) for item in archive["metadata"]]

	if data.ndim != 3:
		raise ValueError("data must have shape (segments, channels, samples)")
	if len(metadata) != data.shape[0]:
		raise ValueError("metadata count must match the number of segments")
	if not include_anomalous:
		keep = np.asarray([not item.get("is_anomalous", False) for item in metadata])
		data = data[keep]
		metadata = [item for item, selected in zip(metadata, keep) if selected]
	return data, metadata


def _validate_data(data: np.ndarray) -> np.ndarray:
	values = np.asarray(data, dtype=float)
	if values.ndim != 3:
		raise ValueError("data must have shape (segments, channels, samples)")
	if not np.isfinite(values).all():
		raise ValueError("data must contain only finite values")
	return values


def time_domain_features(
	data: np.ndarray,
	channel_names: Sequence[str] | None = None,
) -> list[dict[str, float | int | str]]:
	"""Return amplitude, spread, and variability metrics per segment/channel."""

	values = _validate_data(data)
	names = list(channel_names or [f"channel_{i}" for i in range(values.shape[1])])
	if len(names) != values.shape[1]:
		raise ValueError("channel_names must match the channel dimension")
	rows: list[dict[str, float | int | str]] = []
	for segment_index, segment in enumerate(values):
		for channel_index, signal in enumerate(segment):
			mean = float(np.mean(signal))
			rows.append(
				{
					"segment_index": segment_index,
					"channel_index": channel_index,
					"channel": names[channel_index],
					"mean": mean,
					"std": float(np.std(signal)),
					"min": float(np.min(signal)),
					"max": float(np.max(signal)),
					"peak_to_peak": float(np.ptp(signal)),
					"rms": float(np.sqrt(np.mean(signal**2))),
					"mean_absolute": float(np.mean(np.abs(signal))),
					"median_absolute_deviation": float(
						np.median(np.abs(signal - np.median(signal)))
					),
				}
			)
	return rows


def summarize_time_domain(
	data: np.ndarray,
	channel_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
	"""Return compact metric arrays shaped ``(segments, channels)``."""

	values = _validate_data(data)
	names = list(channel_names or [f"channel_{i}" for i in range(values.shape[1])])
	if len(names) != values.shape[1]:
		raise ValueError("channel_names must match the channel dimension")
	centered = values - np.mean(values, axis=2, keepdims=True)
	return {
		"mean": np.mean(values, axis=2),
		"std": np.std(values, axis=2),
		"peak_to_peak": np.ptp(values, axis=2),
		"rms": np.sqrt(np.mean(values**2, axis=2)),
		"mean_absolute": np.mean(np.abs(values), axis=2),
		"variability": np.sqrt(np.mean(centered**2, axis=2)),
	}
