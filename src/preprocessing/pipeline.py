"""Composable preprocessing pipeline for one EEG recording."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.anomaly_detection.detect_anomalies import AnomalyDetectionResult, detect_anomalies

from .filters import FilterConfig, filter_raw
from .load_data import EventRecord, LoadedRecording, load_bdf
from .segmentation import EEGSegment, segment_recording


@dataclass
class ProcessedRecording:
	"""Outputs from the modular loading, filtering, segmentation, and QC steps."""

	loaded: LoadedRecording
	filtered_raw: Any
	segments: list[EEGSegment]
	anomalies: AnomalyDetectionResult


def process_recording(
	bdf_path: str | Path,
	*,
	metadata_path: str | Path | None = None,
	events_path: str | Path | None = None,
	subject_id: str,
	session_id: str,
	filter_config: FilterConfig | None = None,
	window_duration: float = 2.0,
	overlap: float = 0.0,
	include_partial: bool = False,
	preload: bool = False,
	loader_kwargs: dict[str, Any] | None = None,
	anomaly_kwargs: dict[str, Any] | None = None,
) -> ProcessedRecording:
	"""Process one BDF recording without mixing subject/session information."""

	loaded = load_bdf(
		bdf_path,
		metadata_path=metadata_path,
		events_path=events_path,
		preload=preload,
		**(loader_kwargs or {}),
	)
	filtered_raw = filter_raw(loaded.raw, config=filter_config, copy=True)
	segments = segment_recording(
		filtered_raw,
		loaded.events,
		subject_id=subject_id,
		session_id=session_id,
		window_duration=window_duration,
		overlap=overlap,
		include_partial=include_partial,
	)
	anomalies = detect_anomalies(segments, **(anomaly_kwargs or {}))
	return ProcessedRecording(
		loaded=loaded,
		filtered_raw=filtered_raw,
		segments=segments,
		anomalies=anomalies,
	)


def save_processed_segments(
	processed: ProcessedRecording,
	output_path: str | Path,
	*,
	include_anomalous: bool = True,
) -> Path:
	"""Save segments and traceability metadata as one compressed NumPy archive.

	The archive is easy to load with ``numpy.load`` and keeps anomaly flags so
	flagged data can be reviewed instead of being silently discarded.
	"""

	segments = processed.segments if include_anomalous else processed.anomalies.normal_segments
	output = Path(output_path)
	if output.suffix != ".npz":
		output = output.with_suffix(".npz")
	output.parent.mkdir(parents=True, exist_ok=True)

	if segments:
		shapes = {segment.data.shape for segment in segments}
		if len(shapes) != 1:
			raise ValueError("All saved segments must have the same shape; disable partial windows")
		data = np.stack([segment.data for segment in segments])
	else:
		data = np.empty((0, 0, 0), dtype=np.float64)

	reports_by_id = {report.segment_id: report for report in processed.anomalies.reports}
	metadata = []
	for segment in segments:
		report = reports_by_id[segment.segment_id]
		metadata.append(
			{
				"subject_id": segment.subject_id,
				"session_id": segment.session_id,
				"condition": segment.condition,
				"segment_id": segment.segment_id,
				"start_seconds": segment.start_seconds,
				"end_seconds": segment.end_seconds,
				"sampling_frequency": segment.sampling_frequency,
				"channel_names": list(segment.channel_names),
				"is_anomalous": report.is_anomalous,
				"anomaly_reasons": list(report.reasons),
			}
		)
	metadata_array = np.asarray([json.dumps(item) for item in metadata], dtype=str)
	metadata_array = np.resize(metadata_array, len(metadata)) if metadata else metadata_array
	np.savez_compressed(output, data=data, metadata=metadata_array)
	return output


def process_dataset(
	dataset_root: str | Path,
	output_dir: str | Path,
	*,
	filter_config: FilterConfig | None = None,
	window_duration: float = 2.0,
	overlap: float = 0.0,
	include_partial: bool = False,
	overwrite: bool = False,
	anomaly_kwargs: dict[str, Any] | None = None,
) -> list[Path]:
	"""Process every BDF below a dataset root one recording at a time."""

	root = Path(dataset_root)
	if not root.is_dir():
		raise NotADirectoryError(f"Dataset root does not exist: {root}")
	output_root = Path(output_dir)
	output_root.mkdir(parents=True, exist_ok=True)
	outputs: list[Path] = []

	for bdf_path in sorted(root.rglob("*_eeg.bdf")):
		subject_id = next(
			(part for part in bdf_path.parts if part.startswith("sub-")),
			bdf_path.stem,
		)
		session_id = next(
			(part for part in bdf_path.parts if part.startswith("ses-")),
			"unknown-session",
		)
		output_path = output_root / f"{subject_id}_{session_id}_segments.npz"
		if output_path.exists() and not overwrite:
			outputs.append(output_path)
			continue

		metadata_path = bdf_path.with_suffix(".json")
		events_path = bdf_path.with_name(bdf_path.name.replace("_eeg.bdf", "_events.tsv"))
		processed = process_recording(
			bdf_path,
			metadata_path=metadata_path,
			events_path=events_path,
			subject_id=subject_id,
			session_id=session_id,
			filter_config=filter_config,
			window_duration=window_duration,
			overlap=overlap,
			include_partial=include_partial,
			anomaly_kwargs=anomaly_kwargs,
		)
		outputs.append(save_processed_segments(processed, output_path))
	return outputs