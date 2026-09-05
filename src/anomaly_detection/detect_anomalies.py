"""Rule-based and optional ML-assisted EEG segment quality checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.preprocessing.segmentation import EEGSegment


@dataclass(frozen=True)
class AnomalyReport:
	"""Quality decision and reasons for one segment."""

	segment_id: int
	is_anomalous: bool
	reasons: tuple[str, ...]
	max_abs_amplitude: float
	standard_deviation: float
	flat_channel_fraction: float


@dataclass
class AnomalyDetectionResult:
	"""Partitioned segments and the audit report for every input segment."""

	normal_segments: list[EEGSegment]
	anomalous_segments: list[EEGSegment]
	reports: list[AnomalyReport]

	@property
	def flagged_count(self) -> int:
		return len(self.anomalous_segments)


def detect_anomalies(
	segments: Sequence[EEGSegment],
	*,
	max_abs_amplitude: float = 500e-6,
	max_standard_deviation: float = 200e-6,
	min_standard_deviation: float = 1e-8,
	flat_value_tolerance: float = 1e-9,
	max_flat_channel_fraction: float = 0.5,
	use_isolation_forest: bool = False,
	isolation_contamination: float = "auto",
	random_state: int = 42,
) -> AnomalyDetectionResult:
	"""Flag poor-quality EEG segments without deleting them.

	Rule checks identify non-finite values, excessive amplitudes, unusual
	variance, and flat channels.  An Isolation Forest can optionally add a
	data-driven flag, but it is disabled by default because its result depends
	on the recording batch and should be reviewed before use.
	"""

	if max_abs_amplitude <= 0 or max_standard_deviation <= 0:
		raise ValueError("amplitude and standard-deviation thresholds must be positive")
	if min_standard_deviation < 0 or flat_value_tolerance < 0:
		raise ValueError("flatness thresholds must be non-negative")
	if not 0 <= max_flat_channel_fraction <= 1:
		raise ValueError("max_flat_channel_fraction must be in the range [0, 1]")

	reports: list[AnomalyReport] = []
	feature_rows: list[list[float]] = []
	for segment in segments:
		data = np.asarray(segment.data)
		if data.ndim != 2:
			raise ValueError(f"Segment {segment.segment_id} must contain a 2D channels x samples array")
		finite = np.isfinite(data)
		reasons: list[str] = []
		if not finite.all():
			reasons.append("contains_nan_or_inf")
		finite_data = data[finite]
		segment_max = float(np.max(np.abs(finite_data))) if finite_data.size else float("inf")
		segment_std = float(np.std(finite_data)) if finite_data.size else float("inf")
		if segment_max > max_abs_amplitude:
			reasons.append("excessive_amplitude")
		if segment_std > max_standard_deviation:
			reasons.append("excessive_variance")
		if segment_std < min_standard_deviation:
			reasons.append("near_flat_segment")

		channel_ranges = np.ptp(np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0), axis=1)
		flat_fraction = float(np.mean(channel_ranges <= flat_value_tolerance)) if data.shape[0] else 1.0
		if flat_fraction > max_flat_channel_fraction:
			reasons.append("too_many_flat_channels")
		reports.append(
			AnomalyReport(
				segment_id=segment.segment_id,
				is_anomalous=bool(reasons),
				reasons=tuple(reasons),
				max_abs_amplitude=segment_max,
				standard_deviation=segment_std,
				flat_channel_fraction=flat_fraction,
			)
		)
		feature_rows.append([segment_max, segment_std, flat_fraction])

	if use_isolation_forest and len(segments) >= 2:
		try:
			from sklearn.ensemble import IsolationForest
		except ImportError as exc:
			raise ImportError(
				"Isolation Forest requires scikit-learn; install it or disable "
				"use_isolation_forest."
			) from exc
		model = IsolationForest(
				contamination=isolation_contamination,
				random_state=random_state,
			)
		predictions = model.fit_predict(np.asarray(feature_rows))
		for report_index, (report, prediction) in enumerate(zip(reports, predictions)):
			if prediction == -1 and "isolation_forest_outlier" not in report.reasons:
				reports[report_index] = AnomalyReport(
					segment_id=report.segment_id,
					is_anomalous=True,
					reasons=(*report.reasons, "isolation_forest_outlier"),
					max_abs_amplitude=report.max_abs_amplitude,
					standard_deviation=report.standard_deviation,
					flat_channel_fraction=report.flat_channel_fraction,
				)

	normal_segments = [segment for segment, report in zip(segments, reports) if not report.is_anomalous]
	anomalous_segments = [segment for segment, report in zip(segments, reports) if report.is_anomalous]
	return AnomalyDetectionResult(
		normal_segments=normal_segments,
		anomalous_segments=anomalous_segments,
		reports=reports,
	)
