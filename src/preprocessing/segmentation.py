"""Event-aligned EEG segmentation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .load_data import EventRecord


SUPPORTED_CONDITIONS = frozenset({"rs", "easy", "medium", "diff"})


@dataclass
class EEGSegment:
	"""One fixed-length EEG segment and its traceability metadata."""

	data: np.ndarray
	subject_id: str
	session_id: str
	condition: str
	segment_id: int
	start_seconds: float
	end_seconds: float
	sampling_frequency: float
	channel_names: tuple[str, ...]

	@property
	def n_channels(self) -> int:
		return int(self.data.shape[0])

	@property
	def n_samples(self) -> int:
		return int(self.data.shape[1])


def _event_value(event: EventRecord | dict[str, Any] | Any, name: str, default: Any = None) -> Any:
	if isinstance(event, dict):
		return event.get(name, default)
	return getattr(event, name, default)


def _event_windows(
	event: EventRecord | dict[str, Any] | Any,
	window_duration: float,
	overlap: float,
	include_partial: bool,
) -> Iterable[tuple[float, float]]:
	onset = float(_event_value(event, "onset"))
	duration = float(_event_value(event, "duration", 0.0) or 0.0)
	if onset < 0 or duration < 0:
		raise ValueError("Event onset and duration must be non-negative")

	end = onset + duration if duration else onset + window_duration
	step = window_duration * (1.0 - overlap)
	start = onset
	while start < end:
		window_end = start + window_duration
		if window_end <= end or include_partial:
			yield start, min(window_end, end)
		if window_end >= end:
			break
		start += step


def segment_recording(
	raw: Any,
	events: Iterable[EventRecord | dict[str, Any] | Any],
	*,
	subject_id: str,
	session_id: str,
	window_duration: float = 2.0,
	overlap: float = 0.0,
	include_partial: bool = False,
	allowed_conditions: frozenset[str] = SUPPORTED_CONDITIONS,
) -> list[EEGSegment]:
	"""Create event-aligned, fixed-length EEG windows.

	A two-second window is the default because it provides useful short-time
	frequency resolution while keeping segments manageable for downstream EDA.
	Windows are generated only inside each event, preventing condition overlap
	and preserving subject/session identity for later leakage-aware splitting.
"""

	if window_duration <= 0:
		raise ValueError("window_duration must be positive")
	if not 0 <= overlap < 1:
		raise ValueError("overlap must be in the range [0, 1)")
	if not hasattr(raw, "get_data") or not hasattr(raw, "info"):
		raise TypeError("raw must be an MNE Raw-like object")

	sampling_frequency = float(raw.info["sfreq"])
	window_samples = int(round(window_duration * sampling_frequency))
	if window_samples < 1:
		raise ValueError("window_duration is too short for the recording sampling frequency")

	segments: list[EEGSegment] = []
	for event in events:
		condition = str(_event_value(event, "trial_type", "")).strip()
		if condition not in allowed_conditions:
			raise ValueError(
				f"Unsupported condition {condition!r}; expected one of {sorted(allowed_conditions)}"
			)
		for start_seconds, end_seconds in _event_windows(
			event, window_duration, overlap, include_partial
		):
			start_sample = int(round(start_seconds * sampling_frequency))
			stop_sample = start_sample + window_samples
			if stop_sample > raw.n_times:
				if not include_partial:
					break
				stop_sample = raw.n_times
			data = np.asarray(raw.get_data(start=start_sample, stop=stop_sample), dtype=np.float64)
			if data.shape[1] != window_samples and not include_partial:
				continue
			segments.append(
				EEGSegment(
					data=data,
					subject_id=str(subject_id),
					session_id=str(session_id),
					condition=condition,
					segment_id=len(segments),
					start_seconds=start_seconds,
					end_seconds=start_seconds + data.shape[1] / sampling_frequency,
					sampling_frequency=sampling_frequency,
					channel_names=tuple(raw.ch_names),
				)
			)
	return segments
