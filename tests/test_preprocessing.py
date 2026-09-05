from pathlib import Path

import mne
import numpy as np
import pytest

from src.anomaly_detection.detect_anomalies import detect_anomalies
from src.preprocessing.filters import FilterConfig, filter_raw
from src.preprocessing.load_data import (
	EventRecord,
	read_events_tsv,
	validate_metadata,
)
from src.preprocessing.segmentation import segment_recording


def make_raw(sfreq: float = 100.0, n_times: int = 1000) -> mne.io.RawArray:
	info = mne.create_info(["C3", "C4"], sfreq=sfreq, ch_types="eeg")
	data = np.zeros((2, n_times))
	data[0] = 20e-6 * np.sin(np.linspace(0, 20 * np.pi, n_times))
	return mne.io.RawArray(data, info, verbose=False)


def test_events_tsv_and_metadata_validation(tmp_path: Path) -> None:
	events_path = tmp_path / "events.tsv"
	events_path.write_text(
		"onset\tduration\tsample\tvalue\ttrial_type\n"
		"1.0\t2.0\t100\t1\teasy\n",
		encoding="utf-8",
	)
	events = read_events_tsv(events_path)
	assert events == (EventRecord(1.0, 2.0, 100, "1", "easy"),)
	validate_metadata(500, 61)
	with pytest.raises(ValueError, match="sampling frequency"):
		validate_metadata(250, 61)


def test_events_tsv_rejects_missing_columns(tmp_path: Path) -> None:
	events_path = tmp_path / "events.tsv"
	events_path.write_text("onset\tduration\n0\t1\n", encoding="utf-8")
	with pytest.raises(ValueError, match="trial_type"):
		read_events_tsv(events_path)


def test_filter_raw_preserves_original() -> None:
	raw = make_raw(sfreq=500.0, n_times=10000)
	original = raw.get_data().copy()
	filtered = filter_raw(raw, config=FilterConfig(), verbose=False)
	assert filtered is not raw
	assert filtered.get_data().shape == raw.get_data().shape
	np.testing.assert_array_equal(raw.get_data(), original)


def test_segment_recording_preserves_labels_and_timing() -> None:
	segments = segment_recording(
		make_raw(),
		[EventRecord(1.0, 4.0, 100, None, "medium")],
		subject_id="sub-1",
		session_id="ses-2",
		window_duration=2.0,
	)
	assert len(segments) == 2
	assert segments[0].condition == "medium"
	assert segments[0].subject_id == "sub-1"
	assert segments[0].session_id == "ses-2"
	assert segments[0].start_seconds == 1.0
	assert segments[0].data.shape == (2, 200)


def test_segment_recording_rejects_unknown_condition() -> None:
	with pytest.raises(ValueError, match="Unsupported condition"):
		segment_recording(
			make_raw(),
			[EventRecord(0.0, 2.0, 0, None, "unknown")],
			subject_id="sub-1",
			session_id="ses-1",
		)


def test_anomaly_detection_returns_audit_reasons() -> None:
	segments = segment_recording(
		make_raw(),
		[EventRecord(0.0, 2.0, 0, None, "rs")],
		subject_id="sub-1",
		session_id="ses-1",
	)
	segments[0].data[0, 0] = np.nan
	result = detect_anomalies(segments)
	assert result.flagged_count == 1
	assert "contains_nan_or_inf" in result.reports[0].reasons


def test_anomaly_detection_rejects_invalid_threshold() -> None:
	with pytest.raises(ValueError, match="thresholds"):
		detect_anomalies([], max_abs_amplitude=0)
