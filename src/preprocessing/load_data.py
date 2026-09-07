"""Utilities for loading and validating one BDF EEG recording at a time.

The loader keeps the MNE ``Raw`` object lazy by default.  This avoids reading
the complete recording into memory until a later preprocessing step needs it.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class EventRecord:
	"""An event from a BIDS ``events.tsv`` file."""

	onset: float
	duration: float
	sample: int | None
	value: str | None
	trial_type: str


@dataclass(frozen=True)
class RecordingMetadata:
	"""Metadata needed by downstream preprocessing and segmentation."""

	sampling_frequency: float
	eeg_channel_count: int
	channel_names: tuple[str, ...]
	duration_seconds: float


@dataclass
class LoadedRecording:
	"""Loaded EEG data and the metadata/events associated with it."""

	raw: Any
	metadata: RecordingMetadata
	events: tuple[EventRecord, ...]


def _require_file(path: Path, description: str) -> Path:
	if not path.is_file():
		raise FileNotFoundError(f"{description} does not exist: {path}")
	return path


def _as_float(value: Any, field_name: str) -> float:
	try:
		return float(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"Metadata field '{field_name}' must be numeric; got {value!r}") from exc


def _as_int(value: Any, field_name: str) -> int:
	try:
		numeric_value = float(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"Metadata field '{field_name}' must be an integer; got {value!r}") from exc
	if not numeric_value.is_integer():
		raise ValueError(f"Metadata field '{field_name}' must be an integer; got {value!r}")
	return int(numeric_value)


def read_metadata_json(metadata_path: str | Path) -> dict[str, Any]:
	"""Read a recording's JSON sidecar and return its top-level fields."""

	path = _require_file(Path(metadata_path), "Metadata JSON file")
	try:
		with path.open("r", encoding="utf-8") as metadata_file:
			metadata = json.load(metadata_file)
	except json.JSONDecodeError as exc:
		raise ValueError(f"Invalid metadata JSON in {path}: {exc}") from exc
	if not isinstance(metadata, dict):
		raise ValueError(f"Metadata JSON must contain an object: {path}")
	return metadata


def read_events_tsv(events_path: str | Path) -> tuple[EventRecord, ...]:
	"""Read BIDS event rows while preserving condition labels.

	Required columns are ``onset`` and ``trial_type``.  ``duration``,
	``sample``, and ``value`` are optional because valid BIDS files may omit
	them.
	"""

	path = _require_file(Path(events_path), "Events TSV file")
	with path.open("r", encoding="utf-8-sig", newline="") as events_file:
		reader = csv.DictReader(events_file, delimiter="\t")
		fieldnames = set(reader.fieldnames or ())
		missing = {"onset", "trial_type"} - fieldnames
		if missing:
			missing_fields = ", ".join(sorted(missing))
			raise ValueError(f"Events TSV is missing required column(s): {missing_fields}")

		events = []
		for row_number, row in enumerate(reader, start=2):
			try:
				onset = float(row["onset"])
				duration_text = row.get("duration", "")
				sample_text = row.get("sample", "")
				duration = float(duration_text) if duration_text else 0.0
				sample = int(float(sample_text)) if sample_text else None
			except (TypeError, ValueError) as exc:
				raise ValueError(f"Invalid numeric event value on TSV row {row_number}") from exc

			trial_type = (row.get("trial_type") or "").strip()
			if not trial_type:
				raise ValueError(f"Missing trial_type on TSV row {row_number}")
			events.append(
				EventRecord(
					onset=onset,
					duration=duration,
					sample=sample,
					value=row.get("value") or None,
					trial_type=trial_type,
				)
			)
	return tuple(events)


def validate_metadata(
	sampling_frequency: float,
	eeg_channel_count: int,
	*,
	expected_sampling_frequency: float = 500.0,
	expected_eeg_channel_count: int = 61,
	tolerance: float = 1e-6,
) -> None:
	"""Validate the dataset-level sampling frequency and EEG channel count."""

	if abs(float(sampling_frequency) - expected_sampling_frequency) > tolerance:
		raise ValueError(
			"Unexpected sampling frequency: "
			f"expected {expected_sampling_frequency} Hz, got {sampling_frequency} Hz"
		)
	if int(eeg_channel_count) != expected_eeg_channel_count:
		raise ValueError(
			"Unexpected EEG channel count: "
			f"expected {expected_eeg_channel_count}, got {eeg_channel_count}"
		)


def _import_mne() -> Any:
	try:
		import mne
	except ImportError as exc:
		raise ImportError(
			"Loading BDF files requires MNE-Python. Install project dependencies "
			"with 'pip install mne'."
		) from exc
	return mne


def load_bdf(
	bdf_path: str | Path,
	*,
	metadata_path: str | Path | None = None,
	events_path: str | Path | None = None,
	preload: bool = False,
	expected_sampling_frequency: float = 500.0,
	expected_eeg_channel_count: int = 61,
	verbose: str | bool | None = None,
) -> LoadedRecording:
	"""Load and validate one BDF recording.

	Parameters are paths to one recording and its optional BIDS sidecars.
	``preload=False`` is intentional: MNE reads data on demand and therefore
	avoids unnecessarily loading a complete recording into memory.
	"""

	bdf_file = _require_file(Path(bdf_path), "BDF recording")
	mne = _import_mne()
	raw = mne.io.read_raw_bdf(bdf_file, preload=preload, verbose=verbose)

	eeg_channel_count = len(mne.pick_types(raw.info, eeg=True, exclude=[]))
	sampling_frequency = float(raw.info["sfreq"])
	validate_metadata(
		sampling_frequency,
		eeg_channel_count,
		expected_sampling_frequency=expected_sampling_frequency,
		expected_eeg_channel_count=expected_eeg_channel_count,
	)

	sidecar = read_metadata_json(metadata_path) if metadata_path is not None else {}
	if "SamplingFrequency" in sidecar:
		sidecar_frequency = _as_float(sidecar["SamplingFrequency"], "SamplingFrequency")
		if abs(sidecar_frequency - sampling_frequency) > 1e-6:
			raise ValueError(
				"BDF and JSON metadata disagree on sampling frequency: "
				f"{sampling_frequency} Hz vs {sidecar_frequency} Hz"
			)
	if "EEGChannelCount" in sidecar:
		sidecar_channels = _as_int(sidecar["EEGChannelCount"], "EEGChannelCount")
		if sidecar_channels != eeg_channel_count:
			raise ValueError(
				"BDF and JSON metadata disagree on EEG channel count: "
				f"{eeg_channel_count} vs {sidecar_channels}"
			)

	events = read_events_tsv(events_path) if events_path is not None else ()
	metadata = RecordingMetadata(
		sampling_frequency=sampling_frequency,
		eeg_channel_count=eeg_channel_count,
		channel_names=tuple(raw.ch_names),
		duration_seconds=float(raw.n_times / sampling_frequency),
	)
	return LoadedRecording(raw=raw, metadata=metadata, events=events)


def iter_bdf_files(dataset_root: str | Path) -> Iterator[Path]:
	"""Yield BDF recordings below a dataset root in deterministic order."""

	root = Path(dataset_root)
	if not root.is_dir():
		raise NotADirectoryError(f"Dataset root does not exist: {root}")
	yield from sorted(root.rglob("*.bdf"))
