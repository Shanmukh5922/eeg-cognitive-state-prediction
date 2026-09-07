"""Utilities for loading EEG recordings and feature tables."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


# ============================================================
# PERSON 3 - FEATURE TABLE LOADING
# ============================================================

def load_feature_data(path, required_columns=("condition",)):
    """Load a CSV or Parquet feature table for modeling.

    Person 2 should provide one row per EEG window and numerical feature
    columns, plus ``condition`` and any available metadata columns such as
    ``subject_id``, ``session_id`` and ``window_id``.
    """

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature data file not found: {input_path}"
        )

    if input_path.suffix.lower() == ".parquet":
        data = pd.read_parquet(input_path)
    elif input_path.suffix.lower() in {".csv", ".txt"}:
        data = pd.read_csv(input_path)
    else:
        raise ValueError(
            "Feature data must be a CSV or Parquet file."
        )

    if data.empty:
        raise ValueError(
            f"Feature data is empty: {input_path}"
        )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Feature data is missing required columns: "
            + ", ".join(missing_columns)
        )

    return data


def save_feature_data(data, path):
    """Save Person 2's feature table as CSV or Parquet."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".parquet":
        data.to_parquet(output_path, index=False)

    elif output_path.suffix.lower() in {".csv", ".txt"}:
        data.to_csv(output_path, index=False)

    else:
        raise ValueError(
            "Feature data must be saved as CSV or Parquet."
        )

    return output_path


# ============================================================
# MAIN - BDF EEG RECORDING LOADING
# ============================================================

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
        raise FileNotFoundError(
            f"{description} does not exist: {path}"
        )
    return path


def _as_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Metadata field '{field_name}' must be numeric; "
            f"got {value!r}"
        ) from exc


def _as_int(value: Any, field_name: str) -> int:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Metadata field '{field_name}' must be an integer; "
            f"got {value!r}"
        ) from exc

    if not numeric_value.is_integer():
        raise ValueError(
            f"Metadata field '{field_name}' must be an integer; "
            f"got {value!r}"
        )

    return int(numeric_value)


def read_metadata_json(
    metadata_path: str | Path,
) -> dict[str, Any]:
    """Read a recording's JSON sidecar."""

    path = _require_file(
        Path(metadata_path),
        "Metadata JSON file"
    )

    try:
        with path.open(
            "r",
            encoding="utf-8"
        ) as metadata_file:
            metadata = json.load(metadata_file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid metadata JSON in {path}: {exc}"
        ) from exc

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Metadata JSON must contain an object: {path}"
        )

    return metadata


def read_events_tsv(
    events_path: str | Path,
) -> tuple[EventRecord, ...]:
    """Read BIDS event rows while preserving condition labels."""

    path = _require_file(
        Path(events_path),
        "Events TSV file"
    )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as events_file:

        reader = csv.DictReader(
            events_file,
            delimiter="\t"
        )

        fieldnames = set(
            reader.fieldnames or ()
        )

        missing = {
            "onset",
            "trial_type"
        } - fieldnames

        if missing:
            missing_fields = ", ".join(
                sorted(missing)
            )

            raise ValueError(
                "Events TSV is missing required "
                f"column(s): {missing_fields}"
            )

        events = []

        for row_number, row in enumerate(
            reader,
            start=2
        ):
            try:
                onset = float(
                    row["onset"]
                )

                duration_text = row.get(
                    "duration",
                    ""
                )

                sample_text = row.get(
                    "sample",
                    ""
                )

                duration = (
                    float(duration_text)
                    if duration_text
                    else 0.0
                )

                sample = (
                    int(float(sample_text))
                    if sample_text
                    else None
                )

            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid numeric event value "
                    f"on TSV row {row_number}"
                ) from exc

            trial_type = (
                row.get("trial_type") or ""
            ).strip()

            if not trial_type:
                raise ValueError(
                    f"Missing trial_type on TSV row {row_number}"
                )

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
    """Validate sampling frequency and EEG channel count."""

    if (
        abs(
            float(sampling_frequency)
            - expected_sampling_frequency
        )
        > tolerance
    ):
        raise ValueError(
            "Unexpected sampling frequency: "
            f"expected {expected_sampling_frequency} Hz, "
            f"got {sampling_frequency} Hz"
        )

    if (
        int(eeg_channel_count)
        != expected_eeg_channel_count
    ):
        raise ValueError(
            "Unexpected EEG channel count: "
            f"expected {expected_eeg_channel_count}, "
            f"got {eeg_channel_count}"
        )


def _import_mne() -> Any:
    try:
        import mne

    except ImportError as exc:
        raise ImportError(
            "Loading BDF files requires MNE-Python. "
            "Install project dependencies with "
            "'pip install mne'."
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
    """Load and validate one BDF recording."""

    bdf_file = _require_file(
        Path(bdf_path),
        "BDF recording"
    )

    mne = _import_mne()

    raw = mne.io.read_raw_bdf(
        bdf_file,
        preload=preload,
        verbose=verbose
    )

    eeg_channel_count = len(
        mne.pick_types(
            raw.info,
            eeg=True,
            exclude=[]
        )
    )

    sampling_frequency = float(
        raw.info["sfreq"]
    )

    validate_metadata(
        sampling_frequency,
        eeg_channel_count,
        expected_sampling_frequency=(
            expected_sampling_frequency
        ),
        expected_eeg_channel_count=(
            expected_eeg_channel_count
        ),
    )

    sidecar = (
        read_metadata_json(metadata_path)
        if metadata_path is not None
        else {}
    )

    if "SamplingFrequency" in sidecar:
        sidecar_frequency = _as_float(
            sidecar["SamplingFrequency"],
            "SamplingFrequency"
        )

        if (
            abs(
                sidecar_frequency
                - sampling_frequency
            )
            > 1e-6
        ):
            raise ValueError(
                "BDF and JSON metadata disagree "
                "on sampling frequency: "
                f"{sampling_frequency} Hz vs "
                f"{sidecar_frequency} Hz"
            )

    if "EEGChannelCount" in sidecar:
        sidecar_channels = _as_int(
            sidecar["EEGChannelCount"],
            "EEGChannelCount"
        )

        if sidecar_channels != eeg_channel_count:
            raise ValueError(
                "BDF and JSON metadata disagree "
                "on EEG channel count: "
                f"{eeg_channel_count} vs "
                f"{sidecar_channels}"
            )

    events = (
        read_events_tsv(events_path)
        if events_path is not None
        else ()
    )

    metadata = RecordingMetadata(
        sampling_frequency=sampling_frequency,
        eeg_channel_count=eeg_channel_count,
        channel_names=tuple(raw.ch_names),
        duration_seconds=float(
            raw.n_times / sampling_frequency
        ),
    )

    return LoadedRecording(
        raw=raw,
        metadata=metadata,
        events=events
    )


def iter_bdf_files(
    dataset_root: str | Path
) -> Iterator[Path]:
    """Yield BDF recordings below a dataset root."""

    root = Path(dataset_root)

    if not root.is_dir():
        raise NotADirectoryError(
            f"Dataset root does not exist: {root}"
        )

    yield from sorted(
        root.rglob("*.bdf")
    )