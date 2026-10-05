from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_selection.select_features import select_features


def _normalize_condition(value: object) -> str:
    """Normalize archive condition labels to the model's expected names."""
    text = str(value).strip().lower()
    if text == "rs":
        return "rest"
    return text


def build_model_feature_table(
    data_dir: str | Path = "data/processed",
    output_path: str | Path | None = None,
    max_features: int = 30,
) -> pd.DataFrame:
    """Create a one-row-per-window feature table from the processed EEG archives."""
    source_dir = Path(data_dir)
    archive_paths = sorted(source_dir.glob("*_segments.npz"))
    if not archive_paths:
        raise FileNotFoundError(f"No processed segment archives found in {source_dir}")

    all_data: list[np.ndarray] = []
    all_metadata: list[dict] = []
    channel_names: list[str] | None = None

    for archive_path in archive_paths:
        with np.load(archive_path, allow_pickle=True) as archive:
            if "data" not in archive or "metadata" not in archive:
                raise ValueError(f"Archive {archive_path} is missing required arrays.")
            segment_data = np.asarray(archive["data"], dtype=float)
            metadata = [json.loads(str(item)) for item in archive["metadata"]]

        if segment_data.size == 0:
            continue

        if channel_names is None:
            channel_names = list(metadata[0]["channel_names"])

        for item in metadata:
            item["condition"] = _normalize_condition(item.get("condition", ""))

        all_data.append(segment_data)
        all_metadata.extend(metadata)

    if not all_metadata:
        raise ValueError("No usable segment metadata was found in the processed archives.")

    combined_data = np.concatenate(all_data, axis=0)
    feature_result = select_features(
        combined_data,
        all_metadata,
        sampling_frequency=float(all_metadata[0]["sampling_frequency"]),
        channel_names=channel_names,
        max_features=max_features,
    )

    rows: list[dict] = []
    for index, item in enumerate(all_metadata):
        feature_values = feature_result.values[index]
        row = {
            "subject_id": item.get("subject_id"),
            "session_id": item.get("session_id"),
            "window_id": f"{item.get('subject_id','unknown')}_{item.get('session_id','unknown')}_{item.get('segment_id','index')}",
            "condition": item.get("condition"),
        }
        for name, value in zip(feature_result.feature_names, feature_values):
            row[name] = float(value)
        rows.append(row)

    table = pd.DataFrame(rows)
    if output_path is None:
        output_path = source_dir / "model_features.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


if __name__ == "__main__":
    build_model_feature_table()
    print("Saved model feature table to data/processed/model_features.csv")
