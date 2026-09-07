"""Load the window-level feature table produced by the preprocessing stage."""

from pathlib import Path

import pandas as pd


def load_feature_data(path, required_columns=("condition",)):
	"""Load a CSV or Parquet feature table for modeling.

	Person 2 should provide one row per EEG window and numerical feature
	columns, plus ``condition`` and any available metadata columns such as
	``subject_id``, ``session_id`` and ``window_id``.
	"""

	input_path = Path(path)
	if not input_path.exists():
		raise FileNotFoundError(f"Feature data file not found: {input_path}")

	if input_path.suffix.lower() == ".parquet":
		data = pd.read_parquet(input_path)
	elif input_path.suffix.lower() in {".csv", ".txt"}:
		data = pd.read_csv(input_path)
	else:
		raise ValueError("Feature data must be a CSV or Parquet file.")

	if data.empty:
		raise ValueError(f"Feature data is empty: {input_path}")

	missing_columns = [
		column for column in required_columns if column not in data.columns
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
		raise TypeError("data must be a pandas DataFrame.")

	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	if output_path.suffix.lower() == ".parquet":
		data.to_parquet(output_path, index=False)
	elif output_path.suffix.lower() in {".csv", ".txt"}:
		data.to_csv(output_path, index=False)
	else:
		raise ValueError("Feature data must be saved as CSV or Parquet.")

	return output_path
