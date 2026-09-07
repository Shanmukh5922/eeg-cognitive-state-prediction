"""Feature selection utilities for the Person 2 to Person 3 handoff."""

import pandas as pd


DEFAULT_METADATA_COLUMNS = {
	"condition",
	"label",
	"subject_id",
	"session_id",
	"window_id",
}


def get_numeric_features(data, excluded_columns=None):
	"""Return numeric feature names, excluding labels and identifiers."""

	if not isinstance(data, pd.DataFrame):
		raise TypeError("data must be a pandas DataFrame.")

	excluded = DEFAULT_METADATA_COLUMNS | set(excluded_columns or ())
	features = [
		column for column in data.columns
		if column not in excluded
		and pd.api.types.is_numeric_dtype(data[column])
	]
	if not features:
		raise ValueError("No numerical feature columns were found.")
	return features


def select_features(data, feature_columns=None, excluded_columns=None):
	"""Return a validated feature matrix and the selected column names."""

	columns = feature_columns or get_numeric_features(
		data,
		excluded_columns=excluded_columns
	)
	missing = [column for column in columns if column not in data.columns]
	if missing:
		raise ValueError("Missing feature columns: " + ", ".join(missing))
	if data[columns].isna().any().any():
		raise ValueError("Selected features contain missing values.")
	return data[columns].copy(), list(columns)
