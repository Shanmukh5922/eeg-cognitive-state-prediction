import os
import pandas as pd


def build_personal_baseline(
    data,
    subject_id,
    subject_column="subject_id",
    condition_column="condition",
    baseline_condition="rest"
):
    """
    Create a personal EEG baseline for one subject.

    The baseline is calculated from the average
    EEG feature values during the resting condition.
    """

    # Select data belonging to the required subject
    subject_data = data[
        data[subject_column].astype(str) == str(subject_id)
    ]

    if subject_data.empty:
        raise ValueError(
            f"No data found for subject {subject_id}"
        )

    # Select resting-state recordings
    baseline_data = subject_data[
        subject_data[condition_column]
        .astype(str)
        .str.lower()
        == baseline_condition.lower()
    ]

    if baseline_data.empty:
        raise ValueError(
            f"No '{baseline_condition}' data found "
            f"for subject {subject_id}"
        )

    # Columns that are identifiers or labels
    excluded_columns = {
        subject_column,
        condition_column,
        "session_id",
        "window_id",
        "label"
    }

    # Find numerical EEG feature columns
    feature_columns = [
        column
        for column in baseline_data.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(
            baseline_data[column]
        )
    ]

    if not feature_columns:
        raise ValueError(
            "No numerical EEG feature columns found."
        )

    # Calculate the average value of each feature
    baseline = baseline_data[
        feature_columns
    ].mean()

    return baseline


def save_baseline(
    baseline,
    subject_id,
    output_directory="models/baselines"
):
    """
    Save the personal baseline as a CSV file.
    """

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    file_path = os.path.join(
        output_directory,
        f"baseline_{subject_id}.csv"
    )

    baseline.to_csv(
        file_path,
        header=["baseline_value"]
    )

    return file_path


def load_baseline(
    subject_id,
    input_directory="models/baselines"
):
    """
    Load a previously saved personal baseline.
    """

    file_path = os.path.join(
        input_directory,
        f"baseline_{subject_id}.csv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Baseline file not found: {file_path}"
        )

    baseline = pd.read_csv(
        file_path,
        index_col=0
    ).squeeze("columns")

    return baseline


if __name__ == "__main__":

    print("Personal baseline module loaded successfully.")