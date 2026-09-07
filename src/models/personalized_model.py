import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def get_feature_columns(
    data,
    target_column="condition",
    subject_column="subject_id"
):
    """
    Identify numerical EEG feature columns.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    excluded_columns = {
        target_column,
        subject_column,
        "session_id",
        "window_id",
        "label"
    }

    feature_columns = [
        column
        for column in data.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(
            data[column]
        )
    ]

    if not feature_columns:
        raise ValueError(
            "No numerical EEG features found."
        )

    return feature_columns


def create_personalized_features(
    data,
    baseline,
    feature_columns
):
    """
    Convert normal EEG features into
    baseline-relative personalized features.

    Personalized feature =
    Current EEG feature - Personal baseline
    """

    baseline = pd.Series(baseline, dtype="float64")
    personalized_data = pd.DataFrame(
        index=data.index
    )

    for feature in feature_columns:

        if feature not in baseline.index:
            raise ValueError(
                f"Feature '{feature}' is missing "
                f"from the personal baseline."
            )

        personalized_data[
            f"{feature}_delta"
        ] = (
            data[feature]
            - baseline[feature]
        )

    if personalized_data.isna().any().any():
        raise ValueError("Personalized features contain missing values.")

    return personalized_data


def create_models():
    """
    Create the three ML models.
    """

    models = {

        "Logistic Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000
                )
            )
        ]),

        "SVM": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf"
                )
            )
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )
    }

    return models


def train_personalized_models(
    data,
    baseline,
    target_column="condition"
):
    """
    Train personalized ML models using
    the individual's EEG baseline.
    """

    # Get original EEG feature columns
    feature_columns = get_feature_columns(
        data,
        target_column=target_column
    )

    # Convert features into baseline-relative features
    X = create_personalized_features(
        data,
        baseline,
        feature_columns
    )

    if data.empty:
        raise ValueError("The modeling DataFrame is empty.")

    if data[target_column].isna().any() or data[target_column].nunique() < 2:
        raise ValueError(
            "The target must be present, non-null, and contain at least two classes."
        )

    # Cognitive-state labels
    y = data[
        target_column
    ]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = create_models()

    results = {}

    for model_name, model in models.items():

        print("\n" + "=" * 55)
        print(
            f"Training Personalized {model_name}"
        )
        print("=" * 55)

        # Train
        model.fit(
            X_train,
            y_train
        )

        # Predict
        predictions = model.predict(
            X_test
        )

        # Metrics
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        results[model_name] = {
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        print(
            f"Accuracy  : {accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1 Score  : {f1:.4f}"
        )

    # Select best model using F1-score
    best_model_name = max(
        results,
        key=lambda name:
        results[name]["f1_score"]
    )

    best_model = results[
        best_model_name
    ]["model"]

    print("\n" + "=" * 55)
    print(
        f"BEST PERSONALIZED MODEL: "
        f"{best_model_name}"
    )
    print("=" * 55)

    return (
        best_model,
        best_model_name,
        feature_columns,
        X_test,
        y_test,
        results
    )


def save_personalized_model(
    model,
    model_name,
    feature_columns,
    baseline,
    output_path="models/personalized_model.pkl"
):
    """
    Save the personalized model together
    with its feature list and personal baseline.
    """

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    baseline_series = pd.Series(baseline, dtype="float64")
    model_package = {
        "model": model,
        "model_name": model_name,
        "feature_columns": feature_columns,
        "baseline": baseline_series.to_dict()
    }

    joblib.dump(
        model_package,
        output_path
    )

    print(
        "\nPersonalized model saved to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":

    print(
        "Personalized model module loaded successfully."
    )