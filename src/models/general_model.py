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


def prepare_data(
    data,
    target_column="condition",
    subject_column="subject_id"
):
    """
    Separate EEG features from the target label.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    if data.empty:
        raise ValueError("The modeling DataFrame is empty.")

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
            "No numerical EEG feature columns were found."
        )

    X = data[feature_columns].copy()
    y = data[target_column].copy()

    if y.isna().any():
        raise ValueError("The target column contains missing values.")

    if y.nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")

    if X.isna().any().any():
        raise ValueError("Feature columns contain missing values.")

    return X, y, feature_columns


def _split_data(data, X, y, test_size=0.20, random_state=42):
    """Split by subject when subject IDs are available to prevent leakage."""

    if "subject_id" not in data.columns:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    subject_ids = data["subject_id"].astype(str)
    unique_subjects = subject_ids.nunique()

    if unique_subjects < 2:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    train_subjects, test_subjects = train_test_split(
        subject_ids.unique(),
        test_size=test_size,
        random_state=random_state
    )
    train_mask = subject_ids.isin(train_subjects)
    test_mask = subject_ids.isin(test_subjects)

    if y[train_mask].nunique() < 2 or y[test_mask].nunique() < 2:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    return (
        X.loc[train_mask],
        X.loc[test_mask],
        y.loc[train_mask],
        y.loc[test_mask]
    )


def create_models():
    """
    Create the three ML models required for the project.
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


def train_general_models(
    data,
    target_column="condition"
):
    """
    Train and evaluate the general population-level models.
    """

    X, y, feature_columns = prepare_data(
        data=data,
        target_column=target_column
    )

    # Split the dataset into training and testing data
    X_train, X_test, y_train, y_test = _split_data(
        data,
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    models = create_models()

    results = {}

    for model_name, model in models.items():

        print("\n" + "=" * 50)
        print(f"Training: {model_name}")
        print("=" * 50)

        # Train the model
        model.fit(
            X_train,
            y_train
        )

        # Predict test data
        predictions = model.predict(
            X_test
        )

        # Calculate evaluation metrics
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

    # Select the model with the highest F1 score
    best_model_name = max(
        results,
        key=lambda name:
        results[name]["f1_score"]
    )

    best_model = results[
        best_model_name
    ]["model"]

    print("\n" + "=" * 50)
    print(
        f"BEST GENERAL MODEL: {best_model_name}"
    )
    print("=" * 50)

    return (
        best_model,
        best_model_name,
        feature_columns,
        X_test,
        y_test,
        results
    )


def save_general_model(
    model,
    model_name,
    feature_columns,
    output_path="models/general_model.pkl"
):
    """
    Save the trained general model.
    """

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    model_package = {
        "model": model,
        "model_name": model_name,
        "feature_columns": feature_columns
    }

    joblib.dump(
        model_package,
        output_path
    )

    print(
        f"\nGeneral model saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":

    print(
        "General model module loaded successfully."
    )