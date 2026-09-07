import os
import joblib
import pandas as pd


def load_model(model_path):
    """
    Load a saved machine-learning model.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    package = joblib.load(
        model_path
    )

    if not isinstance(package, dict) or "model" not in package:
        raise ValueError("Saved model does not contain a valid model package.")

    return package


def predict_general(
    data,
    model_path="models/general_model.pkl"
):
    """
    Predict cognitive state using the
    general population-level model.
    """

    package = load_model(
        model_path
    )

    model = package["model"]

    feature_columns = package[
        "feature_columns"
    ]

    # Check that required features exist
    missing_features = [
        feature
        for feature in feature_columns
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing features for general model: "
            + ", ".join(missing_features)
        )

    X = data[
        feature_columns
    ]

    prediction = model.predict(
        X
    )

    return prediction


def predict_personalized(
    data,
    model_path="models/personalized_model.pkl"
):
    """
    Predict cognitive state using the
    personalized model.

    The new EEG features are converted into
    baseline-relative features.
    """

    package = load_model(
        model_path
    )

    model = package["model"]

    feature_columns = package[
        "feature_columns"
    ]

    if "baseline" not in package:
        raise ValueError("Saved personalized model has no baseline.")

    baseline = pd.Series(package["baseline"], dtype="float64")

    # Check that required features exist
    missing_features = [
        feature
        for feature in feature_columns
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing features for personalized model: "
            + ", ".join(missing_features)
        )

    personalized_data = pd.DataFrame(
        index=data.index
    )

    # Calculate difference from personal baseline
    for feature in feature_columns:

        if feature not in baseline.index:
            raise ValueError(
                f"Saved personalized model is missing baseline '{feature}'."
            )

        personalized_data[
            f"{feature}_delta"
        ] = (
            data[feature]
            - baseline[feature]
        )

    prediction = model.predict(
        personalized_data
    )

    return prediction


def predict_both(
    data,
    general_model_path="models/general_model.pkl",
    personalized_model_path="models/personalized_model.pkl"
):
    """
    Generate predictions from both models.
    """

    general_prediction = predict_general(
        data,
        general_model_path
    )

    personalized_prediction = predict_personalized(
        data,
        personalized_model_path
    )

    results = pd.DataFrame({

        "general_prediction":
            general_prediction,

        "personalized_prediction":
            personalized_prediction

    })

    return results


def save_predictions(
    predictions,
    output_path="results/reports/predictions.csv"
):
    """
    Save predictions to a CSV file.
    """

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    predictions.to_csv(
        output_path,
        index=False
    )

    print(
        f"Predictions saved to: {output_path}"
    )


if __name__ == "__main__":

    print(
        "Prediction module loaded successfully."
    )