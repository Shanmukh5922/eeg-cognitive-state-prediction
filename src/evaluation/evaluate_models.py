print("Evaluation module loaded successfully.")


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for a classification model.
    """

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def _get_confusion_labels(y_true, y_pred):
    """Use the canonical cognitive-state ordering across all confusion matrices."""

    preferred_labels = ["rest", "easy", "medium", "diff"]
    observed = {str(value) for value in set(y_true) | set(y_pred)}
    labels = [label for label in preferred_labels if label in observed]
    if not labels:
        labels = sorted(observed)
    return labels


def create_confusion_matrix(
    y_true,
    y_pred,
    title="Confusion Matrix",
    output_path="results/figures/confusion_matrix.png",
    labels=None,
    normalize=False,
):
    """
    Create and save a confusion matrix using the held-out test-set predictions.
    """

    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    from sklearn.metrics import confusion_matrix

    y_true = [str(value) for value in y_true]
    y_pred = [str(value) for value in y_pred]
    labels = labels if labels is not None else _get_confusion_labels(y_true, y_pred)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix = matrix.astype(float) / row_sums
        fmt = ".2f"
        value_label = "Row-normalized fraction"
        cmap = "Blues"
    else:
        fmt = "d"
        value_label = "Count"
        cmap = "Blues"

    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        cbar_kws={"label": value_label},
        vmin=0,
    )

    plt.xlabel("Predicted Cognitive State")
    plt.ylabel("Actual Cognitive State")
    plt.title(title)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")


def compare_models(
    general_results,
    personalized_results
):
    """
    Compare the general and personalized models.
    """

    import pandas as pd

    comparison = pd.DataFrame(
        {
            "General Model": [
                general_results["accuracy"],
                general_results["precision"],
                general_results["recall"],
                general_results["f1_score"]
            ],

            "Personalized Model": [
                personalized_results["accuracy"],
                personalized_results["precision"],
                personalized_results["recall"],
                personalized_results["f1_score"]
            ]
        },

        index=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    )

    comparison["Improvement"] = (
        comparison["Personalized Model"]
        - comparison["General Model"]
    )

    return comparison


def save_comparison_report(
    comparison,
    output_path="results/reports/model_comparison.csv"
):
    """
    Save general vs personalized comparison.
    """

    import os

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    comparison.to_csv(
        output_path
    )

    print(
        f"Comparison report saved to: {output_path}"
    )


def print_comparison(
    comparison
):
    """
    Display the comparison in the terminal.
    """

    print("\n")
    print("=" * 70)
    print("GENERAL VS PERSONALIZED MODEL")
    print("=" * 70)

    print(
        comparison
    )

    print("=" * 70)


if __name__ == "__main__":

    print(
        "Evaluation file executed successfully."
    )