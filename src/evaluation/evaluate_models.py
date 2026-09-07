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


def create_confusion_matrix(
    y_true,
    y_pred,
    title="Confusion Matrix",
    output_path="results/figures/confusion_matrix.png"
):
    """
    Create and save a confusion matrix.
    """

    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.metrics import confusion_matrix

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    plt.figure(
        figsize=(7, 6)
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel(
        "Predicted Cognitive State"
    )

    plt.ylabel(
        "Actual Cognitive State"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Confusion matrix saved to: {output_path}"
    )


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