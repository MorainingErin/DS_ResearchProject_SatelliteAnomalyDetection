import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, timestamps, residuals, ground_truth_manoeuvers, tolerance_window=5):
        """
        Initialize the Evaluator class.

        :param residuals: Array of residuals from the model.
        :param timestamps: Array of timestamps corresponding to the residuals.
        :param ground_truth_maneuvers: Array of timestamps for ground truth maneuvers.
        :param tolerance_window: Tolerance window (in the same units as timestamps) for comparing anomalies with maneuvers.
        """
        self.timestamps = timestamps
        self.residuals = residuals
        self.ground_truth_manoeuvers = ground_truth_manoeuvers
        self.tolerance_window = tolerance_window

    def identify_anomalies(self, threshold):
        """
        Identify anomalies based on a threshold.

        :param threshold: Threshold for residuals to classify anomalies.
        :return: Array of timestamps where anomalies are detected.
        """
        anomaly_indices = np.where(self.residuals > threshold)[0]
        return self.timestamps[anomaly_indices]

    def compare_with_maneuvers(self, predicted_anomalies):
        """
        Compare predicted anomalies with ground truth maneuvers using a tolerance window.

        :param predicted_anomalies: Array of timestamps where anomalies are detected.
        :return: TP, FP, TN, FN counts.
        """
        tp, fp, fn = 0, 0, 0
        matched_manoeuvers = set()

        # Check for true positives and false positives
        for anomaly in predicted_anomalies:
            if any(abs((anomaly - manoeuver).total_seconds()/3600/24) <= self.tolerance_window for manoeuver in self.ground_truth_manoeuvers):
                tp += 1
                matched_manoeuvers.add(next(manoeuver for manoeuver in self.ground_truth_manoeuvers if abs((anomaly - manoeuver).total_seconds()/3600/24) <= self.tolerance_window))
            else:
                fp += 1

        # Check for false negatives
        for maneuver in self.ground_truth_manoeuvers:
            if maneuver not in matched_manoeuvers:
                fn += 1

        # True negatives are not well-defined in this context, so we return None for TN
        return tp, fp, None, fn

    def calculate_precision_recall(self, tp, fp, fn):
        """
        Calculate precision and recall.

        :param tp: True positives.
        :param fp: False positives.
        :param fn: False negatives.
        :return: Precision and recall values.
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return precision, recall

    def plot_precision_recall_curve(self):
        """
        Plot the precision-recall curve for different thresholds.
        """
        precisions, recalls, thresholds = precision_recall_curve(
            [1 if any(abs((ts - manoeuver).total_seconds()/3600/24) <= self.tolerance_window for manoeuver in self.ground_truth_manoeuvers) else 0 for ts in self.timestamps],
            self.residuals
        )
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker='o', label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid()
        plt.show()

    def evaluate(self, thresholds):
        """
        Evaluate the model for a list of thresholds.

        :param thresholds: List of thresholds to evaluate.
        :return: List of precision and recall values for each threshold.
        """
        results = []
        for threshold in thresholds:
            predicted_anomalies = self.identify_anomalies(threshold)
            tp, fp, _, fn = self.compare_with_maneuvers(predicted_anomalies)
            precision, recall = self.calculate_precision_recall(tp, fp, fn)
            results.append((threshold, precision, recall))
        return results