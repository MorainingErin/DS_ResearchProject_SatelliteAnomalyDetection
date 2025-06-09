import numpy as np
import pandas as pd
# from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


class Evaluator:
    def __init__(self, model_name, satellite_name, out_path,
                 timestamp, residuals, ground_truth_manoeuvers, matching_max_days=1):

        self.model_name = model_name
        self.satellite_name = satellite_name
        self.out_path = out_path

        self.residuals = pd.Series(data=residuals, index=pd.to_datetime(timestamp))
        self.ground_truth_manoeuvers = pd.to_datetime(ground_truth_manoeuvers['end_timestamp'])
        self.matching_max_days = matching_max_days
        self.dict_predictions_to_ground_truth = {}
        self.dict_ground_truth_to_predictions = {}

        self.precision_recall_thresholds = []

    def convert_timestamp_series_to_epoch(self, series):
        return (
            (series - pd.Timestamp(year=1970, month=1, day=1)) // pd.Timedelta(seconds=1)
        ).values

    def compute_simple_matching_precision_recall_for_one_threshold(self, threshold):
        self.dict_predictions_to_ground_truth = {}
        self.dict_ground_truth_to_predictions = {}
        
        matching_max_distance_seconds = pd.Timedelta(days=self.matching_max_days).total_seconds()

        manoeuvre_timestamps_seconds = self.convert_timestamp_series_to_epoch(self.ground_truth_manoeuvers)
        pred_time_stamps_seconds = self.convert_timestamp_series_to_epoch(self.residuals.index)
        predictions = self.residuals.to_numpy()

        for i in range(predictions.shape[0]):
            if predictions[i] >= threshold:
                left_index = np.searchsorted(
                    manoeuvre_timestamps_seconds, pred_time_stamps_seconds[i]
                )

                if left_index != 0:
                    left_index -= 1

                index_of_closest = left_index

                if (left_index < self.ground_truth_manoeuvers.shape[0] - 1) and (
                    abs(manoeuvre_timestamps_seconds[left_index] - pred_time_stamps_seconds[i])
                    > abs(manoeuvre_timestamps_seconds[left_index + 1] - pred_time_stamps_seconds[i])
                ):
                    index_of_closest = left_index + 1

                diff = abs(manoeuvre_timestamps_seconds[index_of_closest] - pred_time_stamps_seconds[i])

                if diff < matching_max_distance_seconds:
                    self.dict_predictions_to_ground_truth[i] = (
                        index_of_closest,
                        diff,
                    )
                    if index_of_closest in self.dict_ground_truth_to_predictions:
                        self.dict_ground_truth_to_predictions[index_of_closest].append(i)
                    else:
                        self.dict_ground_truth_to_predictions[index_of_closest] = [i]

        positive_prediction_indices = np.argwhere(predictions >= threshold)[:, 0]
        list_false_positives = [
            pred_ind for pred_ind in positive_prediction_indices if pred_ind not in self.dict_predictions_to_ground_truth.keys()
        ]
        list_false_negatives = [
            true_ind for true_ind in np.arange(0, len(self.ground_truth_manoeuvers))
            if true_ind not in self.dict_ground_truth_to_predictions.keys()
        ]

        precision = len(self.dict_ground_truth_to_predictions) / (len(self.dict_ground_truth_to_predictions) + len(list_false_positives))
        recall = len(self.dict_ground_truth_to_predictions) / (len(self.dict_ground_truth_to_predictions) + len(list_false_negatives))

        return precision, recall

    def compute_all_precision_recall(self):
        thresholds = np.sort(self.residuals.to_numpy())[:-10]

        for threshold in thresholds:
            precision, recall = self.compute_simple_matching_precision_recall_for_one_threshold(threshold)
            self.precision_recall_thresholds.append((threshold, precision, recall))

        return self.precision_recall_thresholds

    def plot_precision_recall_curve(self):
        """
        Plot the precision-recall curve for different thresholds.
        """
        precisions = [item[1] for item in self.precision_recall_thresholds]
        recalls = [item[2] for item in self.precision_recall_thresholds]
        # precisions, recalls, thresholds = precision_recall_curve(
        #     [1 if any(abs((ts - manoeuver).total_seconds()/3600/24) <= self.tolerance_window for manoeuver in self.ground_truth_manoeuvers) else 0 for ts in self.timestamps],
        #     self.residuals
        # )
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker='o', label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid()
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)

        plt.savefig(self.out_path / f'{self.model_name}_Precision_Recall_Curve.png')
        plt.close()