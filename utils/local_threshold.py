import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LocalThresholdEvaluator:
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

    def local_normalization(self, local_days=3, kernel=None):
        norm_residuals = self.residuals.copy()
        kernel = np.array(kernel)
        k_len = len(kernel)
        half_k = k_len // 2

        for i in range(1, len(norm_residuals)):
            start_index = max(0, i - k_len)
            end_index = i
            # start_index = max(0, i - half_k)
            # end_index = min(len(norm_residuals), i + half_k + 1)
            local_window = norm_residuals[start_index:end_index].values

            # Adjust kernel if window is at the edge
            if len(local_window) != k_len:
                # k_adj = kernel[half_k - (i - start_index): half_k + (end_index - i)]
                k_adj = kernel[-len(local_window):]
                k_adj = k_adj / k_adj.sum()  # Normalize
            else:
                k_adj = kernel

            norm_residuals.iloc[i] = np.sum(local_window * k_adj)

        # pd.DataFrame(norm_residuals, columns=['norm_residuals']).to_csv(self.out_path / f'{self.model_name}_norm_residuals.csv', index=False)

        return norm_residuals
    
    def calculate_ratio_between_original_and_normed(self, local_days, kernel=None):
        norm_residuals = self.local_normalization(local_days, kernel)
        original = self.residuals.values
        normed = norm_residuals.values

        ratios = original / normed
        return ratios

    def filter_anomalies_for_one_threshold(self, ratio, threshold):
        anomalies = (ratio > threshold).astype(int)
        return anomalies

    def convert_timestamp_series_to_epoch(self, series):
        return (
            (series - pd.Timestamp(year=1970, month=1, day=1)) // pd.Timedelta(seconds=1)
        ).values

    def compute_simple_matching_precision_recall_for_one_threshold(self, ratios, threshold):
        self.dict_predictions_to_ground_truth = {}
        self.dict_ground_truth_to_predictions = {}
        
        matching_max_distance_seconds = pd.Timedelta(days=self.matching_max_days).total_seconds()

        manoeuvre_timestamps_seconds = self.convert_timestamp_series_to_epoch(self.ground_truth_manoeuvers)
        pred_time_stamps_seconds = self.convert_timestamp_series_to_epoch(self.residuals.index)
        predictions = self.filter_anomalies_for_one_threshold(ratios, threshold)

        for i in range(predictions.shape[0]):
            if predictions[i] == 1:
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

        positive_prediction_indices = np.argwhere(predictions == 1)[:, 0]
        list_false_positives = [
            pred_ind for pred_ind in positive_prediction_indices if pred_ind not in self.dict_predictions_to_ground_truth.keys()
        ]
        list_false_negatives = [
            true_ind for true_ind in np.arange(0, len(self.ground_truth_manoeuvers))
            if true_ind not in self.dict_ground_truth_to_predictions.keys()
        ]

        # if threshold > 4.65 and threshold < 5.0:
        #     pd.DataFrame(positive_prediction_indices, columns=["index"]).to_csv(self.out_path / "positive_prediction_indices.csv", index=False)
        #     pd.DataFrame(list_false_positives, columns=["index"]).to_csv(self.out_path / "list_false_positives.csv", index=False)
        #     pd.DataFrame(list_false_negatives, columns=["index"]).to_csv(self.out_path / "list_false_negatives.csv", index=False)
        #     pd.DataFrame(self.residuals, columns=["residuals"]).to_csv(self.out_path / "residuals.csv", index=False)

        precision = len(self.dict_ground_truth_to_predictions) / (len(self.dict_ground_truth_to_predictions) + len(list_false_positives))
        recall = len(self.dict_ground_truth_to_predictions) / (len(self.dict_ground_truth_to_predictions) + len(list_false_negatives))

        return precision, recall
    
    def compute_all_precision_recall(self):

        local_days = 10
        # length = 2 * local_days + 1
        kernel = [1 / local_days for i in range(local_days)]
        # kernel=[0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]
        # kernel = [1/6, 1/6, 1/6, 0, 1/6, 1/6, 1/6]
        ratios = self.calculate_ratio_between_original_and_normed(local_days=local_days, kernel=kernel)
        thresholds = np.sort(ratios)[:-10]
        # thresholds = [4.684]
        # thresholds = np.linspace(0.001, 2, num=100)

        for threshold in thresholds:
            # pd.DataFrame(ratios, columns=["ratios"]).to_csv(self.out_path / f"ratios_{threshold}.csv", index=False)
            precision, recall = self.compute_simple_matching_precision_recall_for_one_threshold(ratios, threshold)
            # print(f"Threshold: {threshold}, Precision: {precision}, Recall: {recall}")
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