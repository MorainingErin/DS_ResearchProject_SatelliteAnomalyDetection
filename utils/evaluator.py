import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Evaluator:
    def __init__(self, model_name, satellite_name, out_path, process_type,
                 timestamp, residuals, ground_truth_manoeuvers, matching_max_days=1):

        self.model_name = model_name
        self.satellite_name = satellite_name
        self.out_path = out_path
        self.process_type = process_type

        self.residuals = pd.Series(data=residuals, index=pd.to_datetime(timestamp))
        self.ground_truth_manoeuvers = pd.to_datetime(ground_truth_manoeuvers['end_timestamp'])
        self.matching_max_days = matching_max_days
        self.dict_predictions_to_ground_truth = {}
        self.dict_ground_truth_to_predictions = {}

        self.precision_recall_thresholds = []

    def compute_anomalies_using_global_threshold(self, threshold):
        anomalies = (self.residuals.values >= threshold).astype(int)
        return anomalies.tolist()

    def convert_timestamp_series_to_epoch(self, series):
        return (
            (series - pd.Timestamp(year=1970, month=1, day=1)) // pd.Timedelta(seconds=1)
        ).values

    def match_event_for_one_threshold(self, anomalies):
        self.dict_predictions_to_ground_truth = {}
        self.dict_ground_truth_to_predictions = {}
        
        matching_max_distance_seconds = pd.Timedelta(days=self.matching_max_days).total_seconds()

        manoeuvre_timestamps_seconds = self.convert_timestamp_series_to_epoch(self.ground_truth_manoeuvers)
        pred_time_stamps_seconds = self.convert_timestamp_series_to_epoch(self.residuals.index)

        for i in range(len(anomalies)):
            if anomalies[i] == 1:  # If the prediction is an anomaly
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

    def compute_fp_fn_indices(self, anomalies):
        positive_prediction_indices = np.argwhere(np.array(anomalies) == 1)[:, 0]
        list_false_positives = [
            pred_ind for pred_ind in positive_prediction_indices if pred_ind not in self.dict_predictions_to_ground_truth.keys()
        ]
        list_false_negatives = [
            true_ind for true_ind in np.arange(0, len(self.ground_truth_manoeuvers))
            if true_ind not in self.dict_ground_truth_to_predictions.keys()
        ]
        return list_false_positives, list_false_negatives

    def calculate_precision_recall(self, list_false_positives, list_false_negatives):
        precision = len(self.dict_ground_truth_to_predictions) / (len(self.dict_ground_truth_to_predictions) + len(list_false_positives))
        recall = len(self.dict_ground_truth_to_predictions) / (len(self.dict_ground_truth_to_predictions) + len(list_false_negatives))
        return precision, recall
        # if precision >= 0.75 and recall >= 0.629:
        # # To export the intermediate results for debugging
        #     labels = []
        #     for i in range(len(predictions)):
        #         if i in list_false_positives:
        #             labels.append("fp")
        #         elif i in positive_prediction_indices:
        #             labels.append("tp")
        #         else:
        #             labels.append("n")

        #     df_predictions = pd.DataFrame({
        #         "residual": self.residuals.values,
        #         "label": labels
        #     }, index=self.residuals.index)
        #     df_predictions.to_csv(self.out_path / "prediction_results.csv")
        #     print(f"Exported prediction results to {self.out_path / 'prediction_results.csv'}")

        #     # To export manoeuvre timestamps for debugging
        #     manoeuvre_labels = []
        #     for idx in range(len(self.ground_truth_manoeuvers)):
        #         if idx in self.dict_ground_truth_to_predictions:
        #             manoeuvre_labels.append("tp")
        #         else:
        #             manoeuvre_labels.append("fn")

        #     df_manoeuvres = pd.DataFrame({
        #         "manoeuvre_timestamp": self.ground_truth_manoeuvers,
        #         "label": manoeuvre_labels
        #     })
        #     df_manoeuvres.to_csv(self.out_path / "manoeuvre_results.csv", index=False)
        #     print(f"Exported manoeuvre results to {self.out_path / 'manoeuvre_results.csv'}")

        return precision, recall
    
    def compute_all_precision_recall(self):
        thresholds = np.sort(self.residuals.to_numpy())[:-10]

        for threshold in thresholds:
            anomalies = self.compute_anomalies_using_global_threshold(threshold)
            self.match_event_for_one_threshold(anomalies)
            list_false_positives, list_false_negatives = self.compute_fp_fn_indices(anomalies)
            precision, recall = self.calculate_precision_recall(list_false_positives, list_false_negatives)
            self.precision_recall_thresholds.append((threshold, precision, recall))
        return self.precision_recall_thresholds

    def plot_precision_recall_curve(self):

        precisions = [item[1] for item in self.precision_recall_thresholds]
        recalls = [item[2] for item in self.precision_recall_thresholds]

        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker='o', label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve -- {self.process_type} Thresholds")
        plt.legend()
        plt.grid()
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)

        plt.savefig(self.out_path / f'{self.model_name}_{self.process_type}_Precision_Recall_Curve.png')
        plt.close()