import numpy as np
import pandas as pd


class ResultProcessor:
    def __init__(self, model_name, satellite_name, out_path, timestamps,
                 raw_data, predictions, process_type, kernel=None):

        self.model_name = model_name
        self.satellite_name = satellite_name
        self.out_path = out_path

        self.raw_data = raw_data
        self.predictions = predictions
        self.timestamps = timestamps
        self.residuals = np.abs(self.raw_data - self.predictions.values)

        self.process_type = process_type
        self.kernel = kernel

    def local_normalization(self, kernel=None):
        norm_residuals = self.residuals.copy()
        kernel = np.array(kernel)
        k_len = len(kernel)
        # half_k = k_len // 2

        for i in range(1, len(norm_residuals)):
            start_index = max(0, i - k_len)
            end_index = i
            # start_index = max(0, i - half_k)
            # end_index = min(len(norm_residuals), i + half_k + 1)
            local_window = self.residuals.copy()[start_index:end_index].values

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

    def calculate_diff_between_original_and_normed(self, kernel=None):
        norm_residuals = self.local_normalization(kernel)
        original = self.residuals.values
        normed = norm_residuals.values

        diff = original - normed
        return diff

    def process_residuals(self):
        if self.process_type == 'naive':
            return self.timestamps, self.residuals
        elif self.process_type == 'local_diff':
            proc_residuals = pd.Series(data=self.calculate_diff_between_original_and_normed(self.kernel))
            return self.timestamps, proc_residuals