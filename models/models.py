import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib


class ModelRunner():
    def __init__(self, model_folder, verbose):
        self.model_folder = model_folder
        self.verbose = verbose
        pass
    
    def set_model_name(self):
        # To be implemented by child classes
        raise NotImplementedError("create_model_name method should be implemented in child classes")

    def load_model(self):
        # To be implemented by child classes
        raise NotImplementedError("load_model method should be implemented in child classes")

    def save_model(self):
        # To be implemented by child classes
        raise NotImplementedError("save_model method should be implemented in child classes")

    def preprocess_data(self):
        # To be implemented by child classes
        raise NotImplementedError("preprocess_data method should be implemented in child classes")

    def train(self):
        # To be implemented by child classes
        raise NotImplementedError("train method should be implemented in child classes")

    def predict(self):
        # To be implemented by child classes
        raise NotImplementedError("predict method should be implemented in child classes")


class ARIMARunner(ModelRunner):
    def __init__(self, model_folder, verbose, order=(5, 1, 0)):
        super().__init__(model_folder, verbose)
        self.model_path = None
        self.model = None
        self.order = order

    def __repr__(self):
        return "\n".join([
            f"Model name: ARIMA",
            f"Saved path: {self.model_path}",
            "Params:",
            f"  order: {self.order}"
        ])

    def param_str(self):
        return "_".join(map(str, self.order))

    def set_model_name(self, model_name=None):
        """None for default"""
        if model_name is None:
            order_str = self.param_str()
            self.model_path = self.model_folder / f"arima-ord{order_str}.pkl"
        else:
            self.model_path = self.model_folder / model_name
        return self.model_path

    def load_model(self):
        if self.model_path.exists():
            loaded_data = joblib.load(self.model_path)
            self.model = loaded_data["model"]
            self.order = loaded_data["order"]
        else:
            pass

    def save_model(self, to_path=None):
        """Save the model to the specified path. If no path is provided, use the default model path."""
        if to_path is None:
            to_path = self.model_path
        if self.model is None:
            raise ValueError("Model is not trained yet. Cannot save.")
        joblib.dump({
            "model": self.model,
            "order": self.order
        }, to_path)

    def preprocess_data(self, data):
        y = data.dropna()
        y_scaled = (y - y.mean()) / y.std()
        return y_scaled

    def train(self, train_data):
        """Return: score, predicted values"""
        y_proc = self.preprocess_data(train_data)
        self.model = ARIMA(y_proc, order=self.order).fit(
            method_kwargs={"disp": 1 if self.verbose else 0}
        )

        fitted_values = self.model.fittedvalues
        # Reverse the scaling
        fitted_values = fitted_values * train_data.std() + train_data.mean()
        return self.model.aic, fitted_values  # Return AIC for model selection

    def predict(self, test_data):
        """Return: score, predicted values"""
        return self.train(test_data)


class XGBoostRunner(ModelRunner):
    def __init__(self, model_folder, verbose, window=10, n_estimators=100, max_depth=3, learning_rate=0.1):
        # window (int): Rolling window size for feature generation.
        # model_params (dict): Additional parameters for the XGBoost model.

        super().__init__(model_folder, verbose)
        self.model_path = None
        self.model = None
        self.window = window
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
    
    def __repr__(self):
        return "\n".join([
            f"Model name: XGBoost",
            f"Saved path: {self.model_path}",
            "Params:",
            f"  window: {self.window}",
            f"  n_estimators: {self.n_estimators}",
            f"  max_depth: {self.max_depth}",
            f"  learning_rate: {self.learning_rate}"
        ])

    def param_str(self):
        return f"win{self.window}_nest{self.n_estimators}_maxdep{self.max_depth}_lr{self.learning_rate}"

    def set_model_name(self, model_name=None):
        """None for default"""
        if model_name is None:
            para_str = self.param_str()
            self.model_path = self.model_folder / f"xgboost-{para_str}.pkl"
        else:
            self.model_path = self.model_folder / model_name
        return self.model_path

    def load_model(self):
        if self.model_path.exists():
            self.model = xgb.XGBRegressor()
            loaded_data = joblib.load(self.model_path)
            self.model = loaded_data["model"]
            self.window = loaded_data["params"]["window"]
            self.n_estimators = loaded_data["params"]["n_estimators"]
            self.max_depth = loaded_data["params"]["max_depth"]
            self.learning_rate = loaded_data["params"]["learning_rate"]
        else:
            self.model = None

    def save_model(self, to_path=None):
        if to_path is None:
            to_path = self.model_path
        params = {
            "window": self.window,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate
        }
        joblib.dump({
            "model": self.model,
            "params": params
        }, to_path)

    def difference_data(self, data, difference_order=1):
        """
        Apply differencing to the data to make it stationary.
        
        Parameters:
            data (pd.Series): Series containing the orbital element data
            difference_order (int): The order of differencing to apply
        
        Returns:
            pd.Series: Differenced data
        """
        if difference_order > 0:
            return data.diff(periods=difference_order).dropna()
        return data

    def preprocess_data(self, data):
        """
        Convert a time-series column into a matrix X (features) and vector y (target values)
        using a rolling window approach.
        
        Parameters:
            data (pd.Series): Series containing the orbital element data
        
        Returns:
            X (np.array): Feature matrix of shape (n_samples, window_size)
            y (np.array): Target values of shape (n_samples,)
        """
        data = data.values.reshape(-1, 1)  # Ensure data is 2-dimensional
        data = self.scaler.fit_transform(data)

        # Ensure we have enough data points for the rolling window
        if len(data) <= self.window:
            raise ValueError("Data length should be greater than window size.")
        
        # Initialize feature matrix X and target vector y
        X = np.empty((len(data) - self.window, self.window))
        y = np.empty(len(data) - self.window)

        # Populate X and y with data
        for i in range(len(data) - self.window):
            X[i] = data[i:i + self.window].flatten()  # Append past `window_size` values
            y[i] = data[i + self.window].item()  # Extract scalar value for the target

        return X, y

    def train(self, train_data):

        diff_train_data = self.difference_data(train_data, difference_order=1)
        X_train, y_train = self.preprocess_data(diff_train_data)

        if self.model is None:
            self.model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                verbosity=1 if self.verbose else 0
            )

        # self.model.fit(X_train, y_train, early_stopping_rounds=10, verbose=False)
        self.model.fit(X_train, y_train)
        fitted_values = self.model.predict(X_train)

        fitted_reverse = self.scaler.inverse_transform(fitted_values.reshape(-1, 1))

        # Calculate residuals
        residuals = fitted_reverse - diff_train_data[self.window:].values.reshape(-1, 1)
        mse = np.square(np.mean(residuals))  # Mean Squared Error (MSE)
        
        return mse, fitted_reverse

    def predict(self, val_data):

        diff_val_data = self.difference_data(val_data, difference_order=1)
        X_val, y_val = self.preprocess_data(diff_val_data)
        val_predictions = self.model.predict(X_val)

        # Inverse transform the predictions to get them back to the original scale
        val_reverse = self.scaler.inverse_transform(val_predictions.reshape(-1, 1))

        # Calculate residuals
        residuals = val_reverse - diff_val_data[self.window:].values.reshape(-1, 1)
        mse = np.square(np.mean(residuals))  # Mean Squared Error (MSE)

        return mse, val_reverse


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class LSTMRunner(ModelRunner):
    def __init__(self, model_folder, verbose, window=10, hidden_dim=50, num_layers=2, learning_rate=0.001):
        super().__init__(model_folder, verbose)
        self.window = window
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.scaler = StandardScaler()
        self.model = self.build_model()
        self.criterion = nn.MSELoss()
        self.learning_rate =learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def __repr__(self):
        return "\n".join([
            f"Model name: LSTM",
            f"Saved path: {self.model_path}",
            "Params:",
            f"  window: {self.window}",
            f"  hidden_dim: {self.hidden_dim}",
            f"  num_layers: {self.num_layers}",
            f"  learning_rate: {self.learning_rate}"
        ])

    def param_str(self):
        return f"win{self.window}_hiddim{self.hidden_dim}_nlayer{self.num_layers}_lr{self.learning_rate}"

    def set_model_name(self, model_name=None):
        """None for default"""
        if model_name is None:
            para_str = self.param_str()
            self.model_path = self.model_folder / f"lstm-{para_str}.pkl"
        else:
            self.model_path = self.model_folder / model_name
        return self.model_path

    def load_model(self):
        if self.model_path.exists():
            loaded_data = torch.load(self.model_path)
            self.model = loaded_data["model"]
            self.window = loaded_data["params"]["window"]
            self.hidden_dim = loaded_data["params"]["hidden_dim"]
            self.num_layers = loaded_data["params"]["num_layers"]
            self.learning_rate = loaded_data["params"]["learning_rate"]
            self.model = self.build_model()
        else:
            self.model = None

    def save_model(self, to_path=None):
        if to_path is None:
            to_path = self.model_path
        params = {
            "window": self.window,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "learning_rate": self.learning_rate
        }
        torch.save({
            "model": self.model,
            "params": params
        }, to_path)

    def build_model(self):
        return LSTMModel(input_dim=1, hidden_dim=self.hidden_dim, num_layers=self.num_layers)

    def difference_data(self, data, difference_order=1):
        """
        Apply differencing to the data to make it stationary.
        
        Parameters:
            data (pd.Series): Series containing the orbital element data
            difference_order (int): The order of differencing to apply
        
        Returns:
            pd.Series: Differenced data
        """
        if difference_order > 0:
            return data.diff(periods=difference_order).dropna()
        return data

    def preprocess_data(self, data):
        data = data.values.reshape(-1, 1)
        data = self.scaler.fit_transform(data)
        X = []
        y = []
        for i in range(len(data) - self.window):
            X.append(data[i:i+self.window])
            y.append(data[i+self.window])
        X = np.array(X)
        y = np.array(y)
        return X, y

    def train(self, train_data, epochs=50, batch_size=32):
        diff_train_data = self.difference_data(train_data, difference_order=1)
        X, y = self.preprocess_data(diff_train_data)
        dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                y_pred = self.model(batch_X)
                loss = self.criterion(y_pred, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, val_data):
        diff_val_data = self.difference_data(val_data, difference_order=1)
        X_val, y_val = self.preprocess_data(diff_val_data)
        X_tensor = torch.tensor(X_val).float()
        val_predictions = self.model(X_tensor).detach().numpy()

        # Inverse transform the predictions to get them back to the original scale
        val_reverse = self.scaler.inverse_transform(val_predictions)

        # Calculate residuals
        residuals = val_reverse - diff_val_data[self.window:].values.reshape(-1, 1)
        mse = np.square(np.mean(residuals))  # Mean Squared Error (MSE)

        return mse, val_reverse