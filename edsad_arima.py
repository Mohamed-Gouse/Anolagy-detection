"""
Data stream anolagy detection using ARIMA Model
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Parameters for the data stream
np.random.seed(42)
seasonal_period = 50
noise_level = 0.5
anomaly_chance = 0.05
anomaly_magnitude = 5

# Generate a single data point
def generate_data_point(t):
    try:
        # Seasonal component (sinusoidal pattern)
        seasonal = np.sin(2 * np.pi * t / seasonal_period)
        # Noise component
        noise = np.random.normal(0, noise_level)
        # Anomaly component
        if np.random.rand() < anomaly_chance:
            anomaly = np.random.normal(0, anomaly_magnitude)
        else:
            anomaly = 0
        return seasonal + noise + anomaly
    except Exception as e:
        print(f"Error generating data point at time {t}: {e}")
        return 0

# Simulate a data stream
def simulate_data_stream(num_point=200):
    data_stream = []
    for t in range(num_point):
        try:
            data_point = generate_data_point(t)
            data_stream.append(data_point)
        except Exception as e:
            print(f"Error simulating data stream at time {t}: {e}")
    return data_stream

# Define parameters for anomaly detection
window_size = 50
threshold = 2.0

# Function to detect anomalies
def detect_anomalies(data_stream):
    anomalies = []
    residuals = []
    for i in range(window_size, len(data_stream)):
        try:
            # Train ARIMA model with last 'window_size' points
            train_data = data_stream[i - window_size: i]
            model = ARIMA(train_data, order=(1, 1, 1))
            model_fit = model.fit()

            # Forecast the next data point
            forecast = model_fit.forecast()[0]
            actual = data_stream[i]

            # Calculate residual and check if it exceeds the threshold
            residual = np.abs(actual - forecast)
            residuals.append(residual)

            if residual > threshold:
                anomalies.append((i, actual))
            else:
                anomalies.append((i, None))
        except Exception as e:
            print(f"Error detecting anomalies at time {i}: {e}")
            # Add None for anomalies and residuals that couldn't be detected
            anomalies.append((i, None))
            residuals.append(None)

    return anomalies, residuals

# Generate and analyze data
try:
    data = simulate_data_stream()
    anomalies, residuals = detect_anomalies(data)
    
    if len(data) == 0:
        raise ValueError("No data points were generated. Please check the data generation function.")

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(data, color='b', label='Data Stream')
    plt.scatter(*zip(*[(i, val) for i, val in anomalies if val is not None]), color='r', label='Anomalies')
    plt.title('Data Stream with Detected Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

except ValueError as ve:
    print(f"ValueError: {ve}")
except Exception as e:
    print(f"Unexpected error: {e}")
except KeyboardInterrupt:
    print("Anomaly detection stopped.")