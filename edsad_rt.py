"""
Reat-time data stream anolagy detection using Isolation Forest Model
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import time

# Parameters for data stream simulation
time_interval = 0         # Seconds between each data point generation
n_init_data = 100           # Number of initial data points for model training
seasonal_amplitude = 10
daily_pattern_amplitude = 5
noise_level = 2

# Generate a simulated data point based on seasonal and daily patterns with noise
def generate_data_point(t):
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / 50)
    daily_pattern = daily_pattern_amplitude * np.sin(2 * np.pi * t / 10)
    noise = noise_level * np.random.randn()
    return seasonal + daily_pattern + noise

# Initialize Isolation Forest model for anomaly detection
model = IsolationForest(contamination=0.05)

# Initialize timestamps and data for model fitting
timestamps = [datetime.now() + timedelta(seconds=i * time_interval) for i in range(n_init_data)]
data_stream = np.array([generate_data_point(t) for t in range(n_init_data)]).reshape(-1, 1)
model.fit(data_stream)

# Set up the real-time plot with empty data initially
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-', label='Data Stream')
anomaly_points, = ax.plot([], [], 'ro', label='Anomalies')
ax.legend()
plt.xlabel('Time')
plt.ylabel('Value')

# Main loop: generate, detect, and visualize anomalies in real-time
try:
    t = n_init_data
    while plt.fignum_exists(fig.number):  # Check if the figure is still open
        # Generate a new data point
        timestamp = datetime.now()
        data_point = generate_data_point(t)
        t += 1

        # Detect anomaly using Isolation Forest
        is_anomaly = model.predict([[data_point]])[0] == -1

        # Append new timestamp and data point
        timestamps.append(timestamp)
        data_stream = np.append(data_stream, [[data_point]], axis=0)

        # Update plot only after the first data point is generated
        if t > n_init_data:
            line.set_xdata(timestamps)
            line.set_ydata(data_stream)

            if is_anomaly:
                anomaly_points.set_xdata(np.append(anomaly_points.get_xdata(), timestamp))
                anomaly_points.set_ydata(np.append(anomaly_points.get_ydata(), data_point))

            # Adjust plot limits for real-time visualization
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)

        # Incremental model retraining every 10 data points
        if t % 10 == 0:
            model.fit(data_stream[-n_init_data:])

        # Wait for the next data point
        time.sleep(time_interval)

except KeyboardInterrupt:
    print("Real-time anomaly detection stopped.")
finally:
    plt.ioff()
