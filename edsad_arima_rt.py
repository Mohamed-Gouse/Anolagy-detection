import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import time

# Parameters for the data stream
np.random.seed(42)
seasonal_period = 50         # Number of points in the seasonal cycle
noise_level = 0.5            # Standard deviation of random noise
anomaly_chance = 0.05        # Probability of an anomaly at any point
anomaly_magnitude = 5        # Magnitude of the anomaly
window_size = 50             # Number of points used to fit the ARIMA model
threshold = 2.0              # Threshold for detecting an anomaly

# Generate a single data point with seasonal, noise, and potential anomaly components
def generate_data_point(t):
    seasonal = np.sin(2 * np.pi * t / seasonal_period)
    noise = np.random.normal(0, noise_level)
    anomaly = np.random.normal(0, anomaly_magnitude) if np.random.rand() < anomaly_chance else 0
    return seasonal + noise + anomaly

# Initialize real-time data stream and anomaly tracking
data_stream = []
anomalies = []

# Set up live plotting
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-', label='Data Stream')
anomaly_points, = ax.plot([], [], 'ro', label='Anomalies')
ax.legend()
plt.xlabel('Time')
plt.ylabel('Value')

# Simulate real-time data streaming
try:
    t = 0
    while True:  # Adjust as needed for how long to run

        if not plt.fignum_exists(fig.number):
            print("Plot closed. Stopping data stream.")
            break
        data_point = generate_data_point(t)
        data_stream.append(data_point)

        # Check for anomaly after the first window of data is available
        if len(data_stream) >= window_size:
            # Fit ARIMA model to the latest window of data
            train_data = data_stream[-window_size:]
            model = ARIMA(train_data, order=(1, 1, 1))
            model_fit = model.fit()
            
            # Forecast the next point and calculate residual
            forecast = model_fit.forecast()[0]
            residual = abs(data_point - forecast)
            
            # Detect anomaly based on residual threshold
            is_anomaly = residual > threshold
            if is_anomaly:
                anomalies.append((t, data_point))
            else:
                anomalies.append((t, None))

            # Update plot
            line.set_xdata(range(len(data_stream)))
            line.set_ydata(data_stream)
            anomaly_points.set_xdata([idx for idx, val in anomalies if val is not None])
            anomaly_points.set_ydata([val for _, val in anomalies if val is not None])
            
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.1)
        
        t += 1
        # Simulate real-time data stream with a delay
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Real-time data stream stopped.")
finally:
    plt.ioff()
    plt.show()
