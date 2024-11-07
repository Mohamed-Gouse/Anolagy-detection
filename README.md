# Efficient Data Stream Anomaly Detection Scripts

*This repository contains four scripts for detecting anomalies in a data stream using various models. Each script simulates or processes a data stream and applies a different anomaly detection technique to identify unusual patterns.*

---

# Algorithms Used
**Isolation Forest:**
   - [Overview:] Isolation Forest is an unsupervised learning algorithm primarily used for anomaly detection. It operates by isolating anomalies rather than profiling normal data points. The algorithm works by building a series of decision trees that attempt to split data into increasingly smaller segments. Anomalies are isolated in fewer steps because they differ significantly from the majority of the data.
   - [Why_Chosen:] Isolation Forest is particularly efficient for real-time anomaly detection in high-dimensional datasets and is effective in detecting outliers with minimal computational cost. Since this project aims to identify anomalies in real-time data streams, the algorithm's efficiency and adaptability make it a suitable choice for detecting unusual patterns and deviations from the norm.

**ARIMA (AutoRegressive Integrated Moving Average):**
   - [Overview:] ARIMA is a time-series forecasting model that uses past values to predict future points in a series. The model combines autoregressive (AR), differencing (I for Integrated), and moving average (MA) components to make predictions. It’s widely used in applications where understanding and forecasting temporal patterns is essential.
   - [Why_Chosen:] For data streams with strong temporal dependencies or seasonal patterns, ARIMA is highly effective in forecasting values based on past behavior, allowing for accurate detection of anomalies based on deviations from predicted values. Given the project’s focus on real-time anomalies in streaming data, ARIMA provides robust forecasting capabilities that enable it to flag patterns that deviate from expected behavior over time.

---

# Scripts

# 1. Real-Time Data Stream Anomaly Detection using ARIMA Model (Real-time) `edsad_arima_rt.py`

**Description:**
   This script extends ARIMA-based anomaly detection for real-time data streams, adapting to the incoming data in a continuous fashion. Instead of processing pre-defined data, it generates a real-time data stream and detects anomalies live as new data points are generated.

**Key Features:**
   - Real-time data stream generation with sinusoidal patterns, random noise, and periodic anomalies.
   - Anomaly detection using a sliding ARIMA model.
   - Interactive plot updating in real time, with anomalies highlighted in red.

**Explanation:**
   - *Data Generation:* The `generate_data_point(t)` function simulates data points with seasonal components and occasional anomalies.
   - *ARIMA Model Training:* The model is trained on a rolling window of recent data to forecast the next point, with anomalies detected based on residual thresholds.
   - *Live Plotting:* Using `matplotlib`'s interactive mode (`plt.ion()`), the script displays an updating plot, marking detected anomalies in red as the data stream continues.
   - *Infinite Data Stream:* The script will continue generating and analyzing data until manually interrupted or the plot window is closed.

---

# 2. Data Stream Anomaly Detection using ARIMA Model `edsad_arima.py`

**Description:**
   This script simulates a data stream and detects anomalies using an ARIMA (AutoRegressive Integrated Moving Average) model. The data stream consists of a seasonal pattern, noise, and anomalies. The ARIMA model is used to predict the next data point, and anomalies are detected based on the residuals (the difference between the actual and predicted values).

**Key Features:**
   - Generates a synthetic data stream with seasonal and noise components.
   - Simulates anomalies in the data.
   - Uses ARIMA for anomaly detection.
   - Visualizes the data stream and detected anomalies.

**Explanation:**
   - *Data Generation:* The function `generate_data_point(t)` generates a synthetic data point with a seasonal component (sinusoidal), noise, and occasional anomalies.
   - *Data Stream Simulation:* The `simulate_data_stream` function generates a stream of data points over time.
   - *Anomaly Detection:* The `detect_anomalies` function fits an ARIMA model to a window of the previous data points and forecasts the next data point. Anomalies are detected if the residual exceeds a predefined threshold.
   - *Plotting:* The results are visualized with `matplotlib`, with detected anomalies highlighted in red.

---

# 3. Real-Time Data Stream Anomaly Detection using Isolation Forest Model `edsad_rt.py`

**Description:**
   This script performs real-time anomaly detection on a continuously generated data stream using the Isolation Forest model. The model is incrementally trained as new data points are generated. Anomalies are identified based on the Isolation Forest's predictions.

**Key Features:**
   - Real-time simulation of a data stream with seasonal and daily patterns.
   - Anomaly detection using Isolation Forest.
   - Real-time plotting and visualization of the data stream and anomalies.

**Explanation:**
   - *Data Generation:* The function `generate_data_point(t)` generates a new data point at each time step, incorporating seasonal patterns, daily fluctuations, and noise.
   - *Anomaly Detection:* An Isolation Forest model is trained on the initial data points and used to detect anomalies in real time.
   - *Real-Time Plotting:* The script uses `matplotlib`'s interactive mode (`plt.ion()`) to display the data stream in real time, marking anomalies with red dots.
   - *Model Retraining:* The model is retrained every 10 new data points to adapt to changes in the data stream.

---

# 4. Data Stream Anomaly Detection with Pre-determined Dataset (CSV) using Isolation Forest `edsad_ds.py`

**Description:**
   This script reads a pre-determined CSV dataset containing a data stream (e.g., power demand) and performs anomaly detection using the Isolation Forest model. It identifies and visualizes anomalies based on historical data.

**Key Features:**
   - Reads data from a CSV file (`continuous_dataset.csv`).
   - Performs anomaly detection using Isolation Forest.
   - Visualizes detected anomalies and normal data in a time series plot.

**Explanation:**
   - *Data Loading:* The script loads the dataset from a CSV file using `pandas`, selecting the `nat_demand` column and filling missing values with the mean.
   - *Anomaly Detection:* An Isolation Forest model is trained on the dataset, and anomalies are predicted. Predictions (`1` for normal and `-1` for anomaly) are added as a new column (`'anomaly'`).
   - *Plotting:* Anomalies are highlighted in red, with normal data plotted in blue.

---

# Requirements

To run these scripts, you'll need the following Python packages:

- `numpy`
- `matplotlib`
- `statsmodels` (for ARIMA)
- `sklearn` (for Isolation Forest)
- `pandas` (for handling CSV data)

Install the required packages using:

```bash```
pip install numpy matplotlib statsmodels scikit-learn pandas

---

# How to Use

1. **Data Stream Anomaly Detection using ARIMA Model:**
   - Run the script to simulate the data stream and detect anomalies using ARIMA.
   - The plot displays anomalies, and you can adjust the threshold or ARIMA parameters to fine-tune detection.

2. **Real-Time Data Stream Anomaly Detection using Isolation Forest Model:**
   - Start this script to simulate real-time data stream anomaly detection.
   - The model is retrained every 10 new points to keep pace with the changing stream.

3. **Data Stream Anomaly Detection with Pre-determined Dataset (CSV) using Isolation Forest:**
   - Place `continuous_dataset.csv` in the script directory.
   - Run the script to load data, perform anomaly detection, and visualize results.

4. **Real-Time Data Stream Anomaly Detection using ARIMA Model (Real-time):**
   - Run this script to generate a real-time data stream and perform anomaly detection with a continuously updating ARIMA model.
   - Anomalies will appear on the live plot, which updates in real time.

---

# Example Outputs

1. *Real-Time ARIMA Model Anomaly Detection:* Continuously updating time series plot with anomalies highlighted in red.
2. *ARIMA Anomaly Detection:* Time series plot with anomalies marked in red, based on ARIMA residuals.
3. *Real-Time Isolation Forest Anomaly Detection:* Live plot with anomalies highlighted in real-time.
4. *Pre-determined Dataset Isolation Forest Anomaly Detection:* Time series plot with normal data (blue) and anomalies (red).

---

# Conclusion
   These scripts provide a foundation for detecting anomalies in data streams using various methods. You can modify the parameters or integrate them into larger systems based on your requirements. Each method offers a different approach to anomaly detection, making the repository versatile for various use cases.
