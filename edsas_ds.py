"""
Data stream anolagy detection with pre-dereminted data set (csv file) using Isolation Forest Model
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


data = pd.read_csv('continuous_dataset.csv', parse_dates=True, index_col='datetime')
selected_columns = ["nat_demand"]
data = data[selected_columns]
data.fillna(data.mean(), inplace=True)

iso_forest = IsolationForest(contamination=0.01)
iso_forest.fit(data)
anomaly_pred = iso_forest.predict(data)
data['anomaly'] = anomaly_pred

normal_data = data[data['anomaly'] == 1]
anomalies = data[data['anomaly'] == -1]

fig, ax = plt.subplots(figsize=(20, 10))
line, = ax.plot(data.index, data['nat_demand'], label='nat_demand', color='blue')
anomaly_dots, = ax.plot(anomalies.index, anomalies['nat_demand'], 'ro', label="Anomalies")
ax.legend()

ax.set_xlim(data.index.min(), data.index.max())
ax.set_ylim(data['nat_demand'].min() - 10, data['nat_demand'].max() + 10)

plt.title("Efficient Data Stream Anomaly Detection")
plt.xlabel('Datetime')
plt.ylabel('Nat Demand')
plt.grid()
plt.show()
