# AI-Based Short-Term Load Forecasting Using CNN-LSTM

> **This project repository is part of my thesis:**  
> *AI-Based Modeling for Future Electric Load Forecasting in Nineveh Governorate*

---

## About the Project

In this project, I led a group of 3 students to create a deep learning model that predicts 7-days ahead electric load in Nineveh, Iraq.
We used historical electrical load data of the Nineveh power grid and weather data from Open-meteo API to improve prediction accuracy. The project uses a combination of CNN and LSTM layers.

This work was developed as part of our B.Sc. graduation thesis.

## Model Architecture

```
Input → Conv1D (336 filters, kernel=4, tanh) → MaxPooling1D → Dropout(0.2)
      → LSTM (48 units) → Dropout(0.3) → Flatten → Dense(1)
```

## Input Features

The model uses a **7-day lookback window** with the following features:

- **Load Features:** Average load, supply hours, cut hours  
- **Weather Features:** Temperature (mean, max, min), humidity, precipitation, wind speed, pressure, cloud cover  
- **Calendar Features:** Day of month, month, day-type encoding

## Dataset

> **Note:** The CSV files in this repository are **small samples** only. The full dataset used in the is not published.

| File | Description |
|------|-------------|
| `Sample_load_2022_2025.csv` | Sample daily load data (avg load, supply hours, cut hours) |
| `WeatherData.csv` | Sample daily weather data from Open-Meteo |



## Evaluation Metrics

The model is evaluated using:

| Metric | Description |
|--------|-------------|
| **MAPE** | Mean Absolute Percentage Error |
| **MSE**  | Mean Squared Error |
| **RMSE** | Root Mean Squared Error |
| **MAE**  | Mean Absolute Error |

## Requirements

See [`requirements.txt`](requirements.txt) for the full list.