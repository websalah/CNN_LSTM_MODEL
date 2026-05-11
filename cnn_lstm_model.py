import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense, LSTM, Dropout
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential

from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE


folder = os.getcwd()
dataset = folder + "/"
models_path = folder + "/models/"
os.makedirs(models_path, exist_ok=True)

print("=" * 60)
print("Loading data:")
print("=" * 60)
# Load consumption data
load_df = pd.read_csv(dataset + 'Sample_load_2022_2025.csv') # use full dataset instead of this sample dataset 
load_df['date'] = pd.to_datetime(load_df['date'])
load_df.index = load_df['date']
load_df = load_df[['avg_load', 'supply_hours', 'cut_hours', 'year', 'month', 'dow']]

# Load weather data
weather_df = pd.read_csv(dataset + 'WeatherData.csv')
weather_df['date'] = pd.to_datetime(weather_df['time'])
weather_df.index = weather_df['date']
weather_df = weather_df[[
    'temperature_2m_mean (°C)',
    'temperature_2m_max (°C)',
    'temperature_2m_min (°C)',
    'relative_humidity_2m_mean (%)',
    'precipitation_sum (mm)',
    'wind_speed_10m_mean (km/h)',
    'pressure_msl_mean (hPa)',
    'cloud_cover_mean (%)'
]]

# Rename columns for convenience
weather_df.columns = [
    'temperature', 'temp_max', 'temp_min',
    'humidity', 'precipitation',
    'wind_speed', 'pressure', 'cloud_cover'
]

# Combine into a single dataframe
data = pd.concat([load_df, weather_df], axis=1)
print(f"Combined data shape: {data.shape}")
print(f"Date range: {data.index.min()} to {data.index.max()}")

# Handle NaN values
print(f"\nNaN values before cleaning: {data.isnull().values.sum()}")

def fill_nan(df):
    """Replaces NaN values using forward and backward fill."""
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

data = fill_nan(data)
print(f"NaN values after cleaning: {data.isnull().values.sum()}")

# Feature extraction
print("\n" + "=" * 60)
print("Feature extraction:")
print("=" * 60)

# Day encoding
# pandas weekday uses Monday=0 (ISO convention)
weekday_map = {0: 'mon', 1: 'wkd', 2: 'wkd', 3: 'wkd', 4: 'fri', 5: 'wkn', 6: 'wkn'}
day_cat_series = data.index.weekday.map(weekday_map)
day_cat = pd.get_dummies(day_cat_series)[['mon', 'wkd', 'fri', 'wkn']]
day_cat.index = data.index

# Select features for modeling
feature_cols = [
    'avg_load', 'supply_hours', 'cut_hours',
    'temperature', 'temp_max', 'temp_min',
    'humidity', 'precipitation',
    'wind_speed', 'pressure', 'cloud_cover'
]

model_data = data[feature_cols].copy().astype(float)

scalers = {}
for feature in model_data.columns:
    scaler = StandardScaler()
    model_data[feature] = scaler.fit_transform(np.array(model_data[feature]).reshape(-1, 1))
    scalers[feature] = scaler

# Feature extraction
def extract_features(days_back, data_df, day_cat_df):
    """
    Creates feature vectors using daily lookback windows.
    Returns X (input features) and Y (target: next day's avg_load).
    """
    X = []
    Y = []

    weather_features = [
        'temperature', 'temp_max', 'temp_min',
        'humidity', 'precipitation',
        'wind_speed', 'pressure', 'cloud_cover'
    ]
    all_features = list(data_df.columns)

    max_back = max(days_back.values())
    n_days = len(data_df)

    for day_id in range(max_back, n_days - 1):
        # Historical features (lookback window for each feature)
        historical = []
        for feat in all_features:
            back = days_back.get(feat, days_back.get('default', 7))
            h = data_df[feat].iloc[day_id - back:day_id].values
            historical.append(h)

        # Day-ahead weather forecast
        next_day = day_id
        forecast = []
        for feat in weather_features:
            forecast.append(data_df[feat].iloc[next_day])
        forecast = np.array(forecast)

        # Calendar features
        next_date = data_df.index[day_id]
        C1 = next_date.day
        C2 = next_date.month
        C3 = np.array(day_cat_df.iloc[day_id])
        calendar = np.concatenate(([C1, C2], C3))

        # Build feature vector
        feature_vec = np.concatenate([np.concatenate(historical), forecast, calendar])
        X.append(feature_vec)

        # Target: next day avg_load
        Y.append(data_df['avg_load'].iloc[day_id])

    return np.array(X, dtype='float64'), np.array(Y, dtype='float64').reshape(-1, 1)


# Split into train/val/test
def split_train_test_val(A, n_test=365, n_val=365):
    """Splits data chronologically into train, validation and test sets."""
    n_instances = A.shape[0]
    n_train = n_instances - n_val - n_test

    train = A[:n_train]
    val = A[n_train:n_train + n_val]
    test = A[n_train + n_val:]

    return train, val, test


# Configure lookback days
days_back = {col: 7 for col in feature_cols}
days_back['default'] = 7

# Extract features
X, Y = extract_features(days_back, model_data, day_cat)
n_features = X.shape[1]
print("\nFeature extraction complete:")
print(f"  X shape: {X.shape} (samples x features)")
print(f"  Y shape: {Y.shape} (samples x 1)")

# Determine split sizes based on available data
total_samples = X.shape[0]
n_test = min(365, total_samples // 4)
n_val = min(365, total_samples // 4)
print(f"\n  Train: {total_samples - n_val - n_test}, Val: {n_val}, Test: {n_test}")

x_train, x_val, x_test = split_train_test_val(X, n_test, n_val)
y_train, y_val, y_test = split_train_test_val(Y, n_test, n_val)

print(f"  x_train: {x_train.shape}, x_val: {x_val.shape}, x_test: {x_test.shape}")
print(f"  y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")


# Train Final CNN-LSTM Model
print("\n" + "=" * 60)
print("TRAINING FINAL CNN-LSTM MODEL")
print("=" * 60)

# Build the final optimized model
model = Sequential([
    Input(shape=(n_features, 1)),
    Conv1D(336, kernel_size=4, activation='tanh'),
    MaxPooling1D(pool_size=4),
    Dropout(0.2),
    LSTM(48, return_sequences=True),
    Dropout(0.3),
    Flatten(),
    Dense(1)
])

model.compile(loss='mse', optimizer=Adam(learning_rate=0.00225))
model.summary()

# Define callbacks
loss_checkpoint = ModelCheckpoint(
    filepath=models_path + 'model_weights.weights.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    mode='min',
    verbose=1
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
batch_size = 16
epochs = 100

history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=[loss_checkpoint, early_stop]
)

# Load best weights
model.load_weights(models_path + 'model_weights.weights.h5')

# Evaluation
print("Evaluation\n")


def predict(model, x, y):
    """Generates predictions and inverse-transforms them."""
    y_pred = model.predict(x, verbose=0)
    y_pred = scalers['avg_load'].inverse_transform(y_pred)
    y_true = scalers['avg_load'].inverse_transform(y)
    err = mape(y_true, y_pred)
    return y_pred, y_true, err


train_pred, train_true, train_err = predict(model, x_train, y_train)
val_pred, val_true, val_err = predict(model, x_val, y_val)
test_pred, test_true, test_err = predict(model, x_test, y_test)

print(f"  Training   MAPE (overall) = {100 * train_err:.2f}%")
print(f"  Validation MAPE (overall) = {100 * val_err:.2f}%")
print(f"  Testing    MAPE (overall) = {100 * test_err:.2f}%\n")

print("  Per-day MAPE on Test Set:")
max_back = max(days_back.values())
test_dates = model_data.index[max_back + len(x_train) + len(x_val):max_back + len(x_train) + len(x_val) + len(x_test)]
test_df = pd.DataFrame({'Actual': test_true.reshape(-1), 'Predicted': test_pred.reshape(-1)}, index=test_dates)
test_df['DayOfWeek'] = test_df.index.dayofweek

day_mapes = []
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

for i in range(7):
    day_data = test_df[test_df['DayOfWeek'] == i]
    if len(day_data) > 0:
        day_mape = 100 * mape(day_data['Actual'], day_data['Predicted'])
        print(f"    {day_names[i]}: {day_mape:.2f}%")
        day_mapes.append(day_mape)
    else:
        print(f"    {day_names[i]}: N/A")
        day_mapes.append(0)

# Plot MAPE per day
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=day_names, y=day_mapes, hue=day_names, legend=False, palette='viridis')
plt.title('MAPE per Day of the Week on Testing Set', fontsize=15)
plt.ylabel('MAPE (%)', fontsize=12)
plt.xlabel('Day of the Week', fontsize=12)
plt.tight_layout()
plt.savefig(models_path + 'mape_per_day.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Plot saved to: {models_path}mape_per_day.png")

# Plot Actual vs Predicted for each day
for i in range(7):
    day_data = test_df[test_df['DayOfWeek'] == i]
    if len(day_data) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(x=day_data.index, y=day_data['Actual'], label='Actual', linewidth=2, color='#1f77b4')
        sns.lineplot(x=day_data.index, y=day_data['Predicted'], label='Predicted', linewidth=2, color='#ff7f0e')
        plt.xticks(rotation=15)
        plt.title(f'Actual vs Predicted Load on Testing Set - {day_names[i]}', fontsize=15)
        plt.ylabel('Load (MW)', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.legend(fontsize=12, loc='best')
        plt.tight_layout()
        plt.savefig(models_path + f'actual_vs_predicted_{day_names[i].lower()}.png', dpi=150, bbox_inches='tight')
        plt.close()
print(f"  Saved Actual vs Predicted plots for each day to {models_path}")

# Error metrics table
pairs = [(train_true, train_pred), (val_true, val_pred), (test_true, test_pred)]
errors = pd.DataFrame({
    'MAPE (%)': [100 * mape(true, pred) for (true, pred) in pairs],
    'MSE': [MSE(true, pred) for (true, pred) in pairs],
    'RMSE': [np.sqrt(MSE(true.reshape(-1), pred.reshape(-1))) for (true, pred) in pairs],
    'MAE': [MAE(true, pred) for (true, pred) in pairs]
}, index=['Training', 'Validation', 'Testing'])

print("\n  Error Metrics:")
print(errors.round(4).to_string())


# Actual vs Predicted Plot (Test Set - 2025)
print("\n" + "=" * 60)
print("GENERATING ACTUAL VS PREDICTED PLOT")
print("=" * 60)

sns.set_theme(style="ticks")

# Build test date range
max_back = max(days_back.values())
test_dates = model_data.index[max_back + len(x_train) + len(x_val):max_back + len(x_train) + len(x_val) + len(x_test)]

if len(test_dates) == len(test_pred):
    full_data = pd.DataFrame({
        'Actual': test_true.reshape(-1),
        'Predicted': test_pred.reshape(-1)
    }, index=test_dates)
else:
    full_data = pd.DataFrame({
        'Actual': test_true.reshape(-1),
        'Predicted': test_pred.reshape(-1)
    }, index=range(len(test_true)))

# Plot Actual vs Predicted
fig, ax = plt.subplots(figsize=(16, 6))
sns.lineplot(x=full_data.index, y=full_data['Actual'], label='Actual', linewidth=2, color='#1f77b4')
sns.lineplot(x=full_data.index, y=full_data['Predicted'], label='Predicted', linewidth=2, color='#ff7f0e')
plt.xticks(rotation=15)
plt.title('Daily Load Forecast (2025)', fontsize=15)
plt.ylabel('Load (MW)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(fontsize=12, loc='best')
plt.tight_layout()
plt.savefig(models_path + 'actual_vs_predicted_2025.png', dpi=150, bbox_inches='tight')
plt.close()

test_mape = 100 * mape(full_data['Actual'], full_data['Predicted'])
print(f"\n  Test Set MAPE: {test_mape:.2f}%")
print(f"  Plot saved to: {models_path}actual_vs_predicted_2025.png")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
