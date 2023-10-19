# List of file names to process
file_names = [
    'Прошлые данные - LKOH.csv',
    'Прошлые данные - INGR.csv',
    'Прошлые данные - LENT.csv',
    'Прошлые данные - LSRG.csv',
    'Прошлые данные - MVID.csv',
    'Прошлые данные - NVTK.csv',
    'Прошлые данные - OZONDR.csv',
    'Прошлые данные - PIKK.csv',
    'Прошлые данные - ROSN.csv',
    'Прошлые данные - FIVEDR.csv',
    'Прошлые данные - SMLT.csv',
    'Прошлые данные - GAZP (3).csv'
]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']
# Folder for saving the processed data
data_folder = 'data_updated'
n_intervals = 120
import joblib
import os
import numpy as np
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import mean_squared_error


# Функция для формирования строки с описанием архитектуры модели
def get_model_architecture(model):
    architecture = ""
    for layer in model.layers:
        if isinstance(layer, LSTM):
            architecture += f"LSTM_{layer.units}x{layer.return_sequences}_"
        elif isinstance(layer, Dense):
            architecture += f"Dense{layer.units}_"
    return architecture[:-1]


def load_data(file_id, n_intervals, n_features, data_folder='../datasets'):
    scaler_filename = '../scalers/' + file_names[file_id] + '_scaler.pkl'
    with open(scaler_filename, 'rb') as scaler_file:
        scaler = joblib.load(scaler_file)

    file_paths = ['X_test_ltsm_', 'X_train_ltsm_', 'y_test_ltsm_', 'y_train_ltsm_']
    data = []

    for file_path in file_paths:
        file_name = file_path + file_names[file_id] + '.npy'
        full_path = os.path.join(data_folder, file_name)
        data.append(np.load(full_path))

    return data, scaler


def create_model(X_train_ltsm):
    model = Sequential()
    model.add(LSTM(units=77, return_sequences=True, input_shape=(X_train_ltsm.shape[1], X_train_ltsm.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=77, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=77, activation='relu'))
    model.add(Dense(units=1, activation='linear'))

    model.compile(optimizer='adam', loss='mse')
    return model


def train_model(model, X_train_ltsm, y_train_ltsm, X_test_ltsm, y_test_ltsm, file_id, models_dir='models'):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_filename = os.path.join(models_dir, f'{file_names[file_id]}_{get_model_architecture(model)}_best_model.h5')
    model_checkpoint = ModelCheckpoint(model_filename, monitor='val_loss', save_best_only=True)

    history = model.fit(X_train_ltsm, y_train_ltsm,
                        epochs=30, batch_size=32,
                        validation_data=(X_test_ltsm, y_test_ltsm),
                        callbacks=[early_stopping, model_checkpoint])
    return history


def plot_training_history(history):
    plt.plot(history.history['loss'][3:], label='Train Loss')
    plt.plot(history.history['val_loss'][3:], label='Validation Loss')
    plt.legend()
    plt.show()


def invert_scaling_for_forecast(y_forecast, x_on_forecast, n_features, scaler):
    inv_yhat = np.concatenate((y_forecast, x_on_forecast[:, 1 - n_features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    return inv_yhat


def invert_scaling_for_actual(y_actual, x_actual, n_features, scaler):
    y_test_ltsm = y_actual.reshape((len(y_actual), 1))
    inv_y_test = np.concatenate((y_test_ltsm, x_actual[:, 1 - n_features:]), axis=1)
    inv_y_test = scaler.inverse_transform(inv_y_test)
    inv_y_test = inv_y_test[:, 0]
    return inv_y_test


def plot_results(inv_y_train, inv_y_test, inv_yhat):
    plt.figure(figsize=(12, 6))
    plt.plot(inv_y_train, label='Actual (Train)')
    plt.plot(range(len(inv_y_train), len(inv_y_train) + len(inv_y_test)), inv_y_test, label='Actual (Test)')
    plt.plot(range(len(inv_y_train), len(inv_y_train) + len(inv_y_test)), inv_yhat, label='Predicted (Test)')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def forecast_next_days(model, X_test_ltsm, y_test_ltsm, forecast_days, n_intervals, n_features, scaler):
    # Используем исходные данные из тестового набора для первых 60 дней
    forecast = y_test_ltsm[-n_intervals:].tolist()
    forecast = [item for sublist in forecast for item in sublist]
    # Используем последние 60 дней из тестового набора
    last_60_days = X_test_ltsm[-n_intervals:]

    for i in range(forecast_days):
        # Используйте модель для прогнозирования следующего дня
        next_day_prediction = model.predict(last_60_days)
        next_day = next_day_prediction
        # Добавьте прогноз в список прогнозов
        forecast.append(next_day[0, 0])

        # Обновите last_60_days, удалив первый день и добавив новый прогноз
        last_60_days = np.roll(last_60_days, shift=-1, axis=0)
        last_60_days[-1] = next_day

    forecast = forecast[:n_intervals]

    # Обратное масштабирование прогнозов
    forecast = np.array(forecast)
    # Повторяем прогнозы для каждого признака
    forecast = np.repeat(forecast, n_features).reshape(-1, n_features)
    forecast = scaler.inverse_transform(forecast)[:, 0]
    forecast = forecast[::-1]
    return forecast


# Visualize the results
def plot_forecast(inv_y_train, inv_y_test, inv_yhat, forecast, forecast_days):
    plt.figure(figsize=(12, 6))
    plt.plot(inv_y_train, label='Actual (Train)')
    plt.plot(range(len(inv_y_train), len(inv_y_train) + len(inv_y_test)), inv_y_test, label='Actual (Test)')
    plt.plot(range(len(inv_y_train), len(inv_y_train) + len(inv_y_test)), inv_yhat, label='Predicted (Test)')
    plt.plot(range(len(inv_y_train) + len(inv_y_test), len(inv_y_train) + len(inv_y_test) + len(forecast)), forecast,
             label=f'Forecast (Next {forecast_days} days)')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def calculate_metrics(y_true, y_pred):
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # R-squared (Coefficient of Determination)
    r2 = r2_score(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R-squared": r2
    }
