import nni
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import numpy as np
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import mean_squared_error

def create_lstm_dataset(data, dates, time_steps=1, forecast_days=60):
    X, y, date_list = [], [], []
    for i in range(len(data) - time_steps - forecast_days + 1):
        end_ix = i + time_steps
        seq_x, seq_y = data[i:end_ix, :], data[end_ix + forecast_days - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
        date_list.append(dates[end_ix + forecast_days - 1])  # Сохраняем соответствующую дату
    return np.array(X), np.array(y), date_list


# Define the function to create LSTM dataset and save it
def create_and_save_lstm_dataset(file_id, n_intervals, forecast_days, datasets_folder='../datasets',
                                 scalers_folder='../scalers'):
    file_path = '../data_updated/' + file_names[file_id]
    dataset = pd.read_csv(file_path).dropna()

    target_column = dataset.pop('Цена')
    dates = dataset.pop('Дата')
    dataset['Цена'] = target_column

    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_dataset = scaler.fit_transform(dataset)

    X_ltsm, y_ltsm, date_list = create_lstm_dataset(normalized_dataset, dates, n_intervals, forecast_days=forecast_days)

    train_size = int(len(X_ltsm) * 0.8)
    X_train_ltsm, X_test_ltsm = X_ltsm[:train_size], X_ltsm[train_size:]
    y_train_ltsm, y_test_ltsm = y_ltsm[:train_size], y_ltsm[train_size:]

    scaler_filename = '../scalers/' + file_names[file_id] + '_scaler.pkl'
    with open(scaler_filename, 'wb') as scaler_file:
        joblib.dump(scaler, scaler_file)

    # Сохранение данных X_train и y_train_ltsm
    X_train_ltsm_file_path = os.path.join(datasets_folder, f'X_train_ltsm_{file_names[file_id]}')
    X_test_ltsm_file_path = os.path.join(datasets_folder, f'X_test_ltsm_{file_names[file_id]}')
    y_train_ltsm_file_path = os.path.join(datasets_folder, f'y_train_ltsm_{file_names[file_id]}')
    y_test_ltsm_file_path = os.path.join(datasets_folder, f'y_test_ltsm_{file_names[file_id]}')

    np.save(X_train_ltsm_file_path, X_train_ltsm)
    np.save(X_test_ltsm_file_path, X_test_ltsm)
    np.save(y_train_ltsm_file_path, y_train_ltsm)
    np.save(y_test_ltsm_file_path, y_test_ltsm)

    print(f'Data saved to {X_train_ltsm_file_path} and {X_test_ltsm_file_path}')
    print(f'Data saved to {y_train_ltsm_file_path} and {y_test_ltsm_file_path}')
    return date_list


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import sqrt
from keras.losses import mean_squared_error

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
def get_default_parameters():
    '''Получите параметры по умолчанию'''
    params = {'units1': 100,
              'units2': 100,
              'units3': 100,
              'dropout_rate_1': 0.2,
              'dropout_rate_2': 0.2,
              'dropout_rate_3': 0.2,
              'n_intervals': 256,
              'forecast_days': 256,
              }
    return params


RECEIVED_PARAMS = nni.get_next_parameter()
PARAMS = get_default_parameters()
PARAMS.update(RECEIVED_PARAMS)
# Загрузка данных
file_id = 1  # Замените на нужный идентификатор файла
n_intervals = PARAMS['n_intervals']  # Замените на необходимое количество временных интервалов
n_features = 18  # Замените на количество признаков в данных (ваш случай)
forecast_days = PARAMS['forecast_days']
date_list = create_and_save_lstm_dataset(file_id, n_intervals, forecast_days=forecast_days)

data, scaler = load_data(file_id, n_intervals, n_features)
# Преобразование date_list в DataFrame
date_df = pd.DataFrame({'Date': date_list})
# Преобразование столбца 'Date' в формат datetime
date_df['Date'] = pd.to_datetime(date_df['Date'])
X_test_ltsm, X_train_ltsm, y_test_ltsm, y_train_ltsm = data
len(X_train_ltsm)





def create_model(X_train_ltsm, forecast_days, PARAMS):
    model = Sequential()
    model.add(
        LSTM(units=PARAMS['units1'], return_sequences=True, input_shape=(X_train_ltsm.shape[1], X_train_ltsm.shape[2])))
    model.add(Dropout(PARAMS['dropout_rate_1']))
    model.add(LSTM(units=PARAMS['units2'], return_sequences=False))
    model.add(Dropout(PARAMS['dropout_rate_2']))
    model.add(Dense(units=PARAMS['units3'], activation='relu'))
    model.add(Dropout(PARAMS['dropout_rate_3']))
    model.add(Dense(units=1, activation='relu'))

    model.compile(optimizer='adam', loss='mse')
    return model


# Создание модели
model = create_model(X_train_ltsm, forecast_days=1, PARAMS=PARAMS)

# Обучение модели
history = train_model(model, X_train_ltsm, y_train_ltsm, X_test_ltsm, y_test_ltsm, file_id)
results = model.evaluate(X_test_ltsm, y_test_ltsm, batch_size=32)

nni.report_final_result(results)

# %%
# yhat = model.predict(X_test_ltsm)
# X_train_ltsm_res = X_train_ltsm.reshape((X_train_ltsm.shape[0], n_intervals * n_features))
# X_test_ltsm_res = X_test_ltsm.reshape((X_test_ltsm.shape[0], n_intervals * n_features))
#
# inv_y_train = invert_scaling_for_actual(y_train_ltsm, X_train_ltsm_res, n_features, scaler)
# inv_yhat = invert_scaling_for_forecast(yhat, X_test_ltsm_res, n_features, scaler)
# inv_y_test = invert_scaling_for_actual(y_test_ltsm, X_test_ltsm_res, n_features, scaler)
#
# yhat_train = model.predict(X_train_ltsm)
# inv_yhat_train = invert_scaling_for_forecast(yhat_train, X_train_ltsm_res, n_features, scaler)
# # Создаем датафреймы для каждого массива
# df_yhat_train = pd.DataFrame({'yhat_train': inv_yhat_train})
# df_yhat = pd.DataFrame({'yhat': inv_yhat})
# df_y_test = pd.DataFrame({'y_test': inv_y_test})
# df_y_train = pd.DataFrame({'y_train': inv_y_train})
#
# # Объединяем их в один датафрейм
# result_df_pred = pd.concat([df_yhat_train, df_yhat], axis=0)
# result_df_true = pd.concat([df_y_train, df_y_test], axis=0)
# result_df_true = result_df_true.reset_index()
# result_df_pred = result_df_pred.reset_index()
# result_df_true['Дата'] = date_df
# results_all = pd.concat([result_df_true, result_df_pred], axis=1)
# results_all = results_all.drop(columns='index')
# # Создаем случайные данные для примера
# days = results_all['Дата']
# results_all['Shifted_y_test'] = results_all['y_test'].shift(forecast_days)
#
# current_prices = results_all['Shifted_y_test']
# predictions = results_all['yhat']
#
# # Инициализация переменных для стратегии
# buy_points = []
# sell_points = []
#
# position = None
# buy_price = None
# sell_price = None
# profit = 0
#
# commission_rate = 0.04  # Комиссия в 4%
# price_trashhold = 5
# # Применение торговой стратегии
# for i in range(0, len(current_prices)):
#     current_price = current_prices.iloc[i]
#     next_day_prediction = predictions.iloc[i]
#     percentage_change = ((current_price - next_day_prediction) / next_day_prediction) * 100
#     price_difference = abs(percentage_change)
#     if price_difference > price_trashhold:
#         if next_day_prediction > current_price:
#             if position != "buy":
#                 position = "buy"
#                 buy_price = current_price * (1 + commission_rate)  # Учитываем комиссию при покупке
#                 profit -= buy_price
#                 buy_points.append(i)
#                 print('BUY', i)
#                 print(current_price)
#                 print(next_day_prediction)
#         else:
#             if position == "buy":
#                 position = "sell"
#                 sell_price = current_price * (1 - commission_rate)  # Учитываем комиссию при продаже
#                 profit += sell_price
#                 sell_points.append(i)
#                 print('SELL', i)
#                 print(current_price)
#                 print(next_day_prediction)
#
# current_prices = results_all['Shifted_y_test']
#
# for i in range(len(buy_points) - len(sell_points)):
#     profit += current_prices.dropna().iloc[-1] * (1 - commission_rate)
# # График для Current_Price
# buy_dates = days.iloc[buy_points]
# sell_dates = days.iloc[sell_points]
# nni.report_final_result(profit)
