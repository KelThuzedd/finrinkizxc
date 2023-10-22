def create_lstm_dataset(data, dates, time_steps=1, forecast_days=60):
    X, y, date_list = [], [], []
    for i in range(len(data) - time_steps - forecast_days + 1):
        end_ix = i + time_steps
        seq_x, seq_y = data[i:end_ix, :], data[end_ix + forecast_days - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
        date_list.append(dates[end_ix + forecast_days - 1])  # Сохраняем соответствующую дату
    return np.array(X), np.array(y), date_list


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


def invert_scaling_for_forecast(y_forecast, x_on_forecast, n_features, scaler):
    inv_yhat = np.concatenate((y_forecast, x_on_forecast[:, 1 - n_features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    return inv_yhat


import joblib
import os


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


def invert_scaling_for_actual(y_actual, x_actual, n_features, scaler):
    y_test_ltsm = y_actual.reshape((len(y_actual), 1))
    inv_y_test = np.concatenate((y_test_ltsm, x_actual[:, 1 - n_features:]), axis=1)
    inv_y_test = scaler.inverse_transform(inv_y_test)
    inv_y_test = inv_y_test[:, 0]
    return inv_y_test


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


import nni
import logging
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1, l2
import pandas as pd
from tensorflow.python.keras.layers import LSTM
import tensorflow as tf

early_stopping_callback = EarlyStopping(
    monitor='val_loss',  # Мониторим функцию потерь на валидационном наборе
    patience=3,  # Количество эпох без улучшения перед остановкой
    restore_best_weights=True  # Восстановить веса модели к лучшему состоянию
)


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')  # Получаем значение val_loss из логов
        # Отправляем значение val_loss в NNI
        nni.report_intermediate_result(val_loss)


custom_callback = CustomCallback()

LOG = logging.getLogger('keras_regression')
LOG.setLevel(logging.DEBUG)

# Создайте обработчик для записи логов в файл nni_log.txt
file_handler = logging.FileHandler('nni_log.txt')
file_handler.setLevel(logging.DEBUG)

# Определите формат записи логов
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Добавьте обработчик к логгеру
LOG.addHandler(file_handler)
file_handler.flush()


def get_default_parameters():
    '''Получите параметры по умолчанию'''
    params = {'units1': 100,
              'units2': 100,
              'units3': 100,
              'batch_size': 64,
              'l1_rate_1': 0,
              'l1_rate_2': 0,
              'l1_rate_3': 0,
              'l2_rate_1': 0,
              'l2_rate_2': 0,
              'l2_rate_3': 0,
              'dropout_rate_1': 0.2,
              'dropout_rate_2': 0.2,
              'dropout_rate_3': 0.2,
              'learning_rate': 0.001,
              }
    return params


def build_model(PARAMS, file_id, n_intervals, n_features,forecast_days):
    date_list = create_and_save_lstm_dataset(file_id, n_intervals, forecast_days=forecast_days)

    data, scaler = load_data(file_id, n_intervals, n_features)
    # Преобразование date_list в DataFrame
    date_df = pd.DataFrame({'Date': date_list})
    # Преобразование столбца 'Date' в формат datetime
    date_df['Date'] = pd.to_datetime(date_df['Date'])
    X_test_ltsm, X_train_ltsm, y_test_ltsm, y_train_ltsm = data

    # Создаем оптимизатор с установленным learning rate
    custom_optimizer = Adam(learning_rate=PARAMS['learning_rate'])

    model = Sequential()
    model.add(
        LSTM(units=PARAMS['units1'], return_sequences=True, input_shape=(X_train_ltsm.shape[1], X_train_ltsm.shape[2]),
             kernel_regularizer=l1(PARAMS['l1_rate_1']),
             activity_regularizer=l2(PARAMS['l2_rate_1'])))
    model.add(Dropout(PARAMS['dropout_rate_1']))
    model.add(LSTM(units=PARAMS['units2'], return_sequences=False, kernel_regularizer=l1(PARAMS['l1_rate_2']),
                   activity_regularizer=l2(PARAMS['l2_rate_2'])))
    model.add(Dropout(PARAMS['dropout_rate_2']))
    model.add(LSTM(units=PARAMS['units3'], kernel_regularizer=l1(PARAMS['l1_rate_3']),
                   activity_regularizer=l2(PARAMS['l2_rate_3'])))
    model.add(Dropout(PARAMS['dropout_rate_3']))
    model.add(Dense(units=1))

    model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
    return model, y_train_ltsm, y_test_ltsm, X_train_ltsm, X_test_ltsm, scaler, date_df


def run(X_train_ltsm, X_test_ltsm, y_train_ltsm, y_test_ltsm, model, scaler, date_df):
    try:
        model.fit(X_train_ltsm, y_train_ltsm, epochs=30, batch_size=PARAMS['batch_size'],
                  validation_data=(X_test_ltsm, y_test_ltsm),
                  callbacks=[custom_callback, early_stopping_callback])
        # loss = model.evaluate(X_test, y_test, verbose=2)
        yhat = model.predict(X_test_ltsm)
        X_train_ltsm_res = X_train_ltsm.reshape((X_train_ltsm.shape[0], n_intervals * n_features))
        X_test_ltsm_res = X_test_ltsm.reshape((X_test_ltsm.shape[0], n_intervals * n_features))

        inv_y_train = invert_scaling_for_actual(y_train_ltsm, X_train_ltsm_res, n_features, scaler)
        inv_yhat = invert_scaling_for_forecast(yhat, X_test_ltsm_res, n_features, scaler)
        inv_y_test = invert_scaling_for_actual(y_test_ltsm, X_test_ltsm_res, n_features, scaler)

        yhat_train = model.predict(X_train_ltsm)
        inv_yhat_train = invert_scaling_for_forecast(yhat_train, X_train_ltsm_res, n_features, scaler)

        # Создаем датафреймы для каждого массива
        df_yhat_train = pd.DataFrame({'yhat_train': inv_yhat_train})
        df_yhat = pd.DataFrame({'yhat': inv_yhat})
        df_y_test = pd.DataFrame({'y_test': inv_y_test})
        df_y_train = pd.DataFrame({'y_train': inv_y_train})

        # Объединяем их в один датафрейм
        result_df_pred = pd.concat([df_yhat_train, df_yhat], axis=0)
        result_df_true = pd.concat([df_y_train, df_y_test], axis=0)
        result_df_true = result_df_true.reset_index()
        result_df_pred = result_df_pred.reset_index()
        result_df_true['Дата'] = date_df
        results_all = pd.concat([result_df_true, result_df_pred], axis=1)
        results_all = results_all.drop(columns='index')
        results_all['Shifted_y_test'] = results_all['y_test'].shift(-30)

        # Создаем случайные данные для примера
        current_prices = results_all['Shifted_y_test']
        predictions = results_all['yhat']

        # Инициализация переменных для стратегии
        buy_points = []
        sell_points = []

        position = None
        buy_price = None
        sell_price = None
        profit = 0

        commission_rate = 0.04  # Комиссия в 4%
        price_trashhold = 10
        # Применение торговой стратегии
        for i in range(0, len(current_prices)):
            current_price = current_prices.iloc[i]
            next_day_prediction = predictions.iloc[i]
            percentage_change = ((current_price - next_day_prediction) / next_day_prediction) * 100
            price_difference = abs(percentage_change)
            if price_difference > price_trashhold:
                if next_day_prediction > current_price:
                    if position != "buy":
                        position = "buy"
                        buy_price = current_price * (1 + commission_rate)  # Учитываем комиссию при покупке
                        profit -= buy_price
                        buy_points.append(i)
                        print('BUY', i)
                        print(current_price)
                        print(next_day_prediction)
                else:
                    if position == "buy":
                        position = "sell"
                        sell_price = current_price * (1 - commission_rate)  # Учитываем комиссию при продаже
                        profit += sell_price
                        sell_points.append(i)
                        print('SELL', i)
                        print(current_price)
                        print(next_day_prediction)

        current_prices = results_all['Shifted_y_test']

        for i in range(len(buy_points) - len(sell_points)):
            profit += current_prices.dropna().iloc[-1] * (1 - commission_rate)
            print('zxc', current_prices.dropna().iloc[-1] * (1 - commission_rate))

        score = profit
        LOG.debug('profit', score)
        LOG.debug('Function run')
        nni.report_final_result(score)
    except Exception as e:
        LOG.exception(e)


if __name__ == '__main__':
    try:
        # Загрузка данных
        file_id = 5  # Замените на нужный идентификатор файла
        n_intervals = 300  # Замените на необходимое количество временных интервалов
        n_features = 18  # Замените на количество признаков в данных (ваш случай)
        forecast_days = 30
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        model, y_train, y_test, X_train, X_test, scaler, date_df = build_model(PARAMS, file_id, n_intervals, n_features,forecast_days)
        LOG.debug('Parameters received')
        run(X_train, X_test, y_train, y_test, model, scaler=scaler, date_df=date_df)
        LOG.debug('Model run')
    except Exception as exception:
        LOG.exception(exception)
        LOG.debug('Error')
        raise
