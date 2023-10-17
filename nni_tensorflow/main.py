import nni
import logging
from keras.optimizers import Adam
from keras.optimizers import Adam
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1, l2
from sklearn.metrics import r2_score
import pandas as pd
from tensorflow.python.keras.layers import LSTM
import tensorflow as tf
from keras import backend as K


# Определение функции для вычисления R-squared
def r_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def create_dataset(dataset, time_steps=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_steps - 1):
        a = dataset[i:(i + time_steps), :]
        dataX.append(a)
        dataY.append(dataset[i + time_steps, -1])  # Assuming 'Цена' is the last column
    return np.array(dataX), np.array(dataY)


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
# Запуск происходит командой
# nnictl create --config nni-tensorflow/config.yaml

# Загрузите данные


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


def load_data():
    return X_train, X_test, y_train, y_test


def get_default_parameters():
    '''Получите параметры по умолчанию'''
    params = {'units1': 128,
              'units2': 128,
              'units3': 128,
              'batch_size': 128,
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
              'time_steps': 64
              }
    return params


def build_model(PARAMS):
    data = pd.read_parquet('../data/data.parquet')
    train_size = int(len(data) * 0.8)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Target'] = scaler.fit_transform(data['Target'].values.reshape(-1, 1))
    time_steps = PARAMS['time_steps']
    X_ltsm, y = create_dataset(data.values, time_steps)

    # Split into train and test sets
    X_train, X_test = X_ltsm[:train_size], X_ltsm[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Создаем оптимизатор с установленным learning rate
    custom_optimizer = Adam(learning_rate=PARAMS['learning_rate'])

    model = Sequential()
    model.add(LSTM(units=PARAMS['units1'], return_sequences=True, input_shape=(X_train.shape[1], 1),
                   kernel_regularizer=l1(PARAMS['l1_rate_1']),
                   activity_regularizer=l2(PARAMS['l2_rate_1'])))
    model.add(Dropout(PARAMS['dropout_rate_1']))
    model.add(LSTM(units=PARAMS['units2'], return_sequences=False, kernel_regularizer=l1(PARAMS['l1_rate_2']),
                   activity_regularizer=l2(PARAMS['l2_rate_2'])))
    model.add(Dropout(PARAMS['dropout_rate_2']))
    model.add(Dense(units=PARAMS['units3'], kernel_regularizer=l1(PARAMS['l1_rate_3']),
                    activity_regularizer=l2(PARAMS['l2_rate_3'])))
    model.add(Dropout(PARAMS['dropout_rate_3']))
    model.add(Dense(units=1))

    model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
    return model, y_train, y_test, X_train, X_test


def run(X_train, X_test, y_train, y_test, model):
    try:
        model.fit(X_train, y_train, epochs=10, batch_size=PARAMS['batch_size'], validation_data=(X_test, y_test),
                  callbacks=[custom_callback, early_stopping_callback])
        # loss = model.evaluate(X_test, y_test, verbose=2)
        predict_y = model.predict(X_test)
        score = r2_score(y_test, predict_y)
        LOG.debug('r2 score: %s', score)
        LOG.debug('Function run')
        nni.report_final_result(score)
    except Exception as e:
        LOG.exception(e)


if __name__ == '__main__':
    try:
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        model, y_train, y_test, X_train, X_test = build_model(PARAMS)
        LOG.debug('Parameters received')
        run(X_train, X_test, y_train, y_test, model)
        LOG.debug('Model run')
    except Exception as exception:
        LOG.exception(exception)
        LOG.debug('Error')
        raise
