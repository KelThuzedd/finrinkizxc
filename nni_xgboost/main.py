import nni
import pandas as pd
import xgboost as xgb
import logging
from sklearn.metrics import r2_score

# Загрузите данные
# Запуск происходит командой
# nnictl create --config nni-automl/config.yaml
# последний классных экспериметр id=wj7vlhyx

X_train = pd.read_parquet('../data/X_train.parquet')
X_test = pd.read_parquet('../data/X_test.parquet')
y_train = pd.read_parquet('../data/y_train.parquet')
y_test = pd.read_parquet('../data/y_test.parquet')

LOG = logging.getLogger('xgboost_regression')
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
    params = {'max_depth': 3}  # Начальное значение глубины дерева
    return params


def get_model(PARAMS):
    '''Получите модель XGBoost с параметрами из NNI'''
    model = xgb.XGBRegressor(
        max_depth=PARAMS['max_depth'],
        n_estimators=PARAMS['n_estimators'],
        learning_rate=PARAMS['learning_rate'],
        subsample=PARAMS['subsample'],
        gamma=PARAMS['gamma'],  # Добавленный гиперпараметр
        min_child_weight=PARAMS['min_child_weight'],  # Добавленный гиперпараметр
        max_delta_step=PARAMS['max_delta_step'],  # Добавленный гиперпараметр
        colsample_bytree=PARAMS['colsample_bytree'],  # Добавленный гиперпараметр
        reg_alpha=PARAMS['reg_alpha'],  # Добавленный гиперпараметр
        reg_lambda=PARAMS['reg_lambda'],  # Добавленный гиперпараметр
        scale_pos_weight=PARAMS['scale_pos_weight'],  # Добавленный гиперпараметр
        objective='reg:squarederror',
        tree_method='hist',
        device='cuda'
    )

    return model


def run(X_train, X_test, y_train, y_test, model):
    '''Обучите модель и выполните предсказание'''
    try:
        model.fit(X_train, y_train)
        predict_y = model.predict(X_test)
        score = r2_score(y_test, predict_y)
        LOG.debug('r2 score: %s', score)
        LOG.debug('Функция run')
        nni.report_final_result(score)
    except Exception as e:
        # Запишите исключение в лог
        LOG.exception(e)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()

    try:
        # Получите параметры из tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        model = get_model(PARAMS)
        LOG.debug('Параметры получены')
        run(X_train, X_test, y_train, y_test, model)
        LOG.debug('Модель запущена')
    except Exception as exception:
        LOG.exception(exception)
        LOG.debug('Ошибка')
        raise

# %%
