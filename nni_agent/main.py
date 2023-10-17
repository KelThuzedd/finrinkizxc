import nni
import logging
import numpy as np
import tensorflow as tf
from collections import deque
import random
class Agent:
    def __init__(self, state_size, window_size, trend, skip, batch_size, volume, day_of_week,dayofyear,EMA_20,MACD,Signal_Line):
        self.volume = volume
        self.day_of_week = day_of_week

        self.dayofyear = dayofyear
        self.EMA_20 = EMA_20
        self.MACD = MACD
        self.Signal_Line = Signal_Line


        self.state_size = state_size
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend = trend
        self.skip = skip
        self.action_size = 3
        self.batch_size = batch_size
        self.memory = deque(maxlen=batch_size)
        self.inventory = []

        self.gamma = 0.95
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.model = self.build_model(PARAMS)
        # self.model = self.build_model()

    def build_model(self,PARAMS ):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(PARAMS['units1'], activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(PARAMS['units2'], activation='relu'),
            tf.keras.layers.Dense(PARAMS['units3'], activation='relu'),
            tf.keras.layers.Dense(self.action_size)
        ])
        optimizer_model = tf.keras.optimizers.get(PARAMS['optimizer'])
        if PARAMS['learning_rate'] != 0:
            optimizer_model.learning_rate = PARAMS['learning_rate']
        model.compile(optimizer_model, loss='mean_squared_error')
        return model

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state.reshape(1, self.state_size),verbose = 0)[0])
    def get_state(self, t):
        window_size = self.window_size + 1
        d = t - window_size + 1
        block = self.trend[d : t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0 : t + 1]
        block_volume = self.volume[d : t] if d >= 0 else -d * [self.volume[0]] + self.volume[0 : t]  # adjusted slicing
        block_day_of_week = self.day_of_week[d : t] if d >= 0 else -d * [self.day_of_week[0]] + self.day_of_week[0 : t]  # adjusted slicing

        block_dayofyear = self.dayofyear[d : t] if d >= 0 else -d * [self.dayofyear[0]] + self.dayofyear[0 : t]  # adjusted slicing
        block_EMA_20 = self.EMA_20[d : t] if d >= 0 else -d * [self.EMA_20[0]] + self.EMA_20[0 : t]  # adjusted slicing
        block_MACD = self.MACD[d : t] if d >= 0 else -d * [self.MACD[0]] + self.MACD[0 : t]  # adjusted slicing
        block_Signal_Line= self.Signal_Line[d : t] if d >= 0 else -d * [self.Signal_Line[0]] + self.Signal_Line[0 : t]  # adjusted slicing

        res = np.concatenate((np.diff(block), block_volume, block_day_of_week,block_dayofyear,block_EMA_20,block_MACD,block_Signal_Line), axis=None)
        return np.array([res])
    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        states = np.array([state for state, _, _, _, _ in mini_batch])
        actions = np.array([action for _, action, _, _, _ in mini_batch])
        rewards = np.array([reward for _, _, reward, _, _ in mini_batch])
        next_states = np.array([next_state for _, _, _, next_state, _ in mini_batch])
        dones = np.array([done for _, _, _, _, done in mini_batch])

        targets = self.model.predict(states)
        next_state_targets = self.model.predict(next_states)
        max_next_state_targets = np.max(next_state_targets, axis=1)

        targets[np.arange(batch_size), actions] = rewards + self.gamma * max_next_state_targets * (1 - dones)

        self.model.fit(states, targets, batch_size=batch_size, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def buy(self, initial_money,close):
        starting_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        state = self.get_state(0)
        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state)
            next_state = self.get_state(t + 1)

            if action == 1 and initial_money >= self.trend[t] and t < (len(self.trend) - self.half_window):
                inventory.append(self.trend[t])
                initial_money -= self.trend[t]
                states_buy.append(t)
                print('day %d: buy 1 unit at price %f, total balance %f' % (t, self.trend[t], initial_money))

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                initial_money += self.trend[t]
                states_sell.append(t)
                try:
                    invest = ((close[t] - bought_price) / bought_price) * 100
                except:
                    invest = 0
                print('day %d, sell 1 unit at price %f, investment %f %%, total balance %f,' %
                      (t, close[t], invest, initial_money))

            state = next_state
        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        return states_buy, states_sell, total_gains, invest

    def train(self, iterations, checkpoint, initial_money):
        for i in range(iterations):
            total_profit = 0
            inventory = []
            state = self.get_state(0)
            starting_money = initial_money
            for t in range(0, len(self.trend) - 1, self.skip):
                action = self.act(state)
                next_state = self.get_state(t + 1)

                if action == 1 and starting_money >= self.trend[t] and t < (len(self.trend) - self.half_window):
                    inventory.append(self.trend[t])
                    starting_money -= self.trend[t]

                elif action == 2 and len(inventory) > 0:
                    bought_price = inventory.pop(0)
                    total_profit += self.trend[t] - bought_price
                    starting_money += self.trend[t]

                invest = ((starting_money - initial_money) / initial_money)
                self.memory.append((state, action, invest, next_state, starting_money < initial_money))
                state = next_state
                if len(self.memory) > self.batch_size:
                    self.replay(self.batch_size)
            if (i + 1) % checkpoint == 0:
                print('epoch: %d, total rewards: %f.3, total money: %f' % (i + 1, total_profit, starting_money))




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
    params = {'units1': 8,
              'units2': 8,
              'units3': 8,
              'optimizer': 'Adamax',
              'batch_size': 64,
              'learning_rate': 0.01,
              'window_size': 10,
              'skip': 4,
              'iterations': 20,
              }
    return params
def zxc_main(PARAMS):
    import pandas as pd
    data_lkoh = pd.read_parquet('../data/data_lkoh.parquet')

    head = 1000
    split = 0.5
    initial_money = 10000
    window_size = PARAMS['window_size']
    skip = PARAMS['skip']
    batch_size =  PARAMS['batch_size']
    df = data_lkoh.copy()
    data_100 = df.head(head)
    data_100 = data_100.sort_index(ascending=False)
    data_100 = data_100.reset_index(drop=True)
    # Assuming df is your DataFrame and 'Цена' is your target column
    data = data_100['Цена'].values.tolist()
    volume = data_100.Объём.values.tolist()
    day_of_week = data_100.dayofweek.values.tolist()  # assuming you have a column named 'DayOfWeek'

    dayofyear = data_100.dayofyear.values.tolist()  # assuming you have a column named 'dayofyear'
    EMA_20 = data_100.EMA_20.values.tolist()  # assuming you have a column named 'DayOfWeek'
    MACD = data_100.MACD.values.tolist()  # assuming you have a column named 'DayOfWeek'
    Signal_Line = data_100.Signal_Line.values.tolist()  # assuming you have a column named 'DayOfWeek'
    # Define the size of the training set. For example, 80% for training, 20% for testing.
    train_size = int(len(data) * split)

    # Split the data
    train_data = data[:train_size]
    test_data = data[train_size:]
    # Train the agent with the training data
    agent = Agent(state_size = window_size * 7,  # updated state_size
                  window_size = window_size,
                  trend = train_data,
                  skip = skip,
                  batch_size = batch_size,
                  volume = volume[:train_size],
                  day_of_week = day_of_week[:train_size],
                  dayofyear = dayofyear[:train_size],
                  EMA_20 = EMA_20[:train_size],
                  MACD = MACD[:train_size],
                  Signal_Line = Signal_Line[:train_size],
                  )
    Agent.build_model(agent,PARAMS)

    agent.train(iterations = PARAMS['iterations'], checkpoint = 5, initial_money = initial_money)
    # Update the trend, volume, day_of_week, dayofyear, EMA_20, MACD, and Signal_Line of the agent with the test data
    agent.trend = test_data
    agent.volume = volume[train_size:]
    agent.day_of_week = day_of_week[train_size:]
    agent.dayofyear = dayofyear[train_size:]
    agent.EMA_20 = EMA_20[train_size:]
    agent.MACD = MACD[train_size:]
    agent.Signal_Line = Signal_Line[train_size:]


    close = data_100.Цена.values.tolist()[train_size:]

    states_buy_test, states_sell_test, total_gains_test, invest_test = agent.buy(initial_money = initial_money,close=close)
    nni.report_final_result(total_gains_test)
    print(total_gains_test)
    print(total_gains_test)
    print(total_gains_test)
    print(total_gains_test)

if __name__ == '__main__':
    try:
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        LOG.debug('Parameters received')
        zxc_main(PARAMS)
        LOG.debug('Model run')
    except Exception as exception:
        LOG.exception(exception)
        LOG.debug('Error')
        raise

