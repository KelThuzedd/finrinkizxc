import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np


def plot_combined_closing_prices(file_names, column_name):
    # Create a figure to hold the plot
    plt.figure(figsize=(12, 6))

    # Iterate over the file names
    for i, file_name in enumerate(file_names):
        # Construct the file path
        file_path = '../data_updated/' + file_name

        # Read the file into a DataFrame
        dataset = pd.read_csv(file_path)
        dataset['Цена'] = dataset['Цена'] / dataset['Цена'].iloc[0]  # Normalize the prices

        # Preprocess the data
        dataset['Дата'] = pd.to_datetime(dataset['Дата'])
        # Plot the closing prices
        plt.plot(dataset['Дата'], dataset[column_name],
                 label=f'{column_name} для акции (нормализованный) "{file_name}"')

    # Set the title, labels, and legend
    plt.title(f'{column_name} для всех акций (нормализованный)')
    plt.xlabel('Дата')
    plt.ylabel(f'{column_name}')
    plt.legend()

    # Show the plot
    plt.show()


import seaborn as sns

def plot_closing_prices(file_names, colors, column_name, plot_trend=True, plot_moving_average=True, trend_window_length=15,
                        moving_avg_window=30):
    sns.set_style('whitegrid')

    for i, file_name in enumerate(file_names):
        # Construct the file path
        file_path = '../data_updated/' + file_name

        # Read the file into a DataFrame
        dataset = pd.read_csv(file_path)
        dataset['Дата'] = pd.to_datetime(dataset['Дата'])

        # Plot the closing prices
        plt.figure(figsize=(10, 5))
        plt.grid(True)
        plt.plot(dataset['Дата'], dataset[column_name], label=f'{column_name} для акции {file_name}',
                 color=colors[i % len(colors)])

        if plot_trend:
            # Calculate and plot trend line using Savitzky-Golay filter
            trend = savgol_filter(dataset[column_name], window_length=trend_window_length, polyorder=2)
            plt.plot(dataset['Дата'], trend, label=f'Trend for {column_name}', color='black', linestyle='--')

        if plot_moving_average:
            # Calculate and plot moving average
            moving_avg = dataset[column_name].rolling(window=moving_avg_window).mean()
            plt.plot(dataset['Дата'], moving_avg, label=f'{column_name} {moving_avg_window}-day MA', color='green',
                     linestyle='--')

        plt.title(f'График "{column_name}" для акции "{file_name}"')
        plt.xlabel('Дата')
        plt.ylabel(f'Значение "{column_name}"')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

def calculate_statistics(file_names, columns):
    # Initialize an empty DataFrame to store the statistics
    statistics_df = pd.DataFrame(columns=['Company'] + [f'{col}_mean' for col in columns] + [f'{col}_std_dev' for col in columns] + [f'{col}_min' for col in columns] + [f'{col}_max' for col in columns] + [f'{col}_median' for col in columns])

    for file_name in file_names:
        # Construct the file path
        file_path = '../data_updated/' + file_name

        # Read the file into a DataFrame
        dataset = pd.read_csv(file_path)

        # Create an empty list to store statistics for this file
        statistics = [file_name]

        # Calculate statistics for each column
        for column in columns:
            mean = np.mean(dataset[column])
            std_dev = np.std(dataset[column])
            min_val = np.min(dataset[column])
            max_val = np.max(dataset[column])
            median = np.median(dataset[column])

            statistics.extend([mean, std_dev, min_val, max_val, median])

        # Create a new row DataFrame with the statistics
        row_df = pd.DataFrame([statistics], columns=statistics_df.columns)

        # Concatenate the new row with the existing statistics DataFrame
        statistics_df = pd.concat([statistics_df, row_df], ignore_index=True)

    return statistics_df
#%%
