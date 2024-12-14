import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_regression


def getValuesCountForColumn(df,column):
    return df[column].unique()

def getUniqueValuesForColumn(df,column):
    return df[column].unique()

def getNullCountForColumn(df, column):
    return df[column].isnull().sum()

def getNullPercentageForColumn(df,column):
    print(f'Null percentage for {column} is {(df[column].isnull().sum() / len(df)) * 100}%')

def showUniqueValuesForColumn(df,column):
    print(f'Unique values for {column} column:')
    for value in getUniqueValuesForColumn(df,column):
        print(value)

def showNullValueCountForColumn(df,column):
    print(f'Null values count for {column} column:')
    print(getNullCountForColumn(df,column))

def getRainTodayValue(row):
    return 'Yes' if row['Rainfall'] > 1 else 'No' if pd.isnull(row['RainToday']) else row['RainToday']


def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_variance(pca, width=8, dpi=100):
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    fig.set(figwidth=8, dpi=100)
    return axs

import csv
from geopy.exc import GeocoderTimedOut
import time
# Function to geocode place names


def load_geocode_results(filename='geocode_results.csv'):
    """
    Loads geocode results from a CSV file.

    Args:
        filename (str, optional): The name of the CSV file. Defaults to 'geocode_results.csv'.

    Returns:
        dict: The loaded geocode results.
    """

    geocode_results_loaded = {}
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            geocode_results_loaded[row['place']] = {'latitude': float(row['latitude']) if row['latitude'] else None,
                                                   'longitude': float(row['longitude']) if row['longitude'] else None}
    print(f"Geocode results loaded from {filename}")
    return geocode_results_loaded




import re

def split_uppercase_word(text):
    # Use a regular expression to find occurrences where a lowercase letter is followed by an uppercase letter
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)


def save_geocode_results(geocode_results, filename='geocode_results.csv'):
    """
    Saves geocode results to a CSV file.

    Args:
        geocode_results (dict): A dictionary containing geocode results.
        filename (str, optional): The name of the CSV file. Defaults to 'geocode_results.csv'.
    """

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['place', 'latitude', 'longitude']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for place, coordinates in geocode_results.items():
            writer.writerow({'place': place, 'latitude': coordinates['latitude'], 'longitude': coordinates['longitude']})
    print(f"Geocode results saved to {filename}")
    return filename


def getRainTodayValue(row):
    return 'Yes' if row['Rainfall'] > 1 else 'No' if pd.isnull(row['RainToday']) else row['RainToday']