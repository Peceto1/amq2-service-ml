import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np

class DateCyclicEncoder():
    def __init__(self, column_name):
        """
        Initialize the encoder with the date column name to be encoded.
        """
        self.column_name = column_name

    def fit(self, X, y=None):
        """
        Fit method (not needed for this encoder as it's stateless).
        """
        return self

    def transform(self, X):
        """
        Transform the date column in X into cyclical features (sin/cos) for day of the year.
        Retains other columns in the DataFrame.
        """
        X = X.copy()

        # Extract day of year
        X['DayOfYear'] = X[self.column_name].dt.dayofyear

        # Apply cyclical encoding (sine and cosine transformations) for day of year
        X['DayOfYear_sin'] = np.sin(2 * np.pi * X['DayOfYear'] / 365)
        X['DayOfYear_cos'] = np.cos(2 * np.pi * X['DayOfYear'] / 365)

        # Drop the original date and intermediate 'DayOfYear' columns
        X = X.drop(columns=[self.column_name, 'DayOfYear'])

        return X

    def fit_transform(self, X, y=None):
        """
        Fit and transform the data in a single step.
        """
        return self.fit(X).transform(X)


class CyclicalCompassEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def wind_direction_to_degrees(direction):
            compass_to_degrees = {
                'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
                'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
                'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
                'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
            }

            if direction in compass_to_degrees:
                return compass_to_degrees[direction]
            else:
                return np.nan  # Return NaN if direction is invalid
        X = X.copy()
        # Map compass directions to degrees
        X[self.col + '_degrees'] = X[self.col].map(wind_direction_to_degrees)
        # Convert to radians
        X[self.col + '_radians'] = np.deg2rad(X[self.col + '_degrees'])
        # Apply cyclical encoding
        X[self.col + '_sin'] = np.sin(X[self.col + '_radians'])
        X[self.col + '_cos'] = np.cos(X[self.col + '_radians'])
        return X.drop(columns=[self.col, self.col + '_degrees', self.col + '_radians'])
# Example usage
# df = pd.DataFrame({'wind_direction': ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']})

# cyclical_compass_encoder = CyclicalCompassEncoder(col='wind_direction')
# df_encoded = cyclical_compass_encoder.fit_transform(df)

# print(df_encoded)