import pandas as pd
from sklearn.model_selection import train_test_split

from .utilities import getRainTodayValue
from .geocoding import generate_lat_long_features_from_location
from .CustomEncoders import CyclicalCompassEncoder,DateCyclicEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

def data_cleaning(weather_df):

    weather_df = complete_categorical(weather_df)

    weather_df = complete_numerical(weather_df)


    columns_to_check = ['MinTemp', 'MaxTemp', 'Humidity9am', 'Humidity3pm','WindGustSpeed']

    weather_df = weather_df.dropna(subset=columns_to_check)

    _,weather_df = generate_lat_long_features_from_location(weather_df)

    weather_df.drop(columns=['Location'], inplace=True)

    weather_df.drop(columns=['Sunshine','Evaporation','Cloud9am','Cloud3pm'],inplace=True)

    weather_df.drop(columns=['Pressure9am', 'Temp9am', 'Temp3pm'], inplace=True)

    # elimino

    print('dataframe final')
    weather_df.head()
    return weather_df



def complete_categorical(weather_df):
    weather_df[['WindDir9am', 'WindDir3pm', 'WindGustDir']] = weather_df[
        ['WindDir9am', 'WindDir3pm', 'WindGustDir']].fillna('M')
    return weather_df


def complete_numerical(weather_df):

    weather_df['RainToday'] = weather_df.apply(lambda row: getRainTodayValue(row), axis=1)
    weather_df['Pressure3pm'] = weather_df['Pressure3pm'].fillna(weather_df['Pressure3pm'].mean())
    weather_df.loc[(weather_df['RainToday'] == 'No') & (weather_df['Rainfall'].isna()), 'Rainfall'] = 0
    # Eliminamos valores que hayan quedado nulos despues de la conversión

    weather_df = weather_df.dropna(subset=['RainToday', 'RainTomorrow'])
    return weather_df


def feature_encodings(weather_df_to_train):
    # paso yes/no a numericas
    weather_df_to_train['RainToday'] = weather_df_to_train['RainToday'].apply(
        lambda x: 1 if x == 'Yes' else (0 if x == 'No' else x))

    weather_df_to_train['RainTomorrow'] = weather_df_to_train['RainTomorrow'].apply(
        lambda x: 1 if x == 'Yes' else (0 if x == 'No' else x))

    cyclical_compass_encoder = CyclicalCompassEncoder(col='WindGustDir')
    weather_df_to_train = cyclical_compass_encoder.fit_transform(weather_df_to_train)

    cyclical_compass_encoder = CyclicalCompassEncoder(col='WindDir9am')
    weather_df_to_train = cyclical_compass_encoder.fit_transform(weather_df_to_train)

    cyclical_compass_encoder = CyclicalCompassEncoder(col='WindDir3pm')
    weather_df_to_train = cyclical_compass_encoder.fit_transform(weather_df_to_train)

    weather_df_to_train['Date'] = pd.to_datetime(weather_df_to_train['Date'])

    date_encoder = DateCyclicEncoder(column_name='Date')

    weather_df_to_train = date_encoder.fit_transform(weather_df_to_train)


    weather_df_to_train.dropna(inplace=True,axis=0,how='any')
    print('dataframe final with encoded features')
    weather_df_to_train.head()
    return weather_df_to_train


def custom_train_test_split(weather_df_to_train,test_ratio,validation_ratio):
    # Separating features and target
    features = weather_df_to_train.drop(columns=['RainTomorrow'])
    target = weather_df_to_train['RainTomorrow']

    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=(validation_ratio + test_ratio),
                                                        random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                    test_size=test_ratio / (validation_ratio + test_ratio),
                                                    random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_input_data(X_train,X_test,X_val):
    # Define the scaler
    scaler = StandardScaler()

    # Apply scaling to the numerical features (specified manually)
    num_features = ['MinTemp', 'MaxTemp', 'Humidity3pm', 'Humidity9am', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
                    'Pressure3pm', 'latitude', 'longitude']

    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_val[num_features] = scaler.transform(X_val[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])

    return X_train, X_test , X_val





