import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from typing_extensions import Annotated
from utils.data_preprocessing import data_cleaning,feature_encodings


def load_model(model_name: str, alias: str):
    """
    Load a trained model and associated data dictionary.

    This function attempts to load a trained model specified by its name and alias. If the model is not found in the
    MLflow registry, it loads the default model from a file. Additionally, it loads information about the ETL pipeline
    from an S3 bucket. If the data dictionary is not found in the S3 bucket, it loads it from a local file.

    :param model_name: The name of the model.
    :param alias: The alias of the model version.
    :return: A tuple containing the loaded model, its version, and the data dictionary.
    """

    try:
        # Load the trained model from MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except:
        # If there is no registry in MLflow, open the default model
        file_ml = open('/app/files/model.pkl', 'rb')
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0

    try:
        # Load information of the ETL pipeline from S3
        s3 = boto3.client('s3')

        s3.head_object(Bucket='data', Key='data_info/data.json')
        result_s3 = s3.get_object(Bucket='data', Key='data_info/data.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

        data_dictionary["standard_scaler_mean"] = np.array(data_dictionary["standard_scaler_mean"])
        data_dictionary["standard_scaler_std"] = np.array(data_dictionary["standard_scaler_std"])
    except:
        # If data dictionary is not found in S3, load it from local file
        file_s3 = open('/app/files/data.json', 'r')
        data_dictionary = json.load(file_s3)
        file_s3.close()

    return model_ml, version_model_ml, data_dictionary


def check_model():
    """
    Check for updates in the model and update if necessary.

    The function checks the model registry to see if the version of the champion model has changed. If the version
    has changed, it updates the model and the data dictionary accordingly.

    :return: None
    """

    global model
    global data_dict
    global version_model

    try:
        model_name = "weather_model_prod"
        alias = "champion"

        mlflow.set_tracking_uri('http://mlflow:5000')
        client = mlflow.MlflowClient()

        # Check in the model registry if the version of the champion has changed
        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        # If the versions are not the same
        if new_version_model != version_model:
            # Load the new model and update version and data dictionary
            model, version_model, data_dict = load_model(model_name, alias)

    except:
        # If an error occurs during the process, pass silently
        pass



from pydantic import BaseModel, Field
from typing import Literal

class ModelInput(BaseModel):
    """
    Input schema for the Rain in Australia prediction model.

    This class defines the input fields required for the prediction model based on the dataset attributes, including
    descriptions and validation constraints.

    :param MinTemp: Minimum temperature (in degrees Celsius).
    :param MaxTemp: Maximum temperature (in degrees Celsius).
    :param Rainfall: Rainfall amount (in mm).
    :param WindGustSpeed: Speed of the strongest wind gust (in km/h).
    :param WindSpeed9am: Wind speed at 9am (in km/h).
    :param WindSpeed3pm: Wind speed at 3pm (in km/h).
    :param Humidity9am: Humidity at 9am (in percentage).
    :param Humidity3pm: Humidity at 3pm (in percentage).
    :param Pressure3pm: Atmospheric pressure at 3pm (in hPa).
    :param RainToday: Indicates if it rained today. 1: Yes; 0: No.
    :param latitude: Latitude of the location.
    :param longitude: Longitude of the location.
    :param WindGustDir: Direction of the strongest wind gust (categorical).
    :param WindDir9am: Wind direction at 9am (categorical).
    :param WindDir3pm: Wind direction at 3pm (categorical).
    :param Date: Date in the format yyyy-mm-dd.
    """

    MinTemp: float = Field(
        description="Minimum temperature (in degrees Celsius)",
        ge=-50.0,  # Adjust based on dataset range
        le=50.0
    )
    MaxTemp: float = Field(
        description="Maximum temperature (in degrees Celsius)",
        ge=-50.0,
        le=60.0
    )
    Rainfall: float = Field(
        description="Rainfall amount (in mm)",
        ge=0.0,
        le=500.0
    )
    WindGustSpeed: float = Field(
        description="Speed of the strongest wind gust (in km/h)",
        ge=0.0,
        le=150.0
    )
    WindSpeed9am: float = Field(
        description="Wind speed at 9am (in km/h)",
        ge=0.0,
        le=100.0
    )
    WindSpeed3pm: float = Field(
        description="Wind speed at 3pm (in km/h)",
        ge=0.0,
        le=100.0
    )
    Humidity9am: float = Field(
        description="Humidity at 9am (in percentage)",
        ge=0.0,
        le=100.0
    )
    Humidity3pm: float = Field(
        description="Humidity at 3pm (in percentage)",
        ge=0.0,
        le=100.0
    )
    Pressure3pm: float = Field(
        description="Atmospheric pressure at 3pm (in hPa)",
        ge=800.0,
        le=1100.0
    )
    RainToday: int = Field(
        description="Indicates if it rained today. 1: Yes; 0: No",
        ge=0,
        le=1
    )
    latitude: float = Field(
        description="Latitude of the location",
        ge=-90.0,
        le=90.0
    )
    longitude: float = Field(
        description="Longitude of the location",
        ge=-180.0,
        le=180.0
    )
    WindGustDir: Literal[
        'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'
    ] = Field(
        description="Direction of the strongest wind gust (categorical)"
    )
    WindDir9am: Literal[
        'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'
    ] = Field(
        description="Wind direction at 9am (categorical)"
    )
    WindDir3pm: Literal[
        'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'
    ] = Field(
        description="Wind direction at 3pm (categorical)"
    )
    Date: str = Field(
        description="Date in the format yyyy-mm-dd",
        pattern="^\\d{4}-\\d{2}-\\d{2}$"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "MinTemp": 12.3,
                    "MaxTemp": 25.6,
                    "Rainfall": 0.0,
                    "WindGustSpeed": 45.0,
                    "WindSpeed9am": 20.0,
                    "WindSpeed3pm": 25.0,
                    "Humidity9am": 75.0,
                    "Humidity3pm": 60.0,
                    "Pressure3pm": 1012.0,
                    "RainToday": 0,
                    "latitude": -33.8688,
                    "longitude": 151.2093,
                    "WindGustDir": "NE",
                    "WindDir9am": "N",
                    "WindDir3pm": "NW",
                    "Date": "2023-05-12"
                }
            ]
        }
    }



class ModelOutput(BaseModel):
    """
    Output schema for the Weather in australio prediction model.

    This class defines the output fields returned by the weather prediction model along with their descriptions
    and possible values.

    :param int_output: Output of the model. True if it will rain tomorrow.
    :param str_output: Output of the model in string form. Can be "It will Rain" or "It will not Rain".
    """

    int_output: bool = Field(
        description="Output of the model.  True if it will rain tomorrow",
    )
    str_output: Literal["It will Rain", "It will not Rain"] = Field(
        description="Output of the model in string form",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "int_output": True,
                    "str_output": "It will Rain",
                }
            ]
        }
    }


# Load the model before start
model, version_model, data_dict = load_model("weather_model_prod", "champion")

app_weather = FastAPI()


@app_weather.get("/")
async def read_root():
    """
    Root endpoint of the Rain in Australia Detector API.

    This endpoint returns a JSON response with a welcome message to indicate that the API is running.
    """
    return JSONResponse(content=jsonable_encoder({"message": "Welcome to the Rain In Australia API"}))


@app_weather.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks
):
    """
    Endpoint for predicting if it will rain.

    This endpoint receives features related to a weather variables on a certain date and predicts whether the it will rain in
    that location the next day or not using a trained model. It returns the prediction result in both integer and string formats.
    """

    # Extract features from the request and convert them into a list and dictionary
    features_list = [*features.dict().values()]
    features_key = [*features.dict().keys()]

    # Convert features into a pandas DataFrame
    features_df = pd.DataFrame(np.array(features_list).reshape([1, -1]), columns=features_key)


    features_df = feature_encodings(features_df)

    features_df = (features_df-data_dict["standard_scaler_mean"])/data_dict["standard_scaler_std"]


    # Make the prediction using the trained model
    prediction = model.predict(features_df)

    # Convert prediction result into string format
    str_pred = "It Will not Rain"
    if prediction[0] > 0:
        str_pred = "It Will Rain"

    # Check if the model has changed asynchronously
    background_tasks.add_task(check_model)

    # Return the prediction result
    return ModelOutput(int_output=bool(prediction[0].item()), str_output=str_pred)
