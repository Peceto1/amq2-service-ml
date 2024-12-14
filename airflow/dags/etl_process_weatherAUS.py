import datetime

from airflow.decorators import dag, task

markdown_text = """
### ETL Process for Rain in Australia Data

This DAG extracts information from the original CSV file stored in  [kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package?resource=download). 
It preprocesses the data by creating dummy variables and scaling numerical features.
    
After preprocessing, the data is saved back into a S3 bucket as two separate CSV files: one for training and one for 
testing. The split between the training and testing datasets is 70/30 and they are stratified.
"""


default_args = {
    'owner': "Juan Cruz Piñero",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=15),
    'dagrun_timeout': datetime.timedelta(minutes=25)
}


@dag(
    dag_id="process_etl_weatherAUS_data",
    description="ETL process for weather AUS data, separating the dataset into training and testing sets.",
    doc_md=markdown_text,
    tags=["ETL", "Weather AUS"],
    default_args=default_args,
    catchup=False,
)
def process_etl_weatherAUS_data():

    @task.virtualenv(
        task_id="obtain_original_data_weatherAUS",
        requirements=["ucimlrepo==0.0.3",
                      "awswrangler==3.6.0", "pandas"],
        system_site_packages=True
    )
    def get_data():
        """
        Load the raw data from UCI repository
        """
        import awswrangler as wr
        import pandas as pd
        from pathlib import Path
        import os

        # Determine the path to the 'weatherAUS.csv' file
        airflow_dag_path = Path(os.environ.get('AIRFLOW_HOME', '/opt/airflow')) / 'dags'
        file_path = airflow_dag_path / 'weatherAUS.csv'

        # Verify that the file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load the CSV data
        weather_df = pd.read_csv(file_path, on_bad_lines='skip')
        # fetch dataset
        # weather_df = pd.read_csv('weatherAUS.csv', on_bad_lines='skip')

        data_path = "s3://data/raw/weatherAUS.csv"
        dataframe = weather_df

        # target_col = Variable.get("RainTomorrow")


        wr.s3.to_csv(df=dataframe,
                     path=data_path,
                     index=False)


    @task.virtualenv(
        task_id="make_weather_data_cleaning",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True
    )
    def make_feature_encodings():
        """
        Convert categorical variables into numerical variables. drop duplicates and Nan.
        Complete both numerical and categorical when missings are not a lot.
        """
        import json
        import datetime
        import boto3
        import botocore.exceptions
        import mlflow

        import awswrangler as wr
        import numpy as np

        from airflow.models import Variable
        from utils.data_preprocessing import data_cleaning,feature_encodings

        data_original_path = "s3://data/raw/weatherAUS.csv"
        data_end_path = "s3://data/raw/weatherAUS_cleaned.csv"
        dataset = wr.s3.read_csv(data_original_path)

        dataset_cleaned = data_cleaning(dataset)

        dataset_cleaned = feature_encodings(dataset_cleaned)

        wr.s3.to_csv(df=dataset_cleaned,
                     path=data_end_path,
                     index=False)

        # Save information of the dataset
        client = boto3.client('s3')

        data_dict = {}
        try:
            client.head_object(Bucket='data', Key='data_info/data_weather.json')
            result = client.get_object(Bucket='data', Key='data_info/data_weather.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] != "404":
                # Something else has gone wrong.
                raise e

        target_col = Variable.get("target_col_weather")
        dataset_log = dataset.drop(columns=target_col)
        dataset_with_dummies_log = dataset_cleaned.drop(columns=target_col)

        # Upload JSON String to an S3 Object
        data_dict['columns'] = dataset_log.columns.to_list()
        data_dict['columns_after_cleaning'] = dataset_with_dummies_log.columns.to_list()
        data_dict['target_col'] = target_col

        categories_list = ['Location','WindDir3pm','WindDir9am','WindGustDir']


        data_dict['categorical_columns'] = categories_list
        data_dict['columns_dtypes'] = {k: str(v) for k, v in dataset_log.dtypes.to_dict().items()}
        data_dict['columns_dtypes_after_dummy'] = {k: str(v) for k, v in dataset_with_dummies_log.dtypes
                                                                                                 .to_dict()
                                                                                                 .items()}

        category_dummies_dict = {}
        for category in categories_list:
            category_dummies_dict[category] = np.sort(dataset_log[category].unique()).tolist()

        data_dict['categories_values_per_categorical'] = category_dummies_dict

        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key='data_info/data_weather.json',
            Body=data_string
        )

        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Weather in AUS")

        mlflow.start_run(run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                         experiment_id=experiment.experiment_id,
                         tags={"experiment": "etl", "dataset": "Weather AUS"},
                         log_system_metrics=True)

        mlflow_dataset = mlflow.data.from_pandas(dataset,
                                                 source="https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package",
                                                 targets=target_col,
                                                 name="weather_data_complete")
        mlflow_dataset_cleaned = mlflow.data.from_pandas(dataset_cleaned,
                                                         source="https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package",
                                                         targets=target_col,
                                                         name="weather_data_complete_with_encoded")
        mlflow.log_input(mlflow_dataset, context="Dataset")
        mlflow.log_input(mlflow_dataset_cleaned, context="Dataset")

    @task.virtualenv(
        task_id="split_dataset",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn==1.3.2"],
        system_site_packages=True
    )
    def split_dataset():
        """
        Generate a dataset split into a training part and a test part
        """
        import awswrangler as wr
        from sklearn.model_selection import train_test_split
        from airflow.models import Variable

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)

        data_original_path = "s3://data/raw/weatherAUS_cleaned.csv"
        dataset = wr.s3.read_csv(data_original_path)

        test_size = Variable.get("test_size_weather")
        target_col = Variable.get("target_col_weather")

        X = dataset.drop(columns=target_col)
        y = dataset[[target_col]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

        # Clean duplicates
        dataset.drop_duplicates(inplace=True, ignore_index=True)

        save_to_csv(X_train, "s3://data/final/train/weather_X_train.csv")
        print('saved X:train')
        save_to_csv(X_test, "s3://data/final/test/weather_X_test.csv")
        print('saved X:test')

        save_to_csv(y_train, "s3://data/final/train/weather_y_train.csv")
        print('saved y:train')

        save_to_csv(y_test, "s3://data/final/test/weather_y_test.csv")
        print('saved Y:test')


    @task.virtualenv(
        task_id="normalize_numerical_weather_features",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn==1.3.2",
                      "mlflow==2.10.2"],
        system_site_packages=True
    )
    def normalize_data():
        """
        Standardization of numerical columns
        """
        import json
        import mlflow
        import boto3
        import botocore.exceptions

        import awswrangler as wr
        import pandas as pd

        from sklearn.preprocessing import StandardScaler

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)

        print('reading inputs to normlize')
        X_train = wr.s3.read_csv("s3://data/final/train/weather_X_train.csv")
        X_test = wr.s3.read_csv("s3://data/final/test/weather_X_test.csv")

        sc_X = StandardScaler(with_mean=True, with_std=True)
        X_train_arr = sc_X.fit_transform(X_train)
        X_test_arr = sc_X.transform(X_test)

        X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_arr, columns=X_test.columns)

        print('normalized')

        save_to_csv(X_train, "s3://data/final/train/weather_X_train.csv")
        print('saved normalizerd x_train')
        save_to_csv(X_test, "s3://data/final/test/weather_X_test.csv")
        print('saved normalizerd x_test')

        # Save information of the dataset
        client = boto3.client('s3')

        try:
            client.head_object(Bucket='data', Key='data_info/data_weather.json')
            result = client.get_object(Bucket='data', Key='data_info/data_weather.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
                # Something else has gone wrong.
                raise e

        # Upload JSON String to an S3 Object
        data_dict['standard_scaler_mean'] = sc_X.mean_.tolist()
        data_dict['standard_scaler_std'] = sc_X.scale_.tolist()
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key='data_info/data_weather.json',
            Body=data_string
        )

        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Weather in AUS")

        # Obtain the last experiment run_id to log the new information
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):

            mlflow.log_param("Train observations", X_train.shape[0])
            mlflow.log_param("Test observations", X_test.shape[0])
            mlflow.log_param("Standard Scaler feature names", sc_X.feature_names_in_)
            mlflow.log_param("Standard Scaler mean values", sc_X.mean_)
            mlflow.log_param("Standard Scaler scale values", sc_X.scale_)


    get_data() >> make_feature_encodings() >> split_dataset() >> normalize_data()


dag = process_etl_weatherAUS_data()