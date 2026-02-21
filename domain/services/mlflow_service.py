import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
import os


# Set the tracking URI for MLFlow
mlflow.set_tracking_uri(os.getenv('MLFLOW_URI'))

class MLFlowService:
    def set_experiment(experiment_name):
        """
        Sets the experiment name in MLFlow
        :param experiment_name: the name of the experiment
        """
        mlflow.set_experiment(experiment_name)


    def start_run(run_name):
        """
        Starts a new run in MLFlow
        :param run_name: the name of the run
        """
        return mlflow.start_run(run_name=run_name)


    def end_run_if_activated():
        """
        Ends the MLFlow run if it is active
        """
        if mlflow.active_run():
            mlflow.end_run()


    def end_run(status='FINISHED'):
        """
        Ends the MLFlow run
        :param status: the finish status of the run
        """
        mlflow.end_run(status=status)


    def log_pyfunc_model(artifact_path, python_model):
        """
        Logs a Python function model to MLFlow.
        :param artifact_path: the path to save the model
        :param python_model: the Python model to be saved
        """
        mlflow.pyfunc.log_model(artifact_path=artifact_path, python_model=python_model)


    def log_sklearn_model(artifact_path, model):
        """
        Logs a sklearn model to MLFlow
        :param artifact_path: the path to save the model
        :param model: the sklearn model to be saved
        """
        mlflow.sklearn.log_model(model, artifact_path)


    def log_params(key, value):
        """
        Logs a parameter to MLFlow
        :param key: the key of the parameter
        :param value: the value of the parameter
        """
        mlflow.log_param(key, value)


    def log_metric(key, value):
        """
        Logs a metric to MLFlow
        :param key: the key of the metric
        :param value: the value of the metric
        """
        mlflow.log_metric(key, value)


    def load_model_live(model_name):
        """
        Loads a model from MLFlow
        :param model_name: model name
        :return: the loaded model
        """
        model_uri = f"models:/{model_name}@live"
        model = mlflow.sklearn.load_model(model_uri)
        return model


    def load_pyfunc_model_live(model_name):
        """
        Loads a PyFunc model from MLFlow that's in the live stage
        :param model_name: the name of the model
        :return: the loaded model
        """
        model_uri = f"models:/{model_name}@live"
        model = mlflow.pyfunc.load_model(model_uri)
        return model


    def get_model_info(model_name):
        """
        Gets the model information from MLFlow
        :param model_name: the name of model
        :return: the model information
        """
        # Get model version information
        client = MlflowClient()
        return client.get_registered_model(model_name)

    def log_metrics(history):
        """
            Logs the metrics from the history object to MLFlow
            :param history: The history object returned by the model.fit method
        """
        for epoch in range(len(history.epoch)):
            for metric, values in history.history.items():
                # Logs each metric per epoch
                mlflow.log_metric(metric, values[epoch], step=epoch)

    
