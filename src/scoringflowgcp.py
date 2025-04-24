from metaflow import FlowSpec, step, Parameter, conda_base, kubernetes, retry, timeout, catch
import pandas as pd
import mlflow

# point at your MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5001")

@conda_base(
    python='3.9.16',
    libraries={
        'numpy': '1.23.5',
        'scikit-learn': '1.2.2'
    }
)
class ScoringFlow(FlowSpec):
    """
    A Metaflow scoring flow that
    - reads new data from CSV
    - featurizes it
    - loads the latest registered model from MLflow
    - makes predictions in a Kubernetes-powered step
    - writes out a predictions.csv
    """

    input_path = Parameter(
        'input',
        help='Path to CSV of new data',
        required=True
    )

    @step
    def start(self):
        # Load raw data
        self.df = pd.read_csv(self.input_path)
        self.next(self.featurize)

    @step
    def featurize(self):
        # Here you could add additional feature engineering
        self.X_new = self.df.copy()
        self.next(self.load_model)

    @retry(times=3, minutes_between_retries=1)
    @timeout(seconds=300)
    @catch(var='err', print_exception=True)
    @step
    def load_model(self):
        # Fetch the latest model registered under 'best-iris-model'
        client = mlflow.tracking.MlflowClient()
        model_info = client.get_latest_versions('best-iris-model', stages=['None'])[0]
        # Load it into memory
        self.model = mlflow.sklearn.load_model(model_info.source)
        self.next(self.predict)

    @kubernetes(cpu=1, memory=4096, disk=10240)
    @step
    def predict(self):
        # Run inference on Kubernetes node
        self.df['prediction'] = self.model.predict(self.X_new)
        # Persist the output
        self.df.to_csv('predictions.csv', index=False)
        self.next(self.end)

    @step
    def end(self):
        print("Scoring complete. Predictions saved to predictions.csv")

if __name__ == '__main__':
    ScoringFlow()
