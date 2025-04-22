from metaflow import FlowSpec, step, Parameter, Flow
import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5001")

class ScoringFlow(FlowSpec):
    input_path = Parameter('input', required=True, help='Path to CSV of new data')

    @step
    def start(self):
        self.df = pd.read_csv(self.input_path)
        self.next(self.featurize)

    @step
    def featurize(self):
        self.X_new = self.df
        self.next(self.load_model)

    @step
    def load_model(self):
        client = mlflow.tracking.MlflowClient()
        model_info = client.get_latest_versions('best-iris-model', stages=['None'])[0]
        self.model = mlflow.sklearn.load_model(model_info.source)
        self.next(self.predict)

    @step
    def predict(self):
        self.df['prediction'] = self.model.predict(self.X_new)
        self.df.to_csv('predictions.csv', index=False)
        self.next(self.end)

    @step
    def end(self):
        print("Scoring complete. Predictions saved to predictions.csv")

if __name__ == '__main__':
    ScoringFlow()