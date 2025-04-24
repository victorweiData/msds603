# src/trainingflow.py

from metaflow import FlowSpec, step, Parameter
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class TrainingFlow(FlowSpec):

    # command‐line parameters
    seed      = Parameter('seed',
                          default=42,
                          type=int,
                          help='Random seed for reproducibility')
    n_folds   = Parameter('folds',
                          default=5,
                          type=int,
                          help='Number of cross‑validation folds')
    max_trees = Parameter('max_trees',
                          default=100,
                          type=int,
                          help='Number of trees in the Random Forest')

    @step
    def start(self):
        """Load the Iris dataset into memory."""
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        print(f"Iris loaded: X.shape={self.X.shape}, y.shape={self.y.shape}")
        self.next(self.featurize)

    @step
    def featurize(self):
        """Placeholder for any feature transforms."""
        # e.g. self.X = some_transform(self.X)
        self.next(self.tune)

    @step
    def tune(self):
        """Run a simple grid search over n_estimators."""
        rf   = RandomForestClassifier(random_state=self.seed)
        grid = {'n_estimators': [self.max_trees]}
        cv   = GridSearchCV(rf, grid, cv=self.n_folds)
        cv.fit(self.X, self.y)
        self.best_model = cv.best_estimator_
        print("Best params:", cv.best_params_)
        self.next(self.register)

    @step
    def register(self):
        """Log and register the best model in MLflow."""
        mlflow.set_tracking_uri("http://127.0.0.1:5001")
        mlflow.set_experiment('iris‑trainingflow')
        with mlflow.start_run():
            mlflow.sklearn.log_model(
                self.best_model,
                artifact_path='model',
                registered_model_name='best-iris-model'
            )
        print("Registered model as 'best-iris-model'")
        self.next(self.end)

    @step
    def end(self):
        """Final step."""
        print("TrainingFlow complete!")

if __name__ == '__main__':
    TrainingFlow()