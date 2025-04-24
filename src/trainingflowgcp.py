# src/trainingflowgcp.py

import os
from metaflow import (
    FlowSpec, step, Parameter,
    conda_base, kubernetes, resources,
    retry, timeout, catch
)
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

@conda_base(
    python="3.9.16",
    libraries={
        "scikit-learn":   "1.2.2",
        "mlflow":         "2.4.0",
        "databricks-cli": "0.17.6"
    }
)
class TrainingFlow(FlowSpec):

    seed = Parameter(
        "seed", default=42, type=int,
        help="Random seed for reproducibility"
    )
    n_folds = Parameter(
        "folds", default=5, type=int,
        help="Number of CV folds"
    )
    max_trees = Parameter(
        "max_trees", default=100, type=int,
        help="Number of trees in the Random Forest"
    )

    @step
    def start(self):
        # load X,y in one go to satisfy Pylint
        self.X, self.y = load_iris(return_X_y=True)
        print(f"Iris loaded: X.shape={self.X.shape}, y.shape={self.y.shape}")
        self.next(self.featurize)

    @step
    def featurize(self):
        # placeholder for FE
        self.next(self.tune)

    @kubernetes
    @resources(cpu=2, memory=8192)
    @retry(times=3, minutes_between_retries=1)
    @timeout(seconds=1800)
    @catch(var="error", print_exception=True)
    @step
    def tune(self):
        """Grid-search the number of trees."""
        rf = RandomForestClassifier(random_state=self.seed)
        cv = GridSearchCV(
            rf,
            {"n_estimators": [self.max_trees]},
            cv=self.n_folds
        )
        cv.fit(self.X, self.y)
        self.best_model = cv.best_estimator_
        print("Best params:", cv.best_params_)
        self.next(self.register)

    @step
    def register(self):
        """Log & register the model in MLflow."""
        import mlflow, mlflow.sklearn

        mlflow.set_tracking_uri("http://127.0.0.1:5001")
        mlflow.set_experiment('iris‑trainingflow')
        with mlflow.start_run():
            mlflow.sklearn.log_model(
                self.best_model,
                artifact_path="model",
                registered_model_name="best-iris-model"
            )
        print("Registered model as 'best-iris-model'")
        self.next(self.end)

    @step
    def end(self):
        print("🎉  TrainingFlow complete!")

if __name__ == "__main__":
    TrainingFlow()