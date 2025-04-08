# src/preprocessing.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import StandardScaler
import joblib
import yaml

def main():
    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["features"]
    train_path = params["train_path"]
    test_path = params["test_path"]
    chi2percentile = params["chi2percentile"]

    # Load raw data
    train = pd.read_csv(train_path, header=None)
    test = pd.read_csv(test_path, skiprows=1, header=None)

    # Preprocessing pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectPercentile(chi2, percentile=chi2percentile))
    ])

    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_test = test.iloc[:, :-1]

    X_train_trans = pipeline.fit_transform(X_train, y_train)
    X_test_trans = pipeline.transform(X_test)

    # Save outputs
    pd.DataFrame(X_train_trans).to_csv("data/processed_train_data.csv", index=False)
    pd.DataFrame(X_test_trans).to_csv("data/processed_test_data.csv", index=False)
    joblib.dump(pipeline, "data/pipeline.pkl")

if __name__ == "__main__":
    main()