# lab8app/register_iris.py
import mlflow, mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# point MLflow at your local server
mlflow.set_tracking_uri("http://127.0.0.1:5002")
mlflow.set_experiment("iris-demo")

with mlflow.start_run():
    X,y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        registered_model_name="iris-classifier"
    )
    print(f"Registered iris-classifier with accuracy={acc:.3f}")