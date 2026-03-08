import pandas as pd
import mlflow
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load processed data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
y_test = pd.read_csv("data/y_test.csv")

mlflow.set_experiment("Loan Approval Prediction")

with mlflow.start_run(run_name="Logistic Regression Baseline"):

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train.values.ravel())

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)

    joblib.dump(model, "model.joblib")
    mlflow.log_artifact("model.joblib")

print("Training complete.")