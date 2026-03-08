import pandas as pd
import mlflow
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

mlflow.xgboost.autolog()

mlflow.set_experiment("Loan Approval Prediction")

# load data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
y_test = pd.read_csv("data/y_test.csv")

with mlflow.start_run(run_name="XGBoost Model"):

    model = xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train.values.ravel())

    preds = model.predict(X_test)

    mlflow.log_metric("precision", precision_score(y_test, preds))
    mlflow.log_metric("recall", recall_score(y_test, preds))
    mlflow.log_metric("f1_score", f1_score(y_test, preds))