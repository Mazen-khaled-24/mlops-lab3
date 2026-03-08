import pandas as pd
import mlflow
import lightgbm as lgb

from sklearn.metrics import precision_score, recall_score, f1_score

mlflow.lightgbm.autolog()

mlflow.set_experiment("Loan Approval Prediction")

# load data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
y_test = pd.read_csv("data/y_test.csv")

with mlflow.start_run(run_name="LightGBM Model"):

    model = lgb.LGBMClassifier(random_state=42)

    model.fit(X_train, y_train.values.ravel())

    preds = model.predict(X_test)

    mlflow.log_metric("precision", precision_score(y_test, preds))
    mlflow.log_metric("recall", recall_score(y_test, preds))
    mlflow.log_metric("f1_score", f1_score(y_test, preds))