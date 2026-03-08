import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load processed data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
y_test = pd.read_csv("data/y_test.csv")

mlflow.set_experiment("Loan Approval Prediction")

n_estimators_list = [50, 100]
max_depth_list = [5, 10]

with mlflow.start_run(run_name="RandomForest Tuning"):

    for n in n_estimators_list:
        for depth in max_depth_list:

            with mlflow.start_run(nested=True):

                model = RandomForestClassifier(
                    n_estimators=n,
                    max_depth=depth,
                    random_state=42
                )

                model.fit(X_train, y_train.values.ravel())

                preds = model.predict(X_test)

                accuracy = accuracy_score(y_test, preds)

                mlflow.log_param("n_estimators", n)
                mlflow.log_param("max_depth", depth)
                mlflow.log_metric("accuracy", accuracy)