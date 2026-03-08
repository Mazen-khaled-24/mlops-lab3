import pandas as pd
import os
from sklearn.model_selection import train_test_split

# load dataset
df = pd.read_csv("data/loan_approval.csv")

# remove missing values
df.dropna(inplace=True)

# convert categorical variables
df = pd.get_dummies(df, drop_first=True)

# split features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ensure data folder exists
os.makedirs("data", exist_ok=True)

# save processed data
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("Preprocessing finished.")