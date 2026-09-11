"""
Step 4: Split the data and train two models.

THE CORE IDEA -- TRAIN/TEST SPLIT:
If we let a model see 100% of our data and then test it on that same data,
it's like grading a student on the exact questions they studied. It'll look
great but tell us nothing about whether it actually learned anything useful.
So we hold back a chunk of data (the "test set") that the model NEVER sees
during training, and only use it afterward to check performance honestly.

We train two models to compare:
1. Logistic Regression -- a simple, fast, easy-to-interpret baseline.
   (Despite the name "regression", it's used for CLASSIFICATION.)
2. Random Forest -- a more powerful model that combines many small decision
   trees. Usually does better on messy real-world tabular data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/encoded_subway_delays.csv")

# X = the "features" -- everything the model is allowed to look at.
# y = the "target" -- the one thing we're trying to predict (had_delay).
X = df.drop(columns=["had_delay"])
y = df["had_delay"]

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}\n")

# train_test_split randomly divides our data: 80% for training, 20% held
# back for testing. "random_state=42" just fixes the randomness so that if
# we re-run this script, we get the exact same split every time -- this
# makes results reproducible, which matters a lot for a portfolio project
# (anyone re-running your code should get the same numbers).
#
# "stratify=y" is an important detail for imbalanced data like ours (31.5%
# vs 68.5%): it makes sure BOTH the train and test sets keep that same
# ~31.5%/68.5% ratio, instead of a random split accidentally putting almost
# all the delays into one side.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train):,} rows")
print(f"Test set: {len(X_test):,} rows (the model will NEVER see this during training)\n")

# ---------------------------------------------------------------------------
# MODEL 1: Logistic Regression (the simple baseline)
# ---------------------------------------------------------------------------
# "max_iter=1000" gives the training algorithm enough attempts to converge
# on a good answer (the default is sometimes too low for datasets this size
# and can produce a warning).
print("Training Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)  # ".fit" is where the actual "learning" happens
print("Done.\n")

# ---------------------------------------------------------------------------
# MODEL 2: Random Forest (a more powerful model)
# ---------------------------------------------------------------------------
# A random forest builds many individual "decision trees" (each one a
# simple flowchart of yes/no questions like "is hour > 20?") on random
# slices of the data, then lets them vote. This usually captures more
# complex patterns than logistic regression, at the cost of being harder
# to interpret directly.
#
# n_estimators=200 means "build 200 individual trees and combine them".
# class_weight="balanced" tells the model to pay extra attention to the
# minority class (had_delay=1, which is only 31.5% of our data) instead of
# just favoring the majority class to rack up easy accuracy.
print("Training Random Forest...")
rand_forest = RandomForestClassifier(
    n_estimators=200, max_depth=12, class_weight="balanced",
    random_state=42, n_jobs=-1,
)
rand_forest.fit(X_train, y_train)
print("Done.\n")

# Save both models and the test set to disk so the next script (evaluation)
# can load them without retraining. joblib is a standard tool for saving
# trained scikit-learn models to a file.
joblib.dump(log_reg, "data/model_logistic_regression.joblib")
joblib.dump(rand_forest, "data/model_random_forest.joblib")
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("Saved both trained models and the test set to the data/ folder.")
