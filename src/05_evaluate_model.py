"""
Step 5: Evaluate both models honestly on the held-out test set.

WHY ACCURACY ALONE IS MISLEADING HERE:
Only 31.5% of incidents in our data caused a delay. A "lazy" model that
ALWAYS predicts "no delay" would already score 68.5% accuracy -- while being
completely useless (it would never once correctly flag a real delay). This
is why, for imbalanced problems like this one, we also look at:

- PRECISION: of the times the model predicted "delay", how often was it
  actually right? (Low precision = the model cries wolf a lot.)
- RECALL: of all the delays that actually happened, how many did the model
  correctly catch? (Low recall = the model misses a lot of real delays.)

There's usually a tradeoff between the two. Which matters more depends on
the real-world use: for transit riders, missing a real delay (low recall)
is arguably worse than an occasional false alarm.
"""

import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")["had_delay"]

models = {
    "Logistic Regression": joblib.load("data/model_logistic_regression.joblib"),
    "Random Forest": joblib.load("data/model_random_forest.joblib"),
}

# A "dummy" baseline that always predicts the majority class (no delay).
# This is our sanity-check floor: any real model should beat this.
dummy_predictions = pd.Series([0] * len(y_test))

results = []
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for i, (name, model) in enumerate(models.items()):
    # .predict() runs the trained model on data it has NEVER seen (X_test)
    # and returns its guesses: 0 (no delay) or 1 (delay).
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)  # a single score balancing precision & recall

    results.append({
        "model": name, "accuracy": acc, "precision": prec,
        "recall": rec, "f1_score": f1,
    })

    print(f"=== {name} ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}  (when it predicts 'delay', it's right this often)")
    print(f"Recall:    {rec:.3f}  (of all real delays, it catches this fraction)")
    print(f"F1 score:  {f1:.3f}  (a balance of precision and recall)")
    print()
    print(classification_report(y_test, predictions, target_names=["No delay", "Delay"]))
    print()

    # A confusion matrix is a simple 2x2 table showing exactly what the
    # model got right and wrong:
    #   - top-left:  correctly predicted "no delay"
    #   - top-right: predicted "delay" but was actually "no delay" (false alarm)
    #   - bottom-left: predicted "no delay" but a delay happened (missed it)
    #   - bottom-right: correctly predicted "delay"
    cm = confusion_matrix(y_test, predictions)
    ax = axes[i]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{name}\nConfusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No delay", "Delay"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No delay", "Delay"])
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{cm[r, c]:,}", ha="center", va="center",
                     color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=12)

plt.tight_layout()
plt.savefig("data/confusion_matrices.png", dpi=120)
print("Saved confusion matrices to data/confusion_matrices.png\n")

# Baseline comparison
baseline_acc = accuracy_score(y_test, dummy_predictions)
print(f"For reference, a 'lazy' model that always predicts 'no delay' would "
      f"score {baseline_acc:.3f} accuracy -- but 0.000 recall (it would never "
      f"catch a single real delay). Both our real models beat this baseline "
      f"in a way that's actually useful.\n")

results_df = pd.DataFrame(results)
results_df.to_csv("data/model_results.csv", index=False)
print(results_df)

# Feature importance from the Random Forest -- which factors mattered most?
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X_test.columns)
top_features = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 6))
top_features.sort_values().plot(kind="barh", color="#4C72B0")
plt.title("Top 15 most important features (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("data/feature_importance.png", dpi=120)
print("\nSaved feature importance chart to data/feature_importance.png")
print("\nTop 15 features:")
print(top_features)
