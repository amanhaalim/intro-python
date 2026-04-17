# ============================================================
#  INTERMEDIATE SKLEARN PROGRAM — Level 2
#  Topic: Classification — Iris Flower Species Prediction
#         Comparing Multiple Classifiers + Feature Engineering
#  Author: [Your Name]
#  Prerequisites: You should understand Level 1 (Linear Regression) first.
#
#  What you will learn:
#    • Classification vs Regression
#    • Real sklearn dataset (Iris)
#    • Feature scaling (StandardScaler)
#    • Pipelines (chain preprocessing + model in one object)
#    • Comparing multiple models on the same data
#    • Confusion matrix and classification report
#    • Cross-validation (a fairer way to measure accuracy)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns                          # Better-looking statistical charts

# sklearn — datasets
from sklearn.datasets import load_iris         # Famous built-in dataset

# sklearn — preprocessing
from sklearn.preprocessing import StandardScaler  # Normalises feature values

# sklearn — model selection
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,   # Runs k-fold cross-validation automatically
    StratifiedKFold    # Ensures each fold has balanced class representation
)

# sklearn — classifiers (we will compare all four)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# sklearn — pipeline (chains steps together neatly)
from sklearn.pipeline import Pipeline

# sklearn — evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,   # Precision, Recall, F1-score per class
    confusion_matrix          # Table of correct vs wrong predictions
)

# ── SECTION 1: Load and Explore the Dataset ───────────────────
# The Iris dataset contains measurements of 150 iris flowers across 3 species.
# It is the "Hello World" of machine learning classification.
#
# Features (what we measure):
#   • sepal length (cm)
#   • sepal width  (cm)
#   • petal length (cm)
#   • petal width  (cm)
#
# Target (what we want to predict):
#   • 0 = Iris setosa
#   • 1 = Iris versicolor
#   • 2 = Iris virginica

iris = load_iris()
X = iris.data           # shape: (150, 4) — 150 rows, 4 feature columns
y = iris.target         # shape: (150,)   — 150 class labels (0, 1, or 2)

feature_names  = iris.feature_names  # ['sepal length (cm)', ...]
class_names    = iris.target_names   # ['setosa', 'versicolor', 'virginica']

print("=" * 60)
print("SECTION 1 — Dataset Overview")
print("=" * 60)
print(f"  Dataset shape       : {X.shape}  (rows, columns)")
print(f"  Features            : {feature_names}")
print(f"  Classes to predict  : {class_names}")
print(f"  Samples per class   : {np.bincount(y)}")  # should be [50, 50, 50]
print()

# Quick statistics for each feature
print("  Feature statistics:")
print(f"  {'Feature':<22} {'Min':>6} {'Max':>6} {'Mean':>6} {'Std':>6}")
print("  " + "-" * 46)
for i, name in enumerate(feature_names):
    print(f"  {name:<22} {X[:,i].min():>6.2f} {X[:,i].max():>6.2f} "
          f"{X[:,i].mean():>6.2f} {X[:,i].std():>6.2f}")

# ── SECTION 2: Why Feature Scaling Matters ───────────────────
# Some algorithms (KNN, Logistic Regression) are sensitive to the SCALE of features.
# If one feature ranges 0–100 and another 0–1, the large-scale feature dominates.
#
# StandardScaler transforms each feature to have:
#   mean = 0  and  standard deviation = 1
# Formula:  z = (x - mean) / std
#
# Decision Trees and Random Forests do NOT need scaling — they split by thresholds.
# We include scaling for ALL models in a Pipeline anyway to keep the code uniform.

# ── SECTION 3: Train / Test Split ────────────────────────────
# stratify=y ensures each split has the same class proportions (50/50/50 ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,    # 25% for testing
    random_state=42,
    stratify=y         # IMPORTANT for classification — keeps class balance
)

print(f"\nSECTION 2 — Data Split (stratified)")
print(f"  Train size : {len(X_train)} samples | class counts: {np.bincount(y_train)}")
print(f"  Test size  : {len(X_test)}  samples | class counts: {np.bincount(y_test)}")

# ── SECTION 4: Build Pipelines ────────────────────────────────
# A Pipeline strings preprocessing steps and a model together.
# Benefits:
#   1. Scaler is FIT only on training data → no data leakage into test set
#   2. Everything (scale + predict) happens with one .fit() / .predict() call
#   3. Easy to swap models without rewriting preprocessing code

# Define four classifiers to compare
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "Decision Tree"      : DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
}

# Wrap each classifier in a Pipeline with StandardScaler
pipelines = {
    name: Pipeline([
        ("scaler", StandardScaler()),  # Step 1: normalise features
        ("model",  clf)                # Step 2: train the classifier
    ])
    for name, clf in classifiers.items()
}

# ── SECTION 5: Train, Predict, and Evaluate All Models ────────
print("\n" + "=" * 60)
print("SECTION 3 — Model Training & Evaluation")
print("=" * 60)

results = {}   # store results for comparison chart later

for name, pipeline in pipelines.items():
    # --- Train ---
    pipeline.fit(X_train, y_train)

    # --- Predict ---
    y_pred = pipeline.predict(X_test)

    # --- Evaluate ---
    accuracy = accuracy_score(y_test, y_pred)

    # Cross-validation: split the TRAINING data into 5 folds,
    # train on 4, validate on 1, repeat 5 times. Gives a more stable accuracy.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

    results[name] = {
        "pipeline"  : pipeline,
        "y_pred"    : y_pred,
        "accuracy"  : accuracy,
        "cv_mean"   : cv_scores.mean(),
        "cv_std"    : cv_scores.std(),
    }

    print(f"\n  ▶ {name}")
    print(f"      Test  accuracy          : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"      Cross-val accuracy      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── SECTION 6: Detailed Report for Best Model ─────────────────
# Find which model had the highest test accuracy
best_name = max(results, key=lambda n: results[n]["accuracy"])
best      = results[best_name]

print("\n" + "=" * 60)
print(f"SECTION 4 — Best Model: {best_name}")
print("=" * 60)

# Classification report shows Precision, Recall, and F1 for each class
# Precision : of all instances predicted as class X, how many actually were X?
# Recall    : of all actual class X instances, how many did we catch?
# F1-score  : harmonic mean of Precision and Recall (balance between the two)
print("\n  Classification Report:")
print(classification_report(
    y_test, best["y_pred"],
    target_names=class_names
))

# ── SECTION 7: Visualisations ─────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Intermediate Example — Iris Classification", fontsize=16, fontweight="bold")

# ---- Plot 1: Accuracy comparison bar chart ----
ax1 = fig.add_subplot(2, 3, 1)
names = list(results.keys())
accs  = [results[n]["accuracy"]  for n in names]
cv_m  = [results[n]["cv_mean"]   for n in names]
cv_s  = [results[n]["cv_std"]    for n in names]

x = np.arange(len(names))
bars1 = ax1.bar(x - 0.2, accs, 0.35, label="Test accuracy",   color="steelblue")
bars2 = ax1.bar(x + 0.2, cv_m, 0.35, label="CV accuracy",     color="darkorange",
                yerr=cv_s, capsize=4)
ax1.set_xticks(x)
ax1.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
ax1.set_ylim(0.85, 1.02)
ax1.set_ylabel("Accuracy")
ax1.set_title("Model Comparison")
ax1.legend(fontsize=8)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

# ---- Plots 2-4: Confusion matrices for each model (pick 3) ----
selected = list(results.keys())[:3]
subplot_positions = [2, 3, 4]
for pos, name in zip(subplot_positions, selected):
    ax = fig.add_subplot(2, 3, pos)
    cm = confusion_matrix(y_test, results[name]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"Confusion Matrix\n{name}", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

# ---- Plot 5: Feature importance from Random Forest ----
ax5 = fig.add_subplot(2, 3, 5)
rf_pipeline   = results["Random Forest"]["pipeline"]
rf_model      = rf_pipeline.named_steps["model"]  # extract the RF model from pipeline
importances   = rf_model.feature_importances_
sorted_idx    = np.argsort(importances)[::-1]
ax5.bar(range(4), importances[sorted_idx], color="teal")
ax5.set_xticks(range(4))
ax5.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=30, ha="right", fontsize=8)
ax5.set_ylabel("Importance")
ax5.set_title("Random Forest\nFeature Importance")

# ---- Plot 6: Pairplot of two most important features ----
ax6 = fig.add_subplot(2, 3, 6)
# The two most informative features by RF importance
top2 = sorted_idx[:2]
colors = ["red", "green", "blue"]
for cls in range(3):
    mask = y == cls
    ax6.scatter(X[mask, top2[0]], X[mask, top2[1]],
                c=colors[cls], label=class_names[cls], alpha=0.7)
ax6.set_xlabel(feature_names[top2[0]])
ax6.set_ylabel(feature_names[top2[1]])
ax6.set_title("Top-2 Features\n(all 150 samples)")
ax6.legend(fontsize=8)

plt.tight_layout()
plt.savefig("02_intermediate_output.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved as '02_intermediate_output.png'")

# ── FINAL SUMMARY ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY — Leaderboard")
print("=" * 60)
print(f"  {'Model':<25} {'Test Acc':>10} {'CV Acc':>10} {'CV Std':>10}")
print("  " + "-" * 57)
for name in sorted(results, key=lambda n: results[n]["cv_mean"], reverse=True):
    r = results[name]
    print(f"  {name:<25} {r['accuracy']:>9.1%} {r['cv_mean']:>9.1%} {r['cv_std']:>9.4f}")

# ============================================================
#  🎯 EXERCISES FOR STUDENTS
# ============================================================
# 1. Change `max_depth` in DecisionTreeClassifier from 4 to 1, then to 10.
#    What happens to training vs test accuracy? This is called "overfitting".
#
# 2. Change `n_neighbors` in KNeighborsClassifier from 5 to 1.
#    What happens? Then try 20. Which is better and why?
#
# 3. Remove StandardScaler from the KNN pipeline by replacing it with "passthrough".
#    Does the accuracy drop? Which features do you think suffer most without scaling?
#
# 4. Try a Support Vector Machine (SVM):
#    from sklearn.svm import SVC
#    Add it to the `classifiers` dict and re-run. Where does it rank?
#
# 5. Print the confusion matrix for the worst-performing model.
#    Which classes does it confuse most? Can you explain why by looking at Plot 6?
# ============================================================
