# ============================================================
#  EXPERT SKLEARN PROGRAM — Level 3
#  Topic: End-to-End ML Pipeline on Real-World Tabular Data
#         (Predicting Credit Default Risk)
#
#  What you will learn:
#    • Simulating a realistic messy dataset (missing values, mixed types)
#    • ColumnTransformer — apply different preprocessing to different columns
#    • Custom Transformer — write your own sklearn-compatible transformer
#    • FeatureUnion — combine multiple feature sets
#    • Voting & Stacking Ensembles
#    • Hyperparameter tuning with GridSearchCV
#    • Learning Curves — diagnosing bias vs variance
#    • ROC / AUC — evaluation for imbalanced binary classification
#    • SHAP-style manual feature contribution (without extra libraries)
#    • Saving and loading a trained pipeline with joblib
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")           # suppress convergence warnings for clarity

# ── sklearn imports (grouped by purpose) ──────────────────────

# Data handling
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold,
    learning_curve, cross_val_score
)

# Preprocessing
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, LabelEncoder,
    PolynomialFeatures, FunctionTransformer
)
from sklearn.impute import SimpleImputer    # Handles missing values

# Pipeline tools
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer  # Applies different transforms per column
from sklearn.base import BaseEstimator, TransformerMixin  # For custom transformers

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC

# Evaluation
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

# Persistence
import joblib
import os

# ── SECTION 1: Simulate a Realistic Messy Dataset ─────────────
# In real life data comes from databases and is rarely perfect.
# We simulate common problems: missing values, categorical columns,
# class imbalance (more non-defaults than defaults).

np.random.seed(42)
N = 2000   # number of loan applicants

# ---- Generate base features with sklearn's helper ----
X_base, y = make_classification(
    n_samples=N,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    weights=[0.80, 0.20],   # 80% did NOT default, 20% did → class imbalance
    random_state=42,
    flip_y=0.05             # 5% label noise (real-world labels are noisy)
)

# ---- Build a meaningful Pandas DataFrame ----
df = pd.DataFrame(X_base, columns=[f"num_{i}" for i in range(10)])

# Add interpretable names for 5 numerical columns
df.rename(columns={
    "num_0": "age",
    "num_1": "income",
    "num_2": "credit_score",
    "num_3": "loan_amount",
    "num_4": "debt_to_income",
}, inplace=True)

# Scale numerical columns to plausible real-world ranges
df["age"]           = (df["age"] * 10 + 45).clip(18, 80).round(0)
df["income"]        = (df["income"] * 20000 + 55000).clip(15000, 200000).round(-2)
df["credit_score"]  = (df["credit_score"] * 80 + 680).clip(300, 850).round(0)
df["loan_amount"]   = (df["loan_amount"] * 15000 + 25000).clip(1000, 100000).round(-2)
df["debt_to_income"]= (df["debt_to_income"] * 0.15 + 0.35).clip(0.05, 0.95).round(3)

# ---- Add categorical features ----
employment_types = np.random.choice(
    ["Salaried", "Self-Employed", "Unemployed", "Retired"],
    size=N, p=[0.55, 0.25, 0.12, 0.08]
)
loan_purpose = np.random.choice(
    ["Home", "Car", "Education", "Medical", "Personal"],
    size=N, p=[0.30, 0.25, 0.15, 0.10, 0.20]
)
df["employment_type"] = employment_types
df["loan_purpose"]    = loan_purpose

# ---- Introduce MISSING values (~8% randomly) ----
# Real datasets ALWAYS have missing data. We need to handle this.
for col in ["income", "credit_score", "employment_type"]:
    missing_idx = np.random.choice(N, size=int(N * 0.08), replace=False)
    df.loc[missing_idx, col] = np.nan

df["default"] = y  # target variable: 1 = defaulted, 0 = did not default

print("=" * 65)
print("SECTION 1 — Dataset Overview")
print("=" * 65)
print(df.describe(include="all").T.to_string())
print(f"\n  Class distribution:\n{df['default'].value_counts(normalize=True).to_string()}")
print(f"\n  Missing values per column:\n{df.isnull().sum().to_string()}")

# ── SECTION 2: Define Features ────────────────────────────────
target      = "default"
num_cols    = ["age", "income", "credit_score", "loan_amount",
               "debt_to_income", "num_5", "num_6", "num_7", "num_8", "num_9"]
cat_cols    = ["employment_type", "loan_purpose"]

X = df[num_cols + cat_cols]
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nSECTION 2 — Train/Test sizes: {len(X_train)} / {len(X_test)}")

# ── SECTION 3: Custom Transformer ─────────────────────────────
# sklearn lets you write your own transformation steps.
# Any class that inherits BaseEstimator + TransformerMixin and implements
# fit() and transform() can be dropped into a Pipeline.
#
# Here we create a transformer that adds an engineered feature:
#   "loan_to_income_ratio" = loan_amount / income

class LoanToIncomeTransformer(BaseEstimator, TransformerMixin):
    """
    Adds a loan-to-income ratio column.
    column_indices: list of [loan_amount_idx, income_idx] in the numerical matrix
    """
    def __init__(self, loan_idx=3, income_idx=1):
        self.loan_idx   = loan_idx
        self.income_idx = income_idx

    def fit(self, X, y=None):
        # Nothing to learn from data — just return self
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        loan   = X[:, self.loan_idx]
        income = X[:, self.income_idx]

        # Avoid division by zero; replace 0 income with NaN then 0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(income != 0, loan / income, 0.0)

        # Append ratio as a new column
        return np.hstack([X, ratio.reshape(-1, 1)])

# ── SECTION 4: Build the ColumnTransformer ────────────────────
# ColumnTransformer applies DIFFERENT preprocessing to DIFFERENT columns.
# This is essential for mixed-type data (numerical + categorical).
#
# Numerical pipeline:
#   1. SimpleImputer   — fills NaN with the column median
#   2. Custom transformer — adds engineered feature
#   3. StandardScaler  — normalises to mean=0, std=1
#
# Categorical pipeline:
#   1. SimpleImputer   — fills NaN with the most frequent value
#   2. OneHotEncoder   — converts category labels to binary columns

numerical_pipeline = Pipeline([
    ("imputer",    SimpleImputer(strategy="median")),
    ("engineer",   LoanToIncomeTransformer(loan_idx=3, income_idx=1)),
    ("scaler",     StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer",    SimpleImputer(strategy="most_frequent")),
    ("onehot",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols),
])

print("\nSECTION 3 — Preprocessor structure:")
print(preprocessor)

# ── SECTION 5: Define Base Models ─────────────────────────────
# We will build an ensemble. First define individual base estimators.

lr  = LogisticRegression(max_iter=500, C=1.0,  class_weight="balanced", random_state=42)
rf  = RandomForestClassifier(n_estimators=200,  class_weight="balanced", random_state=42)
gb  = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,  random_state=42)

# ── SECTION 6: Voting Ensemble ────────────────────────────────
# Soft voting = average the predicted PROBABILITIES from all base models.
# This usually outperforms hard voting (majority-vote of class labels).

voting_clf = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
    voting="soft"
)

# ── SECTION 7: Stacking Ensemble ──────────────────────────────
# Stacking is more powerful than voting:
#   Level 0 (base models): train on the training set, make out-of-fold predictions
#   Level 1 (meta-model) : trained on those out-of-fold predictions
# The meta-model learns HOW to combine the base models' outputs.

stacking_clf = StackingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
    final_estimator=LogisticRegression(max_iter=500),   # meta-model
    cv=5,                # 5-fold cross-val to generate meta-features
    stack_method="predict_proba"
)

# ── SECTION 8: Full Pipelines for Each Model ──────────────────
models = {
    "Logistic Regression"  : Pipeline([("prep", preprocessor), ("clf", lr)]),
    "Random Forest"        : Pipeline([("prep", preprocessor), ("clf", rf)]),
    "Gradient Boosting"    : Pipeline([("prep", preprocessor), ("clf", gb)]),
    "Voting Ensemble"      : Pipeline([("prep", preprocessor), ("clf", voting_clf)]),
    "Stacking Ensemble"    : Pipeline([("prep", preprocessor), ("clf", stacking_clf)]),
}

# ── SECTION 9: Train & Compare All Models ─────────────────────
print("\n" + "=" * 65)
print("SECTION 4 — Model Training & AUC Comparison")
print("=" * 65)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
eval_results = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]  # probability of class 1 (default)
    auc      = roc_auc_score(y_test, y_proba)

    cv_auc = cross_val_score(
        pipe, X_train, y_train,
        cv=cv_strategy, scoring="roc_auc"
    )

    eval_results[name] = {
        "pipe"    : pipe,
        "y_proba" : y_proba,
        "auc"     : auc,
        "cv_auc"  : cv_auc,
    }
    print(f"  {name:<25}  Test AUC={auc:.4f}  "
          f"CV AUC={cv_auc.mean():.4f}±{cv_auc.std():.4f}")

# ── SECTION 10: Hyperparameter Tuning with GridSearchCV ───────
# GridSearchCV exhaustively tries every combination of parameters
# and uses cross-validation to find the best one.
# We tune only Logistic Regression here to keep runtime short;
# in production you would tune all models.

print("\nSECTION 5 — Hyperparameter Tuning (Logistic Regression)")
print("  This may take ~30 seconds ...")

lr_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf",  LogisticRegression(max_iter=500, class_weight="balanced", random_state=42))
])

# Parameter grid: "clf__C" means the parameter "C" of the "clf" step
param_grid = {
    "clf__C"      : [0.01, 0.1, 1, 10, 100],
    "clf__solver" : ["lbfgs", "saga"],
}

grid_search = GridSearchCV(
    lr_pipeline,
    param_grid,
    scoring="roc_auc",
    cv=cv_strategy,
    n_jobs=-1,       # use all CPU cores
    verbose=0
)
grid_search.fit(X_train, y_train)

print(f"  Best parameters : {grid_search.best_params_}")
print(f"  Best CV AUC     : {grid_search.best_score_:.4f}")

tuned_auc = roc_auc_score(y_test, grid_search.predict_proba(X_test)[:,1])
print(f"  Tuned test AUC  : {tuned_auc:.4f}  "
      f"(baseline was {eval_results['Logistic Regression']['auc']:.4f})")

# Add tuned model to results for comparison
eval_results["Tuned LR (GridSearch)"] = {
    "pipe"    : grid_search.best_estimator_,
    "y_proba" : grid_search.predict_proba(X_test)[:,1],
    "auc"     : tuned_auc,
    "cv_auc"  : np.array([grid_search.best_score_])
}

# ── SECTION 11: Best Model Analysis ───────────────────────────
# Identify the overall best model by test AUC
best_name  = max(eval_results, key=lambda n: eval_results[n]["auc"])
best_pipe  = eval_results[best_name]["pipe"]
best_proba = eval_results[best_name]["y_proba"]

print(f"\n  ★ Best model: {best_name}  (Test AUC = {eval_results[best_name]['auc']:.4f})")

# Choose a classification threshold
# Default is 0.5 but in credit risk we often prefer higher recall for defaults
# (false negative = missed default = costly). Here we use 0.4.
THRESHOLD   = 0.4
y_pred_best = (best_proba >= THRESHOLD).astype(int)

print(f"\n  Classification Report (threshold = {THRESHOLD}):")
print(classification_report(y_test, y_pred_best, target_names=["No Default", "Default"]))

# ── SECTION 12: Visualisations ────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.suptitle("Expert Example — Credit Default Risk Prediction", fontsize=16, fontweight="bold")
gs  = gridspec.GridSpec(2, 3, figure=fig)

# ---- Plot 1: ROC curves for all models ----
ax1 = fig.add_subplot(gs[0, 0])
for name, res in eval_results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax1.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", linewidth=1.5)
ax1.plot([0,1],[0,1],"k--", label="Random")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curves — All Models")
ax1.legend(fontsize=7, loc="lower right")

# ---- Plot 2: Precision-Recall Curve (best model) ----
ax2 = fig.add_subplot(gs[0, 1])
prec, rec, thresh = precision_recall_curve(y_test, best_proba)
ap = average_precision_score(y_test, best_proba)
ax2.plot(rec, prec, color="darkorange", linewidth=2, label=f"AP={ap:.3f}")
ax2.axvline(x=rec[np.argmin(np.abs(thresh - THRESHOLD))],
            color="red", linestyle="--", label=f"Threshold={THRESHOLD}")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title(f"Precision-Recall — {best_name}")
ax2.legend(fontsize=8)
# Class imbalance baseline: a random classifier achieves AP ≈ class prevalence
ax2.axhline(y=y.mean(), color="gray", linestyle=":", label=f"Baseline={y.mean():.2f}")

# ---- Plot 3: Confusion Matrix (best model, chosen threshold) ----
ax3 = fig.add_subplot(gs[0, 2])
cm   = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"])
disp.plot(ax=ax3, colorbar=False, cmap="Blues")
ax3.set_title(f"Confusion Matrix — {best_name}\n(threshold={THRESHOLD})")

# ---- Plot 4: Learning Curve ----
# A learning curve shows how model performance changes as training data grows.
# ● If train score >> val score → overfitting → need more data or regularisation
# ● If both scores are low     → underfitting → need more complex model
ax4 = fig.add_subplot(gs[1, 0])
train_sizes, train_scores, val_scores = learning_curve(
    best_pipe, X_train, y_train,
    cv=cv_strategy,
    scoring="roc_auc",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)
ax4.plot(train_sizes, train_scores.mean(axis=1), "o-", color="steelblue", label="Train AUC")
ax4.fill_between(train_sizes,
                 train_scores.mean(1) - train_scores.std(1),
                 train_scores.mean(1) + train_scores.std(1), alpha=0.2, color="steelblue")
ax4.plot(train_sizes, val_scores.mean(axis=1), "o-", color="darkorange", label="Val AUC")
ax4.fill_between(train_sizes,
                 val_scores.mean(1) - val_scores.std(1),
                 val_scores.mean(1) + val_scores.std(1), alpha=0.2, color="darkorange")
ax4.set_xlabel("Training Samples")
ax4.set_ylabel("AUC")
ax4.set_title(f"Learning Curve — {best_name}")
ax4.legend()

# ---- Plot 5: Feature Importance from Random Forest ----
ax5 = fig.add_subplot(gs[1, 1])
rf_pipe = eval_results["Random Forest"]["pipe"]
rf_prep = rf_pipe.named_steps["prep"]
rf_clf  = rf_pipe.named_steps["clf"]

# Reconstruct feature names after preprocessing
num_feature_names = num_cols + ["loan_to_income"]     # custom engineered column
cat_feature_names = list(
    rf_prep.named_transformers_["cat"]
           .named_steps["onehot"]
           .get_feature_names_out(cat_cols)
)
all_feature_names = num_feature_names + cat_feature_names

importances   = rf_clf.feature_importances_
top_n         = 12
top_idx       = np.argsort(importances)[-top_n:]
ax5.barh(range(top_n), importances[top_idx], color="teal")
ax5.set_yticks(range(top_n))
ax5.set_yticklabels([all_feature_names[i] for i in top_idx], fontsize=8)
ax5.set_xlabel("Importance")
ax5.set_title("Top-12 Feature Importances\n(Random Forest)")

# ---- Plot 6: Threshold vs F1 / Precision / Recall ----
ax6 = fig.add_subplot(gs[1, 2])
thresholds = np.linspace(0.01, 0.99, 100)
precisions, recalls, f1s = [], [], []
for t in thresholds:
    yp = (best_proba >= t).astype(int)
    if yp.sum() == 0:
        precisions.append(0); recalls.append(0); f1s.append(0)
        continue
    from sklearn.metrics import precision_score, recall_score, f1_score
    precisions.append(precision_score(y_test, yp, zero_division=0))
    recalls.append(recall_score(y_test, yp, zero_division=0))
    f1s.append(f1_score(y_test, yp, zero_division=0))
ax6.plot(thresholds, precisions, label="Precision",    color="blue")
ax6.plot(thresholds, recalls,    label="Recall",       color="green")
ax6.plot(thresholds, f1s,        label="F1-score",     color="red")
ax6.axvline(x=THRESHOLD, color="black", linestyle="--", label=f"Chosen={THRESHOLD}")
ax6.set_xlabel("Classification Threshold")
ax6.set_ylabel("Score")
ax6.set_title("Threshold vs Precision/Recall/F1")
ax6.legend(fontsize=8)

plt.tight_layout()
plt.savefig("03_expert_output.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved as '03_expert_output.png'")

# ── SECTION 13: Save and Reload the Best Model ────────────────
# In production you save the trained pipeline to disk and reload it later
# without retraining. The whole pipeline (preprocessor + model) is saved.

model_path = "best_credit_risk_model.joblib"
joblib.dump(best_pipe, model_path)
print(f"\nSECTION 6 — Model saved to '{model_path}'")

# Reload and verify
loaded_pipe  = joblib.load(model_path)
reloaded_auc = roc_auc_score(y_test, loaded_pipe.predict_proba(X_test)[:,1])
assert abs(reloaded_auc - eval_results[best_name]["auc"]) < 1e-9, "AUC mismatch!"
print(f"  Reloaded model AUC: {reloaded_auc:.4f}  ✓ matches saved model")

# ── SECTION 14: Scoring New Applicants ────────────────────────
# Simulate three new loan applicants and predict their default probability
new_applicants = pd.DataFrame({
    "age"            : [28,    55,    40],
    "income"         : [32000, 95000, 50000],
    "credit_score"   : [580,   780,   650],
    "loan_amount"    : [20000, 10000, 35000],
    "debt_to_income" : [0.65,  0.20,  0.45],
    "num_5": [0.1, -0.3, 0.8],
    "num_6": [-0.5, 1.2, 0.2],
    "num_7": [0.3, -0.7, 1.1],
    "num_8": [-1.0, 0.4, -0.2],
    "num_9": [0.7, -0.1, 0.5],
    "employment_type": ["Unemployed", "Salaried", "Self-Employed"],
    "loan_purpose"   : ["Personal",  "Home",     "Car"],
})

default_probs = loaded_pipe.predict_proba(new_applicants)[:, 1]

print("\nSECTION 7 — Scoring New Applicants")
print(f"  {'Applicant':<12} {'Income':>10} {'Credit Score':>14} {'Default Risk%':>15}")
print("  " + "-" * 53)
for i, prob in enumerate(default_probs):
    risk_level = "🔴 HIGH" if prob > 0.5 else ("🟡 MEDIUM" if prob > 0.3 else "🟢 LOW")
    print(f"  Applicant {i+1:<2}  ${new_applicants['income'][i]:>9,}  "
          f"{int(new_applicants['credit_score'][i]):>13}  "
          f"{prob*100:>11.1f}%  {risk_level}")

# ── FINAL LEADERBOARD ─────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL LEADERBOARD — Ranked by Test AUC")
print("=" * 65)
print(f"  {'Model':<30} {'Test AUC':>10} {'CV AUC':>10}")
print("  " + "-" * 52)
for name in sorted(eval_results, key=lambda n: eval_results[n]["auc"], reverse=True):
    r = eval_results[name]
    cv_str = f"{r['cv_auc'].mean():.4f}" if len(r["cv_auc"]) > 1 else "  N/A  "
    print(f"  {name:<30} {r['auc']:>10.4f} {cv_str:>10}")

# ============================================================
#  🎯 EXERCISES FOR STUDENTS
# ============================================================
# 1. THRESHOLD TUNING:
#    Change THRESHOLD from 0.4 to 0.3. How does the confusion matrix change?
#    In banking, which is more costly: a false negative (missed default)
#    or a false positive (wrongly rejected good customer)?
#
# 2. CLASS IMBALANCE:
#    Try changing `weights` in make_classification to [0.95, 0.05].
#    Now defaults are very rare. What happens to AUC?
#    Research "SMOTE" (Synthetic Minority Over-sampling Technique).
#    Install imbalanced-learn and add it to the numerical pipeline.
#
# 3. CUSTOM TRANSFORMER:
#    Add a second custom transformer that flags applicants with
#    debt_to_income > 0.5 as a binary "high_debt" column.
#
# 4. FEATURE SELECTION:
#    Add sklearn.feature_selection.SelectKBest to the numerical pipeline
#    (after scaling) with k=7. Does this improve or hurt performance?
#
# 5. HYPERPARAMETER TUNING for Random Forest:
#    Define a param_grid for RandomForestClassifier with:
#    n_estimators: [50, 100, 200], max_depth: [None, 5, 10]
#    Run GridSearchCV and compare with the untuned version.
#
# 6. MODEL PERSISTENCE:
#    Load the saved joblib model in a NEW Python script and score
#    the three new applicants without any training code present.
#    This simulates a real production inference service.
#
# 7. ADVANCED:
#    Replace RandomForest with XGBClassifier from xgboost.
#    Hint: pip install xgboost  then  from xgboost import XGBClassifier
#    Add it to the stacking ensemble as a fourth base model.
# ============================================================
