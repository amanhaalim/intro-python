# ============================================================
#  BEGINNER SKLEARN PROGRAM — Level 1
#  Topic: Linear Regression (Predicting House Prices)
#  Author: [Your Name]
#  Instructions for students:
#    1. Read every comment carefully before running anything.
#    2. Run the file with:  python 01_beginner_linear_regression.py
#    3. Try the exercises at the bottom after you understand the code.
# ============================================================

# ── STEP 0: Import the tools we need ─────────────────────────
# Think of imports like opening your toolbox before starting a job.

import numpy as np                        # NumPy: handles numbers & arrays efficiently
import matplotlib.pyplot as plt           # Matplotlib: draws charts and graphs
from sklearn.linear_model import LinearRegression   # The ML model we will use
from sklearn.model_selection import train_test_split # Splits data into train/test sets
from sklearn.metrics import mean_squared_error, r2_score  # Ways to measure accuracy

# ── STEP 1: Create some data ──────────────────────────────────
# In real life you would load data from a CSV file.
# For now we create a simple fake dataset so you can focus on learning sklearn.
#
# Scenario: We have the SIZE of a house (in sq ft) and want to PREDICT its PRICE.
#   - X = feature  (input)  → house size
#   - y = target   (output) → house price

np.random.seed(42)   # "Seed" the random number generator so results are reproducible
                     # (everyone in class gets the same numbers)

# Create 100 house sizes between 500 and 3500 sq ft
house_sizes = np.random.randint(500, 3500, size=100).reshape(-1, 1)
#   .reshape(-1, 1) turns a flat list [500, 1200, ...] into a column:
#   [[500], [1200], ...]  ← sklearn always wants features in this 2-D shape

# Create matching prices using a simple formula + a little random noise
# Formula: price = (size × 150) + 50000 + noise
noise = np.random.randint(-20000, 20000, size=100)
house_prices = (house_sizes.flatten() * 150) + 50000 + noise

print("=" * 50)
print("STEP 1 — Dataset created")
print(f"  Number of houses : {len(house_sizes)}")
print(f"  Smallest house   : {house_sizes.min()} sq ft")
print(f"  Largest house    : {house_sizes.max()} sq ft")
print("=" * 50)

# ── STEP 2: Split data into Training set and Test set ─────────
# WHY? We train the model on one portion of the data (training set)
#       and then test how well it learned on data it has NEVER seen (test set).
#       This tells us if the model actually generalised or just memorised.
#
# test_size=0.2 means 20% of the data is used for testing, 80% for training.

X_train, X_test, y_train, y_test = train_test_split(
    house_sizes,   # features  (input)
    house_prices,  # targets   (output)
    test_size=0.2, # 20% goes to test set
    random_state=42  # makes the split the same every time you run the program
)

print(f"\nSTEP 2 — Data split")
print(f"  Training samples : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")

# ── STEP 3: Create and Train the Model ───────────────────────
# sklearn follows a simple 3-step pattern for EVERY model:
#   (a) Create  →  model = SomeModel()
#   (b) Train   →  model.fit(X_train, y_train)
#   (c) Predict →  predictions = model.predict(X_test)

# (a) Create the model object — no training happens yet
model = LinearRegression()
print("\nSTEP 3 — Model created:", type(model).__name__)

# (b) Train (fit) the model on the training data
#     The model looks at house sizes & prices and finds the best straight line
model.fit(X_train, y_train)
print("      Model trained successfully!")

# Let's look at what the model learned:
print(f"\n  Slope     (coefficient) : ${model.coef_[0]:,.2f} per sq ft")
print(f"  Intercept (base price)  : ${model.intercept_:,.2f}")
# Interpretation: every extra sq ft adds ~$150 to the price

# ── STEP 4: Make Predictions ──────────────────────────────────
# Now we ask the model to predict prices for houses it has NEVER seen
y_predicted = model.predict(X_test)

print("\nSTEP 4 — Sample Predictions vs Actual Prices")
print(f"  {'Size (sq ft)':<15} {'Actual Price':>15} {'Predicted Price':>17}")
print("  " + "-" * 47)
for size, actual, pred in zip(X_test[:5], y_test[:5], y_predicted[:5]):
    print(f"  {size[0]:<15} ${actual:>14,.0f} ${pred:>15,.0f}")

# ── STEP 5: Evaluate the Model ───────────────────────────────
# Two common evaluation metrics for regression:
#
#   Mean Squared Error (MSE):
#     → Average of (actual - predicted)² for all test samples
#     → Lower is better. Units are in dollars² which is hard to interpret.
#
#   R² Score (R-squared / Coefficient of Determination):
#     → Ranges from 0 to 1.
#     → 1.0 = perfect predictions; 0.0 = no better than guessing the mean.
#     → Think of it as "how much of the variation does our model explain?"

mse = mean_squared_error(y_test, y_predicted)
rmse = np.sqrt(mse)   # Root MSE brings units back to dollars (easier to read)
r2   = r2_score(y_test, y_predicted)

print("\nSTEP 5 — Model Evaluation")
print(f"  Root Mean Squared Error : ${rmse:,.0f}")
print(f"  R² Score                : {r2:.4f}  ({r2*100:.1f}% of variance explained)")

# ── STEP 6: Visualise Results ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Beginner Example — Linear Regression: Predicting House Prices",
             fontsize=14, fontweight="bold")

# ---- Plot 1: The regression line ----
ax = axes[0]
ax.scatter(X_train, y_train, color="steelblue", alpha=0.5, label="Training data")
ax.scatter(X_test,  y_test,  color="orange",    alpha=0.7, label="Test data")
# Draw the fitted line across the full range of sizes
x_line = np.linspace(500, 3500, 200).reshape(-1, 1)
ax.plot(x_line, model.predict(x_line), color="red", linewidth=2, label="Regression line")
ax.set_xlabel("House Size (sq ft)")
ax.set_ylabel("House Price ($)")
ax.set_title("Regression Line Fitted to Data")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# ---- Plot 2: Actual vs Predicted ----
ax = axes[1]
ax.scatter(y_test, y_predicted, color="purple", alpha=0.7)
# A perfect model would have all points ON this diagonal line
perfect_line = [y_test.min(), y_test.max()]
ax.plot(perfect_line, perfect_line, "r--", linewidth=2, label="Perfect prediction")
ax.set_xlabel("Actual Price ($)")
ax.set_ylabel("Predicted Price ($)")
ax.set_title("Actual vs Predicted Prices")
ax.legend()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

plt.tight_layout()
plt.savefig("01_beginner_output.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved as '01_beginner_output.png'")

# ── STEP 7: Predict a New House ───────────────────────────────
new_house_size = np.array([[2000]])   # We want to predict the price of a 2000 sq ft house
predicted_price = model.predict(new_house_size)
print(f"\nSTEP 7 — Predict a new house")
print(f"  A 2000 sq ft house is predicted to cost: ${predicted_price[0]:,.0f}")

# ============================================================
#  🎯 EXERCISES FOR STUDENTS
# ============================================================
# 1. Change `test_size` from 0.2 to 0.5. How does the R² change?
#    Why do you think this happens?
#
# 2. Change the noise range from (-20000, 20000) to (-80000, 80000).
#    What happens to RMSE and R²? What does this tell you about noisy data?
#
# 3. Add a second feature: number of bedrooms.
#    Hint: create a `bedrooms` array, stack it with `house_sizes` using
#    np.column_stack([house_sizes, bedrooms]) and retrain.
#
# 4. Try predicting the price of a 500 sq ft and a 5000 sq ft house.
#    Do the predictions seem reasonable? Why or why not?
# ============================================================
