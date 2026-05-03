"""
Insurance Charges Prediction — Full ML Pipeline
================================================
Dataset  : insurance.csv
Features : age, sex, bmi, children, smoker, region
Target   : charges

Steps
-----
1. Load & inspect data
2. Exploratory Data Analysis (EDA)
3. Preprocessing (encoding + scaling)
4. Train 3 models
5. Evaluate & compare
6. Save best model with pickle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & INSPECT
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv('insurance.csv')
df.rename(columns={'expenses': 'charges'}, inplace=True)   # normalise column name

print("=" * 55)
print("  DATASET OVERVIEW")
print("=" * 55)
print(f"Shape : {df.shape[0]} rows × {df.shape[1]} columns\n")
print(df.head())
print("\n--- Data Types ---")
print(df.dtypes)
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Statistical Summary ---")
print(df.describe())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EDA
# ─────────────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")

fig = plt.figure(figsize=(18, 22))
fig.suptitle(
    "Insurance Charges — Exploratory Data Analysis",
    fontsize=16, fontweight='bold', y=0.98
)

# Distribution: Age
ax1 = fig.add_subplot(4, 3, 1)
sns.histplot(df['age'], bins=20, kde=True, color='steelblue', ax=ax1)
ax1.set_title('Age Distribution')
ax1.set_xlabel('Age'); ax1.set_ylabel('Count')

# Distribution: BMI
ax2 = fig.add_subplot(4, 3, 2)
sns.histplot(df['bmi'], bins=20, kde=True, color='coral', ax=ax2)
ax2.set_title('BMI Distribution')
ax2.set_xlabel('BMI'); ax2.set_ylabel('Count')

# Distribution: Charges
ax3 = fig.add_subplot(4, 3, 3)
sns.histplot(df['charges'], bins=30, kde=True, color='mediumseagreen', ax=ax3)
ax3.set_title('Charges Distribution')
ax3.set_xlabel('Charges ($)'); ax3.set_ylabel('Count')

# Count: Sex
ax4 = fig.add_subplot(4, 3, 4)
sns.countplot(x='sex', data=df, palette='pastel', ax=ax4)
ax4.set_title('Sex Count')

# Count: Smoker
ax5 = fig.add_subplot(4, 3, 5)
sns.countplot(x='smoker', data=df, palette='Set2', ax=ax5)
ax5.set_title('Smoker Count')

# Count: Region
ax6 = fig.add_subplot(4, 3, 6)
sns.countplot(x='region', data=df, palette='Set3', ax=ax6)
ax6.set_title('Region Count')
ax6.tick_params(axis='x', rotation=15)

# Boxplot: Charges by Smoker (outlier detection)
ax7 = fig.add_subplot(4, 3, 7)
sns.boxplot(x='smoker', y='charges', data=df, palette='Set2', ax=ax7)
ax7.set_title('Charges by Smoker')

# Boxplot: Charges by Sex
ax8 = fig.add_subplot(4, 3, 8)
sns.boxplot(x='sex', y='charges', data=df, palette='pastel', ax=ax8)
ax8.set_title('Charges by Sex')

# Boxplot: Charges by Region
ax9 = fig.add_subplot(4, 3, 9)
sns.boxplot(x='region', y='charges', data=df, palette='Set3', ax=ax9)
ax9.set_title('Charges by Region')
ax9.tick_params(axis='x', rotation=15)

# Correlation Heatmap
ax10 = fig.add_subplot(4, 3, 10)
df_enc = df.copy()
for col in ['sex', 'smoker', 'region']:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col])
sns.heatmap(
    df_enc.corr(), annot=True, fmt='.2f',
    cmap='coolwarm', ax=ax10, square=True
)
ax10.set_title('Correlation Heatmap')

# Scatter: Age vs Charges
ax11 = fig.add_subplot(4, 3, 11)
sns.scatterplot(
    x='age', y='charges', hue='smoker',
    data=df, palette='Set1', alpha=0.6, ax=ax11
)
ax11.set_title('Age vs Charges')

# Scatter: BMI vs Charges
ax12 = fig.add_subplot(4, 3, 12)
sns.scatterplot(
    x='bmi', y='charges', hue='smoker',
    data=df, palette='Set1', alpha=0.6, ax=ax12
)
ax12.set_title('BMI vs Charges')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('eda_plots.png', dpi=130, bbox_inches='tight')
plt.close()
print("\n✅  EDA plots saved → eda_plots.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
df_model = df.copy()

le_sex    = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df_model['sex']    = le_sex.fit_transform(df_model['sex'])      # female=0, male=1
df_model['smoker'] = le_smoker.fit_transform(df_model['smoker'])  # no=0, yes=1
df_model['region'] = le_region.fit_transform(df_model['region'])  # 0-3

X = df_model.drop('charges', axis=1)
y = df_model['charges']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\n✅  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
    "Gradient Boosting Regressor": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, random_state=42
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — EVALUATE & COMPARE
# ─────────────────────────────────────────────────────────────────────────────
results = {}
print("\n" + "=" * 65)
print("  MODEL EVALUATION RESULTS")
print("=" * 65)
print(f"{'Model':<32} {'R² Score':>10} {'MAE ($)':>12} {'RMSE ($)':>12}")
print("-" * 68)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2   = r2_score(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results[name] = {'model': model, 'r2': r2, 'mae': mae, 'rmse': rmse}
    print(f"{name:<32} {r2:>10.4f} {mae:>12.2f} {rmse:>12.2f}")

best_name  = max(results, key=lambda n: results[n]['r2'])
best_model = results[best_name]['model']
print(f"\n🏆  Best Model : {best_name}")
print(f"    R² = {results[best_name]['r2']:.4f} | "
      f"MAE = ${results[best_name]['mae']:,.2f} | "
      f"RMSE = ${results[best_name]['rmse']:,.2f}")

# Model comparison chart
short_names = ["Lin. Reg.", "Rnd. Forest", "Grad. Boost"]
colors      = ['#4C72B0', '#55A868', '#C44E52']
fig2, axes  = plt.subplots(1, 3, figsize=(14, 5))
fig2.suptitle("Model Performance Comparison", fontsize=14, fontweight='bold')

for ax, metric, title, fmt in zip(
    axes,
    ['r2', 'mae', 'rmse'],
    ['R² Score  (↑ higher is better)',
     'MAE — Mean Abs. Error  (↓ lower is better)',
     'RMSE  (↓ lower is better)'],
    ['.4f', '.0f', '.0f']
):
    vals = [results[n][metric] for n in results]
    bars = ax.bar(short_names, vals, color=colors, edgecolor='white', width=0.55)
    ax.set_title(title, fontsize=10)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f'{v:{fmt}}',
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=130, bbox_inches='tight')
plt.close()
print("✅  Comparison chart saved → model_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('encoders.pkl', 'wb') as f:
    pickle.dump({'sex': le_sex, 'smoker': le_smoker, 'region': le_region}, f)

print(f"\n✅  Saved best_model.pkl  ({best_name})")
print("✅  Saved scaler.pkl")
print("✅  Saved encoders.pkl")
print("\n🎉  Pipeline complete!")
