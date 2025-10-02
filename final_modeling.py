import pandas as pd
data=pd.read_csv('final_data2.csv')

#KNN Imputation
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Select only the relevant features for imputation
impute_features = ['forty', 'height', 'weight']
impute_data = data[impute_features].copy()

# Step 1: Standardize the columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(impute_data)

# Step 2: KNN imputation on standardized data
imputer = KNNImputer(n_neighbors=5, weights='distance')
imputed_scaled = imputer.fit_transform(scaled_data)

# Step 3: Inverse transform to return to original scale
imputed_original = scaler.inverse_transform(imputed_scaled)

# Step 4: Replace only the 'forty' column in the original DataFrame
data['forty'] = imputed_original[:, 0]

# Strip whitespace, remove commas, and force convert to float
data['total_round_trip_miles'] = (
    data['total_round_trip_miles']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.strip()
    .astype(float)
)

# Final selected features
selected_features = [
     'targets', 'yards', 'avgCushion', 'avgIntendedAirYards',
    'avgSeparation', 'avgYAC', 'percentShareOfIntendedAirYards',
    'turf_percentage', 'total_round_trip_miles', 'total_snaps',
    'avg_opponent_pass_epa', 'sos_win_pct', 'age_season_start',
    'height', 'weight', 'forty', 'past_ir_count'
]

# Create X and y FIRST
X = data[selected_features]
y = data['next_season_ir']

#split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


#Logistic Regression Model
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score, accuracy_score
)

=ipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(solver='liblinear', max_iter=1000, random_state=42))
])

param_grid = {
    'lr__C': [0.0001, 0.001, 0.01, 0.1,]
}

grid = GridSearchCV(
    pipe,
    param_grid,
    scoring='f1',
    cv=StratifiedKFold(n_splits=5),
    n_jobs=-1,
    return_train_score=True
)
grid.fit(X_train, y_train)


best_pipeline = grid.best_estimator_
print(f"\n✅ Best C selected: {grid.best_params_['lr__C']}\n")

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print(f"PR AUC:    {pr_auc:.4f}")

    return {"accuracy": acc, "f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc}

print("🔍 Evaluation on Test Set:")
evaluate_model(best_pipeline, X_test, y_test)
from sklearn.metrics import balanced_accuracy_score
y_test_pred = best_pipeline.predict(X_test)
balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
print(f"Balanced Accuracy: {balanced_acc:.4f}")


#MLP Model
import numpy as np
import random
import tensorflow as tf

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, classification_report

from scikeras.wrappers import KerasClassifier
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# --- Reproducibility ---
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# --- Build model factory ---
def create_model(hidden_units=64, dropout_rate=0.5, l2_reg=1e-3, learning_rate=1e-3, n_features=None):
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(hidden_units, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

n_features = X_train.shape[1]

# --- Wrap in scikeras classifier ---
clf = KerasClassifier(
    model=create_model,
    # pass static args here; grid can override
    n_features=n_features,
    verbose=0
)

# --- Pipeline: SMOTE -> Standardize -> Keras ---
pipe = Pipeline(steps=[
    ("smote", SMOTE(random_state=42)),
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("clf", clf),
])

# --- Hyperparameter grid ---
param_grid = {
    # model architecture
    "clf__hidden_units": [32, 64, 128],
    "clf__dropout_rate": [0.2, 0.5],
    # regularization (typical small values)
    "clf__l2_reg": [1e-4, 1e-3, 1e-2],
    # optimization
    "clf__learning_rate": [1e-3, 1e-2],
    # training dynamics
    "clf__batch_size": [32, 64],
    "clf__epochs": [10, 20],
}

# --- CV & search ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="f1",          # or "f1_macro"/"average_precision" as needed
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=1
)

# If you want class weighting, pass via fit params on the final estimator:
fit_params = {
    "clf__class_weight": {0: 1.0, 1: 2.0},
    "clf__validation_split": 0.2
}

# --- Fit ---
grid.fit(X_train, y_train, **fit_params)

print("Best params:", grid.best_params_)
print("Best CV F1:", grid.best_score_)

# --- Evaluate on held-out test ---
y_pred = grid.best_estimator_.predict(X_test)
print("Test F1:", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import balanced_accuracy_score

balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {balanced_acc:.4f}")

#XGBoost Model
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, classification_report

# --- Base model ---
xgb_base = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_estimators=100,
    colsample_bytree=0.8,
    tree_method="hist",   
    n_jobs=-1
)

# --- Hyperparameter grid (as provided) ---
param_grid = {
    "learning_rate": [0.01, 0.1, 0.3],
    "max_depth": [3, 5, 7],
    "subsample": [0.5, 0.8, 1.0],
    "scale_pos_weight": [1, 2, 3, 5],
}

# --- CV setup & search ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring="f1",          
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV F1:", grid.best_score_)

# --- Test evaluation ---
best_xgb = grid.best_estimator_
y_pred = best_xgb.predict(X_test)

print("🔍 Test F1:", f1_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))
from sklearn.metrics import balanced_accuracy_score

# 📌 Compute Balanced Accuracy
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {balanced_acc:.4f}")

#SHAP for XGBoost Model
import shap
import numpy as np
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(xgb_clf)
shap_vals = explainer.shap_values(X_test)

mean_abs_shap = np.abs(shap_vals).mean(axis=0)
feature_names = X_test.columns if hasattr(X_test, "columns") else [f"f{i}" for i in range(mean_abs_shap.size)]

idx_sorted = np.argsort(mean_abs_shap)
sorted_shap = mean_abs_shap[idx_sorted]
sorted_feats = np.array(feature_names)[idx_sorted]


plt.figure(figsize=(8, 6))
plt.barh(sorted_feats, sorted_shap, color='red')
plt.xlabel('Mean |SHAP value| (Average impact on model output)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Mean Absolute SHAP Values for Feature Importance\nin XGBoost Model Predicting Next-Season IR Placement', fontsize=14)
plt.tight_layout()
plt.show()

#Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, classification_report

# --- Base model (use all columns in X_train / X_test) ---
rf_base = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

# --- Hyperparameter grid (as provided) ---
param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5, 10, 20, None],
    "class_weight": ["balanced"],   # handle class imbalance
}

# --- CV & search ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    scoring="f1",      # or "f1_macro"/"average_precision" as needed
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

# Fit on full training data (all features)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV F1:", grid.best_score_)

# --- Evaluate on test set ---
best_rf = grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print("Test F1:", f1_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

from sklearn.metrics import balanced_accuracy_score

# 📌 Compute Balanced Accuracy (default threshold 0.5)
balanced_acc_rf = balanced_accuracy_score(y_test, y_pred_rf)
print(f"Balanced Accuracy: {balanced_acc_rf:.4f}")



