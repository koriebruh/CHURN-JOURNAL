import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix,
                           accuracy_score, precision_score, recall_score, f1_score,
                           roc_curve, precision_recall_curve, auc)
from sklearn.calibration import calibration_curve
from scipy.stats import wilcoxon, friedmanchisquare
import joblib
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
import optuna
import dice_ml
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset

# Configuration for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')  # Changed from 'seaborn-v0_8-paper' to 'default'
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = "rill_final_enhanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/data", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)

print("="*80)
print("CUSTOMER CHURN PREDICTION ANALYSIS (ENHANCED FOR Q1-Q2 JOURNAL)")
print("="*80)
print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output directory: {OUTPUT_DIR}")


# === STEP 1: DATA LOADING AND EXPLORATION ===
print("\n" + "="*50)
print("STEP 1: DATA LOADING AND EXPLORATION")
print("="*50)

try:
    df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn_cleaned.csv")
    print(f"✓ Dataset loaded successfully")
    print(f"Dataset shape: {df.shape}")
    print(f"\nDataset Info:")
    print(f"- Total samples: {len(df)}")
    print(f"- Total features: {len(df.columns)}")
    print(f"- Missing values: {df.isnull().sum().sum()}")
except FileNotFoundError:
    print("❌ Dataset file not found. Please check the file path.")
    print("Make sure the dataset is in the 'dataset/' folder")
    exit()

# === STEP 2: DATA PREPROCESSING ===
print("\n" + "="*50)
print("STEP 2: DATA PREPROCESSING")
print("="*50)

# Normalize column names
df.columns = df.columns.str.strip().str.replace(" ", "_")
print("✓ Column names normalized")

# Handle TotalCharges conversion
print("Processing TotalCharges column...")
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_before = df['TotalCharges'].isnull().sum()
    print(f"Missing values in TotalCharges: {missing_before}")

    # Remove rows with missing values
    if missing_before > 0:
        df.dropna(inplace=True)
        print(f"✓ Removed {missing_before} rows with missing values")
else:
    print("TotalCharges column not found")

print(f"Final dataset shape: {df.shape}")

# Remove customer ID
if 'customerID' in df.columns:
    df.drop(['customerID'], axis=1, inplace=True)
    print("✓ Customer ID column removed")

# Analyze target variable distribution
if 'Churn' not in df.columns:
    print("❌ Churn column not found in dataset")
    print(f"Available columns: {list(df.columns)}")
    exit()

target_dist = df['Churn'].value_counts()
print(f"\nTarget Variable Distribution:")
for label, count in target_dist.items():
    percentage = (count / len(df)) * 100
    print(f"- {label}: {count} ({percentage:.2f}%)")

# Create class distribution plot
plt.figure(figsize=(8, 6))
target_dist.plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.title('Target Variable Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Churn Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
for i, v in enumerate(target_dist.values):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/target_distribution.png", dpi=300, bbox_inches='tight')
plt.show()



# === STEP 3: FEATURE ENGINEERING ===
print("\n" + "="*50)
print("STEP 3: FEATURE ENGINEERING")
print("="*50)

# Save a copy of the dataset before feature engineering for comparative analysis
df_no_fe = df.copy()
print("✓ Saved dataset copy without feature engineering")

# Check if required columns exist before feature engineering
required_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"⚠ Missing required columns for feature engineering: {missing_cols}")
    print("Skipping advanced feature engineering...")
else:
    # New feature: Engagement Score
    df['Engagement_Score'] = df['tenure'] * df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    print("✓ Added Engagement_Score feature")

    # New feature: Churn Risk Score
    df['Churn_Risk_Score'] = df['tenure'].apply(lambda x: 1 if x < 12 else 0) * df['MonthlyCharges']
    print("✓ Added Churn_Risk_Score feature")

# Service utilization (if service columns exist)
service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
existing_service_cols = [col for col in service_cols if col in df.columns]
if existing_service_cols:
    df['Service_Utilization'] = df[existing_service_cols].eq('Yes').sum(axis=1)
    print("✓ Added Service_Utilization feature")

# Payment reliability (if payment method column exists)
if 'PaymentMethod' in df.columns:
    df['Payment_Reliability'] = df['PaymentMethod'].apply(
        lambda x: 1 if x in ['Bank transfer (automatic)', 'Credit card (automatic)'] else 0
    )
    print("✓ Added Payment_Reliability feature")

# One-hot encoding for categorical variables
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
if 'Churn' in categorical_features:
    categorical_features.remove('Churn')

print(f"Categorical features to encode: {categorical_features}")
if categorical_features:
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    print(f"✓ One-hot encoding completed")
else:
    df_encoded = df.copy()
    print("✓ No categorical features to encode")

print(f"Features after encoding: {df_encoded.shape[1] - 1}")


#### TAMBAHAN: Prepare dataset without feature engineering
categorical_features_no_fe = df_no_fe.select_dtypes(include=['object']).columns.tolist()
if 'Churn' in categorical_features_no_fe:
    categorical_features_no_fe.remove('Churn')

if categorical_features_no_fe:
    df_encoded_no_fe = pd.get_dummies(df_no_fe, columns=categorical_features_no_fe, drop_first=True)
    print(f"✓ One-hot encoding completed for dataset without feature engineering")
else:
    df_encoded_no_fe = df_no_fe.copy()
    print("✓ No categorical features to encode for dataset without feature engineering")

# TAMBAHAN: Prepare features and target for no feature engineering
X_no_fe = df_encoded_no_fe.drop("Churn", axis=1)
y_no_fe = (df_encoded_no_fe["Churn"] == "Yes").astype(int)
print(f"Final feature matrix shape (without FE): {X_no_fe.shape}")
####

# Prepare features and target
X = df_encoded.drop("Churn", axis=1)
y = (df_encoded["Churn"] == "Yes").astype(int)
print(f"Final feature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")

# Feature correlation analysis (only if we have multiple features)
if X.shape[1] > 1:
    plt.figure(figsize=(15, 12))
    corr_matrix = X.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()



# === STEP 4: DATA SCALING AND SPLITTING ===
print("\n" + "="*50)
print("STEP 4: DATA SCALING AND SPLITTING")
print("="*50)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✓ Feature scaling completed")

# Save scaler
joblib.dump(scaler, f"{OUTPUT_DIR}/models/scaler.pkl")
print("✓ Scaler saved")

# Compare imbalance handling methods
imbalance_methods = {
    'SMOTE': SMOTE(random_state=RANDOM_SEED),
    'SMOTEENN': SMOTEENN(random_state=RANDOM_SEED),
    'RandomUnderSampler': RandomUnderSampler(random_state=RANDOM_SEED),
    'ADASYN': ADASYN(random_state=RANDOM_SEED)
}

imbalance_results = {}
for name, sampler in imbalance_methods.items():
    try:
        print(f"Applying {name}...")
        X_res, y_res = sampler.fit_resample(X_scaled, y)
        X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(
            X_res, y_res, stratify=y_res, test_size=0.25, random_state=RANDOM_SEED
        )
        imbalance_results[name] = {'shape': X_res.shape, 'target_dist': np.bincount(y_res)}
        print(f"✓ {name} - Resampled shape: {X_res.shape}, Target distribution: {np.bincount(y_res)}")
    except Exception as e:
        print(f"⚠ {name} failed: {str(e)}")
        continue

# Use SMOTE for main analysis
try:
    smote = SMOTE(random_state=RANDOM_SEED)
    X_scaled_res, y_res = smote.fit_resample(X_scaled, y)
    print(f"✓ SMOTE applied - Resampled dataset shape: {X_scaled_res.shape}")
    print(f"Resampled target distribution: {np.bincount(y_res)}")
except Exception as e:
    print(f"⚠ SMOTE failed, using original data: {str(e)}")
    X_scaled_res, y_res = X_scaled, y

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_res, y_res, stratify=y_res, test_size=0.25, random_state=RANDOM_SEED
)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Training set class distribution: {np.bincount(y_train)}")
print(f"Testing set class distribution: {np.bincount(y_test)}")

# TAMBAHAN: Feature scaling for dataset without feature engineering
X_scaled_no_fe = scaler.fit_transform(X_no_fe)
print("✓ Feature scaling completed for dataset without feature engineering")

# TAMBAHAN: Train-test split for no feature engineering
X_train_no_fe, X_test_no_fe, y_train_no_fe, y_test_no_fe = train_test_split(
    X_scaled_no_fe, y_no_fe, stratify=y_no_fe, test_size=0.25, random_state=RANDOM_SEED
)
print(f"Training set shape (No FE): {X_train_no_fe.shape}")
print(f"Testing set shape (No FE): {X_test_no_fe.shape}")

# TAMBAHAN: Train-test split for feature engineering without SMOTE
X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(
    X_scaled, y, stratify=y, test_size=0.25, random_state=RANDOM_SEED
)
print(f"Training set shape (FE only): {X_train_fe.shape}")
print(f"Testing set shape (FE only): {X_test_fe.shape}")


# === STEP 4.3: Visualize target distribution after SMOTE ===
print("\n" + "="*50)
print("STEP 4.3: VISUALIZE TARGET DISTRIBUTION AFTER SMOTE")
print("="*50)

# Create target distribution after SMOTE (sama seperti Step 2)
# Convert numeric to text labels to match Step 2 format
y_res_labels = pd.Series(['No' if x == 0 else 'Yes' for x in y_res])
target_dist = y_res_labels.value_counts()
print(f"\nTarget Variable Distribution After SMOTE:")
for label, count in target_dist.items():
    percentage = (count / len(y_res)) * 100
    print(f"- {label}: {count} ({percentage:.2f}%)")

# Create class distribution plot (identik dengan Step 2)
plt.figure(figsize=(8, 6))
target_dist.plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.title('Target Distribution After SMOTE', fontsize=14, fontweight='bold')
plt.xlabel('Churn Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
for i, v in enumerate(target_dist.values):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/resampled_target_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# === STEP 4.5: COMPARATIVE MODEL EVALUATION ===
print("\n" + "="*50)
print("STEP 4.5: COMPARATIVE MODEL EVALUATION")
print("="*50)

# Define function to evaluate model
def evaluate_model(model, X_train, y_train, X_test, y_test, scenario_name):
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'Scenario': scenario_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'ROC_AUC': roc_auc_score(y_test, y_pred_proba)
        }

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {scenario_name}', fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plots/confusion_matrix_{scenario_name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.show()

        return metrics
    except Exception as e:
        print(f"⚠ Error evaluating {scenario_name}: {str(e)}")
        return None

# Placeholder for best model (to be determined after tuning)
comparative_results = []

# Define scenarios
scenarios = [
    ('No Feature Engineering', X_train_no_fe, y_train_no_fe, X_test_no_fe, y_test_no_fe),
    ('Feature Engineering', X_train_fe, y_train_fe, X_test_fe, y_test_fe),
    ('Feature Engineering + SMOTE', X_train, y_train, X_test, y_test)
]

# Note: Actual evaluation will be done after selecting the best model in Step 5
print("✓ Scenarios defined for comparative evaluation")


# === STEP 5: MODEL DEFINITION AND HYPERPARAMETER TUNING ===
print("\n" + "="*50)
print("STEP 5: MODEL DEFINITION AND HYPERPARAMETER TUNING")
print("="*50)

# Define models with expanded hyperparameter grids
models_config = {
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "params": {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5]}
    },
    "GBM": {
        "model": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "params": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric='logloss', random_state=RANDOM_SEED, verbosity=0),
        "params": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=RANDOM_SEED),
        "params": {'n_estimators': [100, 200], 'max_depth': [10, None], 'min_samples_split': [2, 5]}
    },
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "params": {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
    },
    "SVC": {
        "model": SVC(probability=True, random_state=RANDOM_SEED),
        "params": {'C': [0.1, 1], 'kernel': ['rbf', 'linear']}
    },
    "NeuralNet": {
        "model": MLPClassifier(max_iter=200, random_state=RANDOM_SEED, early_stopping=True),
        "params": {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001]}
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=RANDOM_SEED),
        "params": {'n_estimators': [50, 100], 'learning_rate': [0.1, 1.0]}
    },
    "LightGBM": {
        "model": LGBMClassifier(random_state=RANDOM_SEED, verbosity=-1),
        "params": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
    },
    "CatBoost": {
        "model": CatBoostClassifier(random_state=RANDOM_SEED, verbose=0),
        "params": {'iterations': [100, 200], 'learning_rate': [0.01, 0.1], 'depth': [3, 5]}
    }
}

# Bayesian optimization for hyperparameter tuning
tuned_models = {}
tuning_results = []

def objective(trial, model_name, X_train, y_train):
    params = {}
    try:
        if model_name == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            }
            model = XGBClassifier(**params, random_state=RANDOM_SEED, eval_metric='logloss', verbosity=0)
        elif model_name == "LightGBM":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10)
            }
            model = LGBMClassifier(**params, random_state=RANDOM_SEED, verbosity=-1)
        elif model_name == "CatBoost":
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 3, 10)
            }
            model = CatBoostClassifier(**params, random_state=RANDOM_SEED, verbose=0)
        else:
            model = models_config[model_name]["model"]
            params = {key: trial.suggest_categorical(key, values)
                     for key, values in models_config[model_name]["params"].items()}
            model.set_params(**params)

        score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
        return score
    except Exception as e:
        print(f"Error in objective for {model_name}: {str(e)}")
        return 0.5  # Return neutral score on error

print("Performing Bayesian hyperparameter tuning...")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce optuna output

for name in models_config.keys():
    print(f"Tuning {name}...")
    try:
        study = optuna.create_study(direction='maximize')
       # Timeout + trial untuk semua model (light tuning)
        if name in ["DecisionTree", "LogisticRegression"]:
            study.optimize(lambda trial: objective(trial, name, X_train, y_train), n_trials=10, timeout=120)
        elif name in ["GBM", "RandomForest", "AdaBoost"]:
            study.optimize(lambda trial: objective(trial, name, X_train, y_train), n_trials=10, timeout=240)
        elif name in ["XGBoost", "LightGBM", "CatBoost", "SVC", "NeuralNet"]:
            study.optimize(lambda trial: objective(trial, name, X_train, y_train), n_trials=15, timeout=360)

        best_params = study.best_params
        model = models_config[name]["model"]
        model.set_params(**best_params)
        tuned_models[name] = model
        print(f"✓ {name} best score: {study.best_value:.4f}")
        tuning_results.append({'Model': name, 'Best_Score': study.best_value, 'Best_Params': best_params})
    except Exception as e:
        print(f"⚠ Error tuning {name}: {str(e)}")
        # Use default model if tuning fails
        tuned_models[name] = models_config[name]["model"]
        tuning_results.append({'Model': name, 'Best_Score': 0.5, 'Best_Params': {}})

# Save tuning results
tuning_df = pd.DataFrame(tuning_results)
tuning_df.to_csv(f"{OUTPUT_DIR}/data/hyperparameter_tuning_results.csv", index=False)
print("✓ Hyperparameter tuning results saved")

# TAMBAHAN: Select best model for comparative evaluation
best_model_name = max(tuned_models, key=lambda k: tuning_df[tuning_df['Model'] == k]['Best_Score'].iloc[0])
best_model = tuned_models[best_model_name]
print(f"Selected best model for comparative evaluation: {best_model_name}")

# TAMBAHAN: Evaluate best model on all scenarios
print("Evaluating best model on different scenarios...")
comparative_results = []  # Inisialisasi di sini agar tersedia untuk Langkah 13

# Define scenarios (assuming these variables are defined in Step 4)
scenarios = [
    ('No Feature Engineering', X_train_no_fe, y_train_no_fe, X_test_no_fe, y_test_no_fe),
    ('Feature Engineering', X_train_fe, y_train_fe, X_test_fe, y_test_fe),
    ('Feature Engineering + SMOTE', X_train, y_train, X_test, y_test)
]

for scenario_name, X_tr, y_tr, X_te, y_te in scenarios:
    try:
        best_model.fit(X_tr, y_tr)
        y_pred = best_model.predict(X_te)
        y_pred_proba = best_model.predict_proba(X_te)[:, 1]

        metrics = {
            'Scenario': scenario_name,
            'Accuracy': accuracy_score(y_te, y_pred),
            'Precision': precision_score(y_te, y_pred, zero_division=0),
            'Recall': recall_score(y_te, y_pred, zero_division=0),
            'F1': f1_score(y_te, y_pred, zero_division=0),
            'ROC_AUC': roc_auc_score(y_te, y_pred_proba)
        }

        # Confusion Matrix
        cm = confusion_matrix(y_te, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {scenario_name}', fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plots/confusion_matrix_{scenario_name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.show()

        comparative_results.append(metrics)
        print(f"✓ {scenario_name} evaluation completed")
        print(f"Metrics: {metrics}")
    except Exception as e:
        print(f"⚠ Error evaluating {scenario_name}: {str(e)}")

# TAMBAHAN: Save comparative results
comparative_results_df = pd.DataFrame(comparative_results)
comparative_results_df.to_csv(f"{OUTPUT_DIR}/data/comparative_model_results.csv", index=False)
print("✓ Comparative model results saved")

# TAMBAHAN: Visualize comparative results
plt.figure(figsize=(12, 6))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']
for metric in metrics_to_plot:
    plt.plot(comparative_results_df['Scenario'], comparative_results_df[metric], marker='o', label=metric)
plt.title('Model Performance Across Scenarios', fontsize=14, fontweight='bold')
plt.xlabel('Scenario')
plt.ylabel('Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/comparative_model_performance.png", dpi=300, bbox_inches='tight')
plt.show()


# === STEP 6: CROSS-VALIDATION EVALUATION ===
print("\n" + "="*50)
print("STEP 6: CROSS-VALIDATION EVALUATION")
print("="*50)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)  # Reduced from 10 to 5
cv_results = {}
metrics_results = {}
cv_metrics_per_fold = {}

print("Performing 5-fold cross-validation...")
print("-" * 80)
print(f"{'Model':<15} {'ROC-AUC':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
print("-" * 80)

for name, model in tuned_models.items():
    try:
        cv_metrics_per_fold[name] = {'roc_auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

        for train_idx, val_idx in cv.split(X_scaled_res, y_res):
            X_cv_train, X_cv_val = X_scaled_res[train_idx], X_scaled_res[val_idx]
            y_cv_train, y_cv_val = y_res[train_idx], y_res[val_idx]

            model.fit(X_cv_train, y_cv_train)
            y_pred = model.predict(X_cv_val)
            y_pred_proba = model.predict_proba(X_cv_val)[:, 1]

            cv_metrics_per_fold[name]['roc_auc'].append(roc_auc_score(y_cv_val, y_pred_proba))
            cv_metrics_per_fold[name]['accuracy'].append(accuracy_score(y_cv_val, y_pred))
            cv_metrics_per_fold[name]['precision'].append(precision_score(y_cv_val, y_pred, zero_division=0))
            cv_metrics_per_fold[name]['recall'].append(recall_score(y_cv_val, y_pred, zero_division=0))
            cv_metrics_per_fold[name]['f1'].append(f1_score(y_cv_val, y_pred, zero_division=0))

        roc_scores = np.array(cv_metrics_per_fold[name]['roc_auc'])
        acc_scores = np.array(cv_metrics_per_fold[name]['accuracy'])
        prec_scores = np.array(cv_metrics_per_fold[name]['precision'])
        rec_scores = np.array(cv_metrics_per_fold[name]['recall'])
        f1_scores = np.array(cv_metrics_per_fold[name]['f1'])

        cv_results[name] = roc_scores
        metrics_results[name] = {
            'roc_auc': roc_scores,
            'accuracy': acc_scores,
            'precision': prec_scores,
            'recall': rec_scores,
            'f1': f1_scores
        }

        print(f"{name:<15} {roc_scores.mean():.4f}±{roc_scores.std():.3f} "
              f"{acc_scores.mean():.4f}±{acc_scores.std():.3f} "
              f"{prec_scores.mean():.4f}±{prec_scores.std():.3f} "
              f"{rec_scores.mean():.4f}±{rec_scores.std():.3f} "
              f"{f1_scores.mean():.4f}±{f1_scores.std():.3f}")
    except Exception as e:
        print(f"⚠ Error evaluating {name}: {str(e)}")
        continue

# Save per-fold metrics
if cv_metrics_per_fold:
    cv_metrics_df = pd.DataFrame({
        'Model': [name for name in cv_metrics_per_fold for _ in range(5)],
        'Fold': list(range(1, 6)) * len(cv_metrics_per_fold),
        'ROC_AUC': [score for name in cv_metrics_per_fold for score in cv_metrics_per_fold[name]['roc_auc']],
        'Accuracy': [score for name in cv_metrics_per_fold for score in cv_metrics_per_fold[name]['accuracy']],
        'Precision': [score for name in cv_metrics_per_fold for score in cv_metrics_per_fold[name]['precision']],
        'Recall': [score for name in cv_metrics_per_fold for score in cv_metrics_per_fold[name]['recall']],
        'F1': [score for name in cv_metrics_per_fold for score in cv_metrics_per_fold[name]['f1']]
    })
    cv_metrics_df.to_csv(f"{OUTPUT_DIR}/data/cv_metrics_per_fold.csv", index=False)
    print("✓ Cross-validation metrics per fold saved")

# Create performance comparison plot
if metrics_results:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics = ['roc_auc', 'accuracy', 'precision', 'recall']
    titles = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx//2, idx%2]
        data = [metrics_results[model][metric] for model in tuned_models.keys() if model in metrics_results]
        labels = [model for model in tuned_models.keys() if model in metrics_results]

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            ax.set_title(f'{title} Distribution Across Models', fontweight='bold')
            ax.set_ylabel(title)
            ax.tick_params(axis='x', rotation=45)
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


# === STEP 7: STATISTICAL SIGNIFICANCE TESTING ===
print("\n" + "="*50)
print("STEP 7: STATISTICAL SIGNIFICANCE TESTING")
print("="*50)

if cv_results:
    # Wilcoxon signed-rank test
    print("Wilcoxon Signed-Rank Test Results:")
    print("-" * 50)
    best_model = max(cv_results.keys(), key=lambda k: cv_results[k].mean())
    print(f"Best performing model (baseline): {best_model}")
    print(f"Baseline ROC-AUC: {cv_results[best_model].mean():.4f} ± {cv_results[best_model].std():.4f}")
    print()

    baseline_scores = cv_results[best_model]
    significance_results = {}
    for name, scores in cv_results.items():
        if name != best_model:
            try:
                stat, p_value = wilcoxon(baseline_scores, scores, alternative='two-sided')
                significance_results[name] = p_value
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                print(f"{name:<15} vs {best_model}: p-value = {p_value:.6f} {significance}")
            except Exception as e:
                print(f"⚠ Error in Wilcoxon test for {name}: {str(e)}")
                significance_results[name] = 1.0

    # Friedman test
    print(f"\nFriedman Test (Multiple Model Comparison):")
    try:
        all_scores = [scores for scores in cv_results.values()]
        friedman_stat, friedman_p = friedmanchisquare(*all_scores)
        print(f"Friedman statistic: {friedman_stat:.4f}")
        print(f"p-value: {friedman_p:.6f}")
    except Exception as e:
        print(f"⚠ Error in Friedman test: {str(e)}")
else:
    print("No cross-validation results available for statistical testing")
    best_model = list(tuned_models.keys())[0]  # Use first model as default
    significance_results = {}

# === STEP 8: FINAL MODEL EVALUATION ===
print("\n" + "="*50)
print("STEP 8: FINAL MODEL EVALUATION")
print("="*50)

# Train the best model
final_model = tuned_models[best_model]
# Convert to DataFrame if needed
if not isinstance(X_train, pd.DataFrame):
    X_train = pd.DataFrame(X_train, columns=X.columns)

if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test, columns=X.columns)


# final_model.fit(X_train, y_train)
# FIX INI DULU
feature_names = X.columns
X_train = pd.DataFrame(X_train, columns=feature_names)
X_test = pd.DataFrame(X_test, columns=feature_names)

# Baru model training kayak biasa
final_model = tuned_models[best_model]
final_model.fit(X_train, y_train)


# Predictions
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

# Comprehensive metrics
test_metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}

print(f"Final Model Performance ({best_model}):")
print("-" * 40)
for metric, value in test_metrics.items():
    print(f"{metric.upper():<10}: {value:.4f}")

print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title(f'Confusion Matrix - {best_model}', fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {best_model}', fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# Lanjutan dari kode sebelumnya - mulai dari ROC Curve yang terpotong
plt.savefig(f"{OUTPUT_DIR}/plots/roc_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
plt.axhline(y=y_test.mean(), color='red', linestyle='--', label=f'Baseline (P = {y_test.mean():.4f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve - {best_model}', fontweight='bold')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/precision_recall_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# Calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)
plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f'{best_model}')
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.ylabel('Fraction of positives')
plt.xlabel('Mean predicted probability')
plt.title(f'Calibration Plot - {best_model}', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/calibration_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# Save final model
joblib.dump(final_model, f"{OUTPUT_DIR}/models/final_model.pkl")
print("✓ Final model saved")

# === STEP 9: FEATURE IMPORTANCE ANALYSIS ===
print("\n" + "="*50)
print("STEP 9: FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Prepare X_test_df for consistency
X_test_df = pd.DataFrame(X_test, columns=X.columns)
print("X_test_df columns:", X_test_df.columns.tolist())
print("X_test_df shape:", X_test_df.shape)
print(f"Model used: {type(final_model).__name__}")

try:
    # Use SHAP for feature importance
    import shap
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_df)
    shap_importance = np.abs(shap_values).mean(axis=0)
    total_importance = shap_importance.sum()
    importance_df = pd.DataFrame({
        'feature': X_test_df.columns,
        'importance': shap_importance / total_importance  # Normalized importance
    }).sort_values('importance', ascending=False)

    print("Top 10 Most Important Features:")
    print("-" * 40)
    print(importance_df.head(10).to_string(index=False))

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance (Normalized)')
    plt.title(f'Top 15 Feature Importances - {type(final_model).__name__}', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Save feature importance
    importance_df.to_csv(f"{OUTPUT_DIR}/data/feature_importance.csv", index=False)
    print("✓ Feature importance saved")
except Exception as e:
    print(f"⚠ Feature importance analysis failed: {str(e)}")
    print("Continuing without feature importance...")
    importance_df = pd.DataFrame()

 # === STEP 10: SHAP ANALYSIS ===
print("\n" + "="*50)
print("STEP 10: SHAP ANALYSIS")
print("="*50)
try:
    print("Generating SHAP explanations...")
    if not isinstance(X_test, pd.DataFrame):
        X_test_df = pd.DataFrame(X_test, columns=X.columns)
    else:
        X_test_df = X_test

    print("X_test_df shape:", X_test_df.shape)
    print("Model used:", type(final_model).__name__)

    # Prepare training data for KernelExplainer (if needed)
    if not 'X_train' in globals():
        print("⚠ X_train not found, using X_test_df[:100] as fallback for KernelExplainer.")
        X_train_df = X_test_df[:100].copy()
    else:
        if not isinstance(X_train, pd.DataFrame):
            X_train_df = pd.DataFrame(X_train, columns=X.columns)
        else:
            X_train_df = X_train

    if best_model.__class__.__name__ in ['LGBMClassifier', 'XGBClassifier', 'GradientBoostingClassifier', 'RandomForestClassifier', 'CatBoostClassifier']:
        explainer = shap.TreeExplainer(final_model, feature_names=X.columns.tolist())
        shap_values = explainer.shap_values(X_test_df[:500], check_additivity=False)
        shap_values_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
    else:
        explainer = shap.KernelExplainer(
            lambda x: final_model.predict_proba(x)[:, 1],
            X_train_df[:100],
            feature_names=X.columns.tolist()
        )
        shap_values = explainer.shap_values(X_test_df[:100])
        shap_values_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_plot, X_test_df[:len(shap_values_plot)],
                     feature_names=X.columns.tolist(), show=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/shap_summary.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_plot, X_test_df[:len(shap_values_plot)],
                     feature_names=X.columns.tolist(), plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/shap_bar.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ SHAP analysis completed")

except Exception as e:
    print(f"⚠ SHAP analysis failed: {str(e)}")
    print("Continuing without SHAP analysis...")


# === STEP 11: BIAS AND FAIRNESS ANALYSIS ===
print("\n" + "="*50)
print("STEP 11: BIAS AND FAIRNESS ANALYSIS")
print("="*50)
try:
    # Check if we have demographic features for fairness analysis
    demographic_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
    available_demo_features = [f for f in demographic_features if f in df.columns or
                              any(col.startswith(f) for col in df.columns)]

    if available_demo_features:
        print(f"Available demographic features: {available_demo_features}")

        # Create fairness analysis for SeniorCitizen if available
        if 'SeniorCitizen' in df.columns:
            # Get original test data (before resampling)
            X_test_orig, y_test_orig = X_scaled, y  # Use original data
            X_test_orig_df = pd.DataFrame(X_test_orig, columns=X.columns, index=df.index)
            y_test_orig_series = pd.Series(y_test_orig, index=df.index)
            y_pred_orig = final_model.predict(X_test_orig_df)

            # Create demographic parity analysis
            senior_mask = df['SeniorCitizen'] == 1

            # Calculate fairness metrics
            senior_pred_rate = y_pred_orig[senior_mask].mean()
            non_senior_pred_rate = y_pred_orig[~senior_mask].mean()
            demographic_parity = abs(senior_pred_rate - non_senior_pred_rate)

            print(f"Demographic Parity Analysis:")
            print(f"- Senior citizens positive prediction rate: {senior_pred_rate:.4f}")
            print(f"- Non-senior citizens positive prediction rate: {non_senior_pred_rate:.4f}")
            print(f"- Demographic parity difference: {demographic_parity:.4f}")

            # Equal opportunity analysis
            senior_tpr = recall_score(y_test_orig_series[senior_mask], y_pred_orig[senior_mask], zero_division=0)
            non_senior_tpr = recall_score(y_test_orig_series[~senior_mask], y_pred_orig[~senior_mask], zero_division=0)
            equalized_odds = abs(senior_tpr - non_senior_tpr)

            print(f"Equal Opportunity Analysis:")
            print(f"- Senior citizens TPR: {senior_tpr:.4f}")
            print(f"- Non-senior citizens TPR: {non_senior_tpr:.4f}")
            print(f"- Equalized odds difference: {equalized_odds:.4f}")

        else:
            print("⚠ SeniorCitizen column not found for fairness analysis")
    else:
        print("⚠ No demographic features found for fairness analysis")

except Exception as e:
    print(f"⚠ Fairness analysis failed: {str(e)}")


import dice_ml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# === STEP 12: COUNTERFACTUAL EXPLANATIONS WITH BULK EVALUATION ===
print("\n" + "="*50)
print("STEP 12: COUNTERFACTUAL EXPLANATIONS WITH BULK EVALUATION")
print("="*50)

def evaluate_counterfactual_quality(original_instances, counterfactuals_list, model, feature_names):
    """
    Evaluate the quality of counterfactuals using multiple metrics
    """
    evaluation_results = {
        'validity': [],
        'proximity': [],
        'diversity': [],
        'sparsity': [],
        'feasibility': []
    }

    for i, cf_list in enumerate(counterfactuals_list):
        if cf_list is not None and hasattr(cf_list, 'final_cfs_df'):
            original = original_instances.iloc[i]
            cfs = cf_list.final_cfs_df

            # Validity: Check if counterfactuals actually flip the prediction
            original_pred = model.predict(original.values.reshape(1, -1))[0]
            cf_preds = model.predict(cfs.iloc[:, :-1].values)  # Exclude target column
            validity = np.mean(cf_preds != original_pred)

            # Proximity: Average L2 distance from original
            distances = []
            for _, cf_row in cfs.iterrows():
                cf_features = cf_row.iloc[:-1].values  # Exclude target
                distance = np.linalg.norm(original.values - cf_features)
                distances.append(distance)
            proximity = np.mean(distances) if distances else float('inf')

            # Diversity: Average pairwise distance between counterfactuals
            diversity_scores = []
            cf_features = cfs.iloc[:, :-1].values
            for j in range(len(cf_features)):
                for k in range(j+1, len(cf_features)):
                    diversity_scores.append(np.linalg.norm(cf_features[j] - cf_features[k]))
            diversity = np.mean(diversity_scores) if diversity_scores else 0

            # Sparsity: Average number of features changed
            sparsity_scores = []
            for _, cf_row in cfs.iterrows():
                cf_features = cf_row.iloc[:-1].values
                changed_features = np.sum(np.abs(original.values - cf_features) > 1e-6)
                sparsity_scores.append(changed_features)
            sparsity = np.mean(sparsity_scores) if sparsity_scores else 0

            # Feasibility: Check if values are within reasonable bounds
            feasibility = 1.0  # Simplified - in practice, you'd check domain constraints

            evaluation_results['validity'].append(validity)
            evaluation_results['proximity'].append(proximity)
            evaluation_results['diversity'].append(diversity)
            evaluation_results['sparsity'].append(sparsity)
            evaluation_results['feasibility'].append(feasibility)
        else:
            # Handle failed counterfactual generation
            evaluation_results['validity'].append(0)
            evaluation_results['proximity'].append(float('inf'))
            evaluation_results['diversity'].append(0)
            evaluation_results['sparsity'].append(0)
            evaluation_results['feasibility'].append(0)

    return evaluation_results

def bulk_counterfactual_analysis(X_test, model, dice_explainer, n_samples=50, n_counterfactuals=3):
    """
    Perform bulk counterfactual analysis on multiple test instances
    """
    print(f"\nPerforming bulk counterfactual analysis on {n_samples} instances...")

    # Select random sample of test instances
    sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    sample_instances = X_test.iloc[sample_indices]

    counterfactuals_list = []
    successful_generations = 0
    failed_generations = 0

    for i, (idx, instance) in enumerate(sample_instances.iterrows()):
        try:
            # Generate counterfactuals for this instance
            query_instance = instance.to_frame().T
            cf_result = dice_explainer.generate_counterfactuals(
                query_instance,
                total_CFs=n_counterfactuals,
                desired_class=1
            )
            counterfactuals_list.append(cf_result.cf_examples_list[0])
            successful_generations += 1

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(sample_instances)} instances...")

        except Exception as e:
            print(f"Failed to generate counterfactuals for instance {i}: {str(e)}")
            counterfactuals_list.append(None)
            failed_generations += 1

    print(f"✓ Bulk analysis completed:")
    print(f"  - Successful generations: {successful_generations}")
    print(f"  - Failed generations: {failed_generations}")
    print(f"  - Success rate: {successful_generations/len(sample_instances)*100:.1f}%")

    return sample_instances, counterfactuals_list

def create_comprehensive_visualizations(evaluation_results, sample_instances, counterfactuals_list, OUTPUT_DIR):
    """
    Create comprehensive visualizations for bulk counterfactual evaluation
    """
    # 1. Quality Metrics Distribution
    plt.figure(figsize=(20, 12))

    # Subplot 1: Validity Distribution
    plt.subplot(2, 3, 1)
    plt.hist(evaluation_results['validity'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(evaluation_results['validity']), color='red', linestyle='--',
                label=f'Mean: {np.mean(evaluation_results["validity"]):.3f}')
    plt.title('Counterfactual Validity Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Validity Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Proximity Distribution
    plt.subplot(2, 3, 2)
    proximity_filtered = [p for p in evaluation_results['proximity'] if p != float('inf')]
    if proximity_filtered:
        plt.hist(proximity_filtered, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(np.mean(proximity_filtered), color='red', linestyle='--',
                    label=f'Mean: {np.mean(proximity_filtered):.3f}')
    plt.title('Counterfactual Proximity Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Average L2 Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: Diversity Distribution
    plt.subplot(2, 3, 3)
    plt.hist(evaluation_results['diversity'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(np.mean(evaluation_results['diversity']), color='red', linestyle='--',
                label=f'Mean: {np.mean(evaluation_results["diversity"]):.3f}')
    plt.title('Counterfactual Diversity Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Average Pairwise Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 4: Sparsity Distribution
    plt.subplot(2, 3, 4)
    plt.hist(evaluation_results['sparsity'], bins=20, alpha=0.7, color='purple', edgecolor='black')
    plt.axvline(np.mean(evaluation_results['sparsity']), color='red', linestyle='--',
                label=f'Mean: {np.mean(evaluation_results["sparsity"]):.3f}')
    plt.title('Counterfactual Sparsity Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Changed Features')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 5: Quality Metrics Summary
    plt.subplot(2, 3, 5)
    metrics_data = pd.DataFrame({
        'Metric': ['Validity', 'Proximity', 'Diversity', 'Sparsity'],
        'Mean': [
            np.mean(evaluation_results['validity']),
            np.mean([p for p in evaluation_results['proximity'] if p != float('inf')]),
            np.mean(evaluation_results['diversity']),
            np.mean(evaluation_results['sparsity'])
        ],
        'Std': [
            np.std(evaluation_results['validity']),
            np.std([p for p in evaluation_results['proximity'] if p != float('inf')]),
            np.std(evaluation_results['diversity']),
            np.std(evaluation_results['sparsity'])
        ]
    })

    x_pos = np.arange(len(metrics_data))
    plt.bar(x_pos, metrics_data['Mean'], yerr=metrics_data['Std'],
            capsize=5, alpha=0.7, color=['skyblue', 'lightgreen', 'orange', 'purple'])
    plt.xticks(x_pos, metrics_data['Metric'], rotation=45)
    plt.title('Quality Metrics Summary', fontsize=12, fontweight='bold')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)

    # Subplot 6: Success Rate Pie Chart
    plt.subplot(2, 3, 6)
    successful = sum(1 for cf in counterfactuals_list if cf is not None)
    failed = len(counterfactuals_list) - successful
    plt.pie([successful, failed], labels=['Successful', 'Failed'],
            autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    plt.title('Counterfactual Generation Success Rate', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/bulk_counterfactual_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Feature Importance Heatmap
    plt.figure(figsize=(12, 8))
    feature_changes = []

    for i, cf_list in enumerate(counterfactuals_list):
        if cf_list is not None and hasattr(cf_list, 'final_cfs_df'):
            original = sample_instances.iloc[i]
            cfs = cf_list.final_cfs_df

            for _, cf_row in cfs.iterrows():
                cf_features = cf_row.iloc[:-1].values
                changes = np.abs(original.values - cf_features)
                feature_changes.append(changes)

    if feature_changes:
        feature_changes_df = pd.DataFrame(feature_changes, columns=sample_instances.columns)
        feature_importance = feature_changes_df.mean().sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importance.values, y=feature_importance.index, palette='viridis')
        plt.title('Average Feature Changes in Counterfactuals', fontsize=14, fontweight='bold')
        plt.xlabel('Average Absolute Change')
        plt.ylabel('Features')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/plots/feature_importance_counterfactuals.png', dpi=300, bbox_inches='tight')
        plt.show()

try:
    print("Generating counterfactual explanations...")

    # Prepare data for DICE
    X_test_df = pd.DataFrame(X_test, columns=X.columns)

    # Convert back to pandas DataFrame and ensure all features are present
    data_for_dice = pd.DataFrame(X_scaled_res, columns=X.columns)
    data_for_dice['target'] = y_res  # Add target column

    # Identify continuous features (numerical columns only)
    continuous_features = data_for_dice.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
    continuous_features = [f for f in continuous_features if f != 'target']  # Exclude target

    # Create DICE data object
    dice_data = dice_ml.Data(dataframe=data_for_dice,
                            continuous_features=continuous_features,
                            outcome_name='target')

    # Create DICE model
    dice_model = dice_ml.Model(model=final_model, backend="sklearn")

    # Create DICE explainer
    dice_explainer = dice_ml.Dice(dice_data, dice_model, method="random")

    # Scale query instances to match data_for_dice
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    query_instances = pd.DataFrame(X_test_scaled, columns=X.columns).head(3)

    # === ORIGINAL SINGLE INSTANCE ANALYSIS ===
    print("\n--- Single Instance Analysis ---")
    counterfactuals = dice_explainer.generate_counterfactuals(query_instances, total_CFs=2, desired_class=1)

    # Display results
    print("\n✓ Counterfactual explanations generated")
    print("\nOriginal instance (scaled):")
    print(query_instances.iloc[0])
    print("\nCounterfactual examples:")
    if hasattr(counterfactuals.cf_examples_list[0], 'final_cfs_df'):
        cf_df = counterfactuals.cf_examples_list[0].final_cfs_df
        print(cf_df.head())
    else:
        print("No counterfactual examples found")

    # Original visualization
    plt.figure(figsize=(14, 6))
    original_instance = query_instances.iloc[0]
    data_to_plot = pd.DataFrame({
        'Feature': original_instance.index,
        'Original': original_instance.values,
        'Counterfactual_1': cf_df.iloc[0].values[:-1],  # Exclude target
        'Counterfactual_2': cf_df.iloc[1].values[:-1]   # Exclude target
    })
    data_melted = data_to_plot.melt(id_vars='Feature', var_name='Type', value_name='Value')

    sns.barplot(x='Feature', y='Value', hue='Type', data=data_melted,
                palette=['#1f77b4', '#ff7f0e', '#2ca02c'],
                width=0.6)
    plt.xticks(rotation=45, ha='right')
    plt.title('Comparison of Original and Counterfactual Instances', fontsize=14, pad=20)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Scaled Values', fontsize=12)
    plt.legend(title='Instance Type', title_fontsize=10, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/counterfactual_comparison_journal_clean_simply.png', dpi=300, bbox_inches='tight')
    plt.show()

    # === NEW BULK EVALUATION ANALYSIS ===
    print("\n" + "="*50)
    print("BULK COUNTERFACTUAL EVALUATION")
    print("="*50)

    # Perform bulk analysis
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    sample_instances, counterfactuals_list = bulk_counterfactual_analysis(
        X_test_scaled_df, final_model, dice_explainer, n_samples=30, n_counterfactuals=3
    )

    # Evaluate counterfactual quality
    print("\nEvaluating counterfactual quality...")
    evaluation_results = evaluate_counterfactual_quality(
        sample_instances, counterfactuals_list, final_model, X.columns
    )

    # Print summary statistics
    print("\n--- BULK EVALUATION SUMMARY ---")
    print(f"Validity (flip rate): {np.mean(evaluation_results['validity']):.3f} ± {np.std(evaluation_results['validity']):.3f}")
    valid_proximities = [p for p in evaluation_results['proximity'] if p != float('inf')]
    if valid_proximities:
        print(f"Proximity (L2 distance): {np.mean(valid_proximities):.3f} ± {np.std(valid_proximities):.3f}")
    print(f"Diversity (pairwise dist): {np.mean(evaluation_results['diversity']):.3f} ± {np.std(evaluation_results['diversity']):.3f}")
    print(f"Sparsity (features changed): {np.mean(evaluation_results['sparsity']):.3f} ± {np.std(evaluation_results['sparsity']):.3f}")

    # Create comprehensive visualizations
    create_comprehensive_visualizations(evaluation_results, sample_instances, counterfactuals_list, OUTPUT_DIR)

    # Additional analysis: Most changed features
    print("\n--- FEATURE CHANGE ANALYSIS ---")
    all_feature_changes = []

    for i, cf_list in enumerate(counterfactuals_list):
        if cf_list is not None and hasattr(cf_list, 'final_cfs_df'):
            original = sample_instances.iloc[i]
            cfs = cf_list.final_cfs_df

            for _, cf_row in cfs.iterrows():
                cf_features = cf_row.iloc[:-1].values
                changes = np.abs(original.values - cf_features)
                all_feature_changes.append(changes)

    if all_feature_changes:
        feature_changes_df = pd.DataFrame(all_feature_changes, columns=X.columns)
        feature_importance = feature_changes_df.mean().sort_values(ascending=False)

        print("Top 10 most frequently changed features:")
        for i, (feature, change) in enumerate(feature_importance.head(10).items()):
            print(f"{i+1:2d}. {feature}: {change:.4f}")

    print("\n✓ Bulk counterfactual evaluation completed successfully!")

except Exception as e:
    print(f"⚠ Counterfactual explanation failed: {str(e)}")
    print("Continuing without counterfactual explanations...")
    import traceback
    traceback.print_exc()


# === STEP 13: COMPREHENSIVE RESULTS SUMMARY ===
print("\n" + "="*50)
print("STEP 13: COMPREHENSIVE RESULTS SUMMARY")
print("="*50)

# Create comprehensive summary
summary_data = {
    'Dataset_Shape': df.shape,
    'Features_After_Engineering': X.shape[1],
    'Best_Model': best_model,
    'Best_CV_ROC_AUC': cv_results[best_model].mean() if best_model in cv_results else 'N/A',
    'Test_Accuracy': test_metrics['accuracy'],
    'Test_Precision': test_metrics['precision'],
    'Test_Recall': test_metrics['recall'],
    'Test_F1': test_metrics['f1'],
    'Test_ROC_AUC': test_metrics['roc_auc'],
    'Total_Runtime': f"{(datetime.now() - datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')).total_seconds():.2f} seconds"
}

print("FINAL ANALYSIS SUMMARY")
print("=" * 60)
for key, value in summary_data.items():
    print(f"{key.replace('_', ' ').title():<25}: {value}")

# Save all results to Excel
try:
    with pd.ExcelWriter(f"{OUTPUT_DIR}/comprehensive_results.xlsx", engine='xlsxwriter') as writer:
        if cv_results:
            model_comparison = pd.DataFrame({
                'Model': list(cv_results.keys()),
                'Mean_ROC_AUC': [scores.mean() for scores in cv_results.values()],
                'Std_ROC_AUC': [scores.std() for scores in cv_results.values()],
                'Min_ROC_AUC': [scores.min() for scores in cv_results.values()],
                'Max_ROC_AUC': [scores.max() for scores in cv_results.values()]
            }).sort_values('Mean_ROC_AUC', ascending=False)
            model_comparison.to_excel(writer, sheet_name='Model_Comparison', index=False)

        # Test metrics
        test_metrics_df = pd.DataFrame([test_metrics])
        test_metrics_df.to_excel(writer, sheet_name='Test_Metrics', index=False)

        # Feature importance (if available)
        if not importance_df.empty:
            importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)

        # Hyperparameter tuning results
        tuning_df.to_excel(writer, sheet_name='Hyperparameter_Tuning', index=False)

        # Summary
        summary_df = pd.DataFrame([summary_data])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # TAMBAHAN: Comparative results
        comparative_results_df.to_excel(writer, sheet_name='Comparative_Results', index=False)

        print("✓ Comprehensive results saved to Excel")

except Exception as e:
    print(f"⚠ Error saving to Excel: {str(e)}")


# === STEP 14: FINAL RECOMMENDATIONS ===
print("\n" + "="*50)
print("STEP 14: FINAL RECOMMENDATIONS")
print("="*50)

print("BUSINESS RECOMMENDATIONS:")
print("-" * 40)
print("1. MODEL DEPLOYMENT:")
print(f"   - Deploy {best_model} model with {test_metrics['roc_auc']:.4f} ROC-AUC")
print(f"   - Expected precision: {test_metrics['precision']:.4f}")
print(f"   - Expected recall: {test_metrics['recall']:.4f}")

print("\n2. FEATURE INSIGHTS:")
if not importance_df.empty:
    top_3_features = importance_df.head(3)['feature'].tolist()
    print(f"   - Focus on top predictive features: {', '.join(top_3_features[:3])}")
else:
    print("   - Feature importance analysis not available for selected model")

print("\n3. MODEL MONITORING:")
print("   - Monitor model performance monthly")
print("   - Retrain when performance drops below 0.80 ROC-AUC")
print("   - Track prediction distribution for data drift")

print("\n4. BUSINESS ACTIONS:")
print("   - Implement targeted retention campaigns for high-risk customers")
print("   - Focus on improving customer engagement metrics")
print("   - Consider pricing strategy adjustments for at-risk segments")

print("\n" + "="*80)
print("ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"All outputs saved to: {OUTPUT_DIR}/")
print("Files generated:")
print("- Models: final_model.pkl, scaler.pkl")
print("- Data: feature_importance.csv, hyperparameter_tuning_results.csv, cv_metrics_per_fold.csv, comparative_model_results.csv")
print("- Plots: target_distribution.png, correlation_matrix.png, model_comparison.png, confusion_matrix_no_feature_engineering.png, confusion_matrix_feature_engineering.png,")
print("         confusion_matrix_feature_engineering_smote.png, ..., comparative_model_performance.png")
print("         confusion_matrix.png, roc_curve.png, precision_recall_curve.png,")
print("         calibration_curve.png, feature_importance.png, shap_summary.png, shap_bar.png")
print("- Summary: comprehensive_results.xlsx")
print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")