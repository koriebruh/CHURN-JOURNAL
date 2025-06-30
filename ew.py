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