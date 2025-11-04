# src/data_preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# --- Configuration Constants ---
IMPOSSIBLE_ZEROS_COLS = ['Glucose', 'Diastolic_BP', 'Skin_Fold', 'Serum_Insulin', 'BMI']
NUMERICAL_COLS = ['Pregnant', 'Glucose', 'Diastolic_BP', 'Skin_Fold', 
                  'Serum_Insulin', 'BMI', 'Diabetes_Pedigree', 'Age']
CATEGORICAL_COLS = ['Age_Group', 'BMI_Category', 'Glucose_Category']






def process_phase_2_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 2: Handles missing values (zeros to NaN + median imputation) and caps outliers (IQR)."""
    print("\n✨ Starting Phase 2: Data Cleaning and Outlier Treatment...")
    df_clean = df.copy()

    # --- Task 1: Missing Value Conversion (Zeros to NaN) ---
    df_clean[IMPOSSIBLE_ZEROS_COLS] = df_clean[IMPOSSIBLE_ZEROS_COLS].replace(0, np.nan)
    print("  -> Task 1: Biologically impossible zeros converted to NaN.")
    
    # --- Task 2: Imputation ---
    median_values = df_clean[IMPOSSIBLE_ZEROS_COLS].median()
    df_clean[IMPOSSIBLE_ZEROS_COLS] = df_clean[IMPOSSIBLE_ZEROS_COLS].fillna(median_values)
    print("  -> Task 2: Missing NaNs filled with Median values.")


    # --- Task 3: Outlier Treatment (Capping with IQR) ---
    def identify_iqr_fences(data, column):
        # ... (IQR calculation logic remains the same)
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        return Q1 - (1.5 * IQR), Q3 + (1.5 * IQR)

    outliers_capped = 0
    # ... (rest of the outlier capping logic remains the same)
    for col in df_clean.columns.drop('Class'):
        lower_fence, upper_fence = identify_iqr_fences(df_clean, col)
        outliers_count = len(df_clean[(df_clean[col] < lower_fence) | (df_clean[col] > upper_fence)])
        outliers_capped += outliers_count
        
        df_clean[col] = np.where(df_clean[col] < lower_fence, lower_fence, df_clean[col])
        df_clean[col] = np.where(df_clean[col] > upper_fence, upper_fence, df_clean[col])
    
    print(f"  -> Task 3: Total of {outliers_capped} extreme values capped.")
    print("✅ Phase 2 Completed successfully.")
    return df_clean


def process_phase_3_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 3: Creates new features, encodes them, and scales numerical data."""
    print("\n🚀 Starting Phase 3: Feature Transformation (Engineering, Encoding, Scaling)...")
    
    df.columns = df.columns.str.strip() 

    # Task 1: Feature Engineering
    age_bins = [20, 35, 50, df['Age'].max() + 1]
    age_labels = ['Young', 'Middle-aged', 'Senior']
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False, include_lowest=True)

    bmi_bins = [0, 18.5, 25, 30, df['BMI'].max() + 1]
    bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df['BMI_Category'] = pd.cut(df['BMI'], bins=bmi_bins, labels=bmi_labels, right=False, include_lowest=True)

    glucose_bins = [0, 100, 140, 200]
    glucose_labels = ['Normal', 'Pre-diabetic', 'Diabetic']
    df['Glucose_Category'] = pd.cut(df['Glucose'], bins=glucose_bins, labels=glucose_labels, right=False, include_lowest=True)
    print("  -> Task 1: 3 new categorical features created.")
    
    # Task 2: Encoding
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True) 
    print(f"  -> Task 2: Encoding complete. Shape: {df_encoded.shape}")

    # Task 3: Scaling
    scaler = StandardScaler() 
    df_encoded[NUMERICAL_COLS] = scaler.fit_transform(df_encoded[NUMERICAL_COLS])
    print("  -> Task 3: Scaling (StandardScaler) applied to original numerical features.")
    
    print("✅ Phase 3 Completed successfully.")
    return df_encoded


def process_phase_5_balance(df_final: pd.DataFrame) -> pd.DataFrame:
    """Phase 5: Applies SMOTE to balance the target class."""
    print("\n⚖️ Starting Phase 5: Data Imbalance Handling (SMOTE)...")
    
    X = df_final.drop('Class', axis=1)
    y = df_final['Class']

    print(f"  -> Class Distribution BEFORE SMOTE: {Counter(y)}")

    # Balancing (SMOTE)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    df_balanced = pd.concat([X_res, y_res], axis=1)
    
    print(f"  -> Class Distribution AFTER SMOTE: {Counter(y_res)}")
    print(f"  -> Total samples increased to {len(df_balanced)}.")
    print("✅ Phase 5 Completed successfully.")
    return df_balanced

# The function below is kept separate as it is exploratory and typically remains in the notebook.
def analyze_phase_4_reduction(df_transformed: pd.DataFrame):
    """Phase 4: Provides correlation analysis for data reduction decisions."""
    target_corr = df_transformed.corr()['Class'].sort_values(ascending=False).head(8)
    print("\n### 📉 PHASE 4: Data Reduction (Exploratory Check) ###")
    print("Top 8 Features Correlated with Target (Decision-making data):")
    print(target_corr)
    print("---------------------------------------------------------")