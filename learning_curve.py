import pandas as pd
import sqlite3
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
import datetime
import os

# ==========================================
# 1. Config & Data Loading
# ==========================================
DB_NAME = "keiba_data_main_2.db"
# Use a font that supports Japanese if available, otherwise default
# On Windows, 'Meiryo' or 'MS Gothic' is standard for Japanese
plt.rcParams['font.family'] = 'Meiryo'

def load_historical_data():
    print("Loading historical data...")
    try:
        if not os.path.exists(DB_NAME):
            print(f"Database {DB_NAME} not found.")
            return pd.DataFrame()

        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql("SELECT * FROM race_results", conn)
        conn.close()
        
        # Numeric conversion
        numeric_cols = ["rank", "bracket", "horse_number", "odds", "popularity", 
                        "weight", "weight_diff", "age", "distance", "burden_weight", "last_3f"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Time conversion
        def time_to_seconds(t):
            try:
                if pd.isna(t): return np.nan
                t = str(t)
                if ":" in t:
                    m, s = t.split(":")
                    return int(m) * 60 + float(s)
                return float(t)
            except: return np.nan
        if "タイム" in df.columns:
            df["time_seconds"] = df["タイム"].apply(time_to_seconds)
        
        # Date conversion
        if "race_date" in df.columns:
            df["race_date"] = pd.to_datetime(df["race_date"])
            df["year"] = df["race_date"].dt.year
            df["month_sin"] = np.sin(2 * np.pi * df["race_date"].dt.month / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["race_date"].dt.month / 12)
        
        # Class
        if "race_class" not in df.columns:
            df["race_class"] = "不明"
        df["race_class"] = df["race_class"].fillna("不明").astype(str)

        return df
    except Exception as e:
        print(f"Data loading error: {e}")
        return pd.DataFrame()

# ==========================================
# 2. Preprocessing (Adapted for Analysis)
# ==========================================
def preprocess_data(df):
    print("Preprocessing data...")
    # Sort by date for correct lag features
    df = df.sort_values(["horse_id", "race_date"])
    
    # --- Feature Engineering ---
    # Race-level stats (Speed Index, PCI)
    # Note: Using transform on the whole dataset implies we know the race stats strictly within that race.
    # This is not leakage for *past* races.
    
    race_group = df.groupby("race_id")
    def calculate_deviation(series):
        mean = series.mean()
        std = series.std()
        if std == 0 or pd.isna(std): return 50
        return 50 + 10 * (mean - series) / std

    df["speed_index"] = race_group["time_seconds"].transform(calculate_deviation)
    df["last_3f_index"] = race_group["last_3f"].transform(calculate_deviation)

    df["time_first_part"] = df["time_seconds"] - df["last_3f"]
    df["pci"] = (df["last_3f"] / df["time_first_part"]) * 100
    df["pci"] = df["pci"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Classification Targets
    df["is_ren"] = (df["rank"] <= 2).astype(int)
    df["is_win"] = (df["rank"] == 1).astype(int)

    # History-based features (expanding/rolling)
    # These MUST rely only on *previous* rows.
    
    # 1. Jockey/Horse pair win rate (Shift 1 to exclude current race)
    df["pair_ren_rate"] = df.groupby(["horse_id", "jockey_id"])["is_ren"].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(0)

    # 2. Bracket win rate (by course conditions)
    # Warning: Grouping by ["place", "surface", "distance", "bracket"] and expanding mean
    # effectively calculates the historical win rate of that bracket *up to that point*.
    # This is correct valid logic for simulation.
    df["bracket_win_rate"] = df.groupby(["place", "surface", "distance", "bracket"])["is_win"].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(0)

    # 3. Lag features per horse
    horse_group = df.groupby("horse_id")
    df["prev_rank"] = horse_group["rank"].shift(1)
    df["prev_speed_index"] = horse_group["speed_index"].shift(1)
    df["prev_last_3f_index"] = horse_group["last_3f_index"].shift(1)
    df["prev_pci"] = horse_group["pci"].shift(1)

    for window in [3, 5]:
        df[f"avg_rank_{window}"] = horse_group["rank"].transform(lambda x: x.shift(1).rolling(window).mean())
        df[f"avg_speed_index_{window}"] = horse_group["speed_index"].transform(lambda x: x.shift(1).rolling(window).mean())
        df[f"avg_last_3f_index_{window}"] = horse_group["last_3f_index"].transform(lambda x: x.shift(1).rolling(window).mean())
        df[f"avg_pci_{window}"] = horse_group["pci"].transform(lambda x: x.shift(1).rolling(window).mean())

    df["cumulative_avg_rank"] = horse_group["rank"].transform(lambda x: x.shift(1).expanding().mean())
    df["distance_avg_rank"] = df.groupby(["horse_id", "distance"])["rank"].transform(lambda x: x.shift(1).expanding().mean())
    df["place_avg_rank"] = df.groupby(["horse_id", "place"])["rank"].transform(lambda x: x.shift(1).expanding().mean())
    df["surface_avg_rank"] = df.groupby(["horse_id", "surface"])["rank"].transform(lambda x: x.shift(1).expanding().mean())
    
    if "sex_age_cat" not in df.columns:
        # Assuming sex/age exist or handled in load
        # For simplicity, if missing, fill default
        if "sex" in df.columns and "age" in df.columns:
             df["sex_age_cat"] = df["sex"] + df["age"].astype(str)
        else:
             df["sex_age_cat"] = "unknown"

    return df

# ==========================================
# 3. Learning Curve Calculation
# ==========================================
def plot_learning_curve_time_split(df):
    # Sort by Date for TimeSeriesSplit
    df = df.sort_values("race_date").reset_index(drop=True)
    
    # Filter valid data for training
    # Must have rank and odds to be valid historical data
    df = df.dropna(subset=["rank", "odds"]).copy()
    
    features = [
        "bracket", "horse_number", "burden_weight", "age", "weight", "weight_diff",
        "sex_age_cat", "place", "surface", "distance", "weather", "condition",
        "pair_ren_rate", "bracket_win_rate", "jockey_id",
        "prev_rank", "prev_speed_index", "prev_last_3f_index", "prev_pci",
        "avg_rank_3", "avg_speed_index_3", "avg_last_3f_index_3", "avg_pci_3",
        "avg_rank_5", "avg_speed_index_5", "avg_last_3f_index_5", "avg_pci_5",
        "cumulative_avg_rank", "distance_avg_rank", "place_avg_rank", "surface_avg_rank",
        "year", "month_sin", "month_cos", "race_class"
    ]
    
    cat_cols = [
        "jockey_id", "place", "surface", "weather", "condition", "sex_age_cat", "race_class"
    ]
    
    # Ensure categorical columns are type 'category'
    for col in cat_cols:
        if col in df.columns: 
            df[col] = df[col].astype("category")
        else:
            # Create dummy if missing to avoid error
            df[col] = "unknown"
            df[col] = df[col].astype("category")

    use_features = [f for f in features if f in df.columns]
    target = (df["rank"] == 1).astype(int)
    
    # --- Time Series Split ---
    # Using 5 splits to show progression
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    train_scores = []
    val_scores = []
    train_sizes = []
    
    params = {
        'random_state': 42,
        'n_estimators': 100, # Reduced from 1000 for faster curve generation, or stick to 1000 if speed ok
        'learning_rate': 0.05,
        'min_child_samples': 100,
        'num_leaves': 31,
        'importance_type': 'gain',
        'verbose': -1
    }
    
    print(f"Starting TimeSeriesSplit ({n_splits} splits)...")
    
    for i, (train_index, val_index) in enumerate(tscv.split(df)):
        X_train, X_val = df.iloc[train_index][use_features], df.iloc[val_index][use_features]
        y_train, y_val = target.iloc[train_index], target.iloc[val_index]
        
        print(f"Split {i+1}/{n_splits}: Train size={len(X_train)}, Val size={len(X_val)}")
        
        # Train
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            categorical_feature=[c for c in cat_cols if c in use_features]
        )
        
        # Evaluate (AUC)
        # Check if validation set has both classes
        if len(np.unique(y_val)) < 2:
            print(f"Warning: Split {i+1} validation set has only 1 class. Skipping AUC.")
            val_auc = 0.5
        else:
            val_preds = model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_preds)
            
        train_preds = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_preds)
        
        train_scores.append(train_auc)
        val_scores.append(val_auc)
        train_sizes.append(len(X_train))
        
        print(f"  Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', color="r", label="Training AUC")
    plt.plot(train_sizes, val_scores, 'o-', color="g", label="Validation AUC")
    plt.title(f"Learning Curve (Blue: Train, Green: Val) - TimeSeriesSplit")
    plt.xlabel("Training Set Size (Number of Rows)")
    plt.ylabel("AUC Score")
    plt.legend(loc="best")
    plt.grid()
    
    output_file = "learning_curve.png"
    plt.savefig(output_file)
    print(f"\nLearning curve saved to {output_file}")

def main():
    df = load_historical_data()
    if df.empty:
        print("No data found.")
        return
        
    df = preprocess_data(df)
    plot_learning_curve_time_split(df)

if __name__ == "__main__":
    main()
