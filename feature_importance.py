
import pandas as pd
import sqlite3
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import os

DB_NAME = "/Users/ryota/programs/keiba/keiba_data_main_2.db"
plt.rcParams['font.family'] = 'Meiryo'

def load_historical_data():
    print("過去データを読み込んでいます...")
    try:
        if not os.path.exists(DB_NAME):
            print(f"データベース {DB_NAME} が見つかりません。")
            return pd.DataFrame()

        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql("SELECT * FROM race_results", conn)
        conn.close()
        
        numeric_cols = ["rank", "bracket", "horse_number", "odds", "popularity", 
                        "weight", "weight_diff", "age", "distance", "burden_weight", "last_3f"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        def time_to_seconds(t):
            try:
                if pd.isna(t): return np.nan
                t = str(t)
                if ":" in t:
                    m, s = t.split(":")
                    if len(m) > 0 and len(s) > 0:
                        return int(m) * 60 + float(s)
                return float(t)
            except: return np.nan
        if "タイム" in df.columns:
            df["time_seconds"] = df["タイム"].apply(time_to_seconds)
        
        if "race_date" in df.columns:
            df["race_date"] = pd.to_datetime(df["race_date"])
            df["year"] = df["race_date"].dt.year
            df["month_sin"] = np.sin(2 * np.pi * df["race_date"].dt.month / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["race_date"].dt.month / 12)
        
        if "race_class" not in df.columns:
            df["race_class"] = "不明"
        df["race_class"] = df["race_class"].fillna("不明").astype(str)

        return df
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return pd.DataFrame()

def preprocess_data(df):
    print("特徴量エンジニアリング (Manual Plot Version) を実行中...")
    df = df.sort_values(["race_date", "race_id"])
    
    df["is_ren"] = (df["rank"] <= 2).astype(int)
    df["is_win"] = (df["rank"] == 1).astype(int)
    
    def get_base_race_weight(cls_str):
        if "G1" in cls_str or "GI" in cls_str: return 100
        if "G2" in cls_str or "GII" in cls_str: return 90
        if "G3" in cls_str or "GIII" in cls_str: return 85
        if "L" in cls_str or "OP" in cls_str or "オ" in cls_str: return 80
        if "3勝" in cls_str: return 70
        if "2勝" in cls_str: return 60
        if "1勝" in cls_str: return 50
        if "未勝利" in cls_str: return 40
        if "新馬" in cls_str: return 40
        return 45

    df["race_class_weight_base"] = df["race_class"].apply(get_base_race_weight)
    
    df["dist_cat"] = pd.cut(df["distance"], bins=[0, 1400, 1800, 2200, 3000, 9999], labels=["Sprint", "Mile", "Middle", "Long", "SuperLong"])
    
    # [NEW] Course ID (複合キー) - User Re-added this
    df["course_id"] = df["place"].astype(str) + "_" + df["surface"].astype(str) + "_" + df["dist_cat"].astype(str)
    
    def expanding_mean(df, group_cols, target_col):
        return df.groupby(group_cols, observed=False)[target_col].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    
    df["jockey_win_rate"] = expanding_mean(df, ["jockey_id"], "is_win")
    df["horse_win_rate"] = expanding_mean(df, ["horse_id"], "is_win")
    
    df["jockey_place_win_rate"] = expanding_mean(df, ["jockey_id", "place"], "is_win")
    df["jockey_dist_win_rate"] = expanding_mean(df, ["jockey_id", "dist_cat"], "is_win")
    df["jockey_surface_win_rate"] = expanding_mean(df, ["jockey_id", "surface"], "is_win")
    
    # [NEW] Horse Course Compatibility - User Re-added
    df["horse_course_win_rate"] = expanding_mean(df, ["horse_id", "course_id"], "is_win")

    # [NEW] Track Bias - User Re-added
    df["bracket_by_course_win_rate"] = expanding_mean(df, ["course_id", "bracket"], "is_win")
    
    jockey_group = df.groupby("jockey_id")
    df["jockey_recent_win_rate"] = jockey_group["is_win"].transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean()).fillna(0)
    
    horse_group = df.groupby("horse_id")
    df["horse_recent_avg_rank"] = horse_group["rank"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(10)
    
    df["prev_date"] = horse_group["race_date"].shift(1)
    df["interval_days"] = (df["race_date"] - df["prev_date"]).dt.days.fillna(999)

    race_group = df.groupby("race_id")
    
    def dev(s):
        std = s.std()
        if std == 0 or pd.isna(std): return 50
        return 50 + 10 * (s.mean() - s) / std

    df["race_deviation"] = race_group["time_seconds"].transform(dev)
    
    df["abs_speed_index"] = df["race_deviation"] + df["race_class_weight_base"]
    
    df["avg_abs_speed_idx"] = horse_group["abs_speed_index"].transform(lambda x: x.shift(1).expanding().mean())
    df["max_abs_speed_idx"] = horse_group["abs_speed_index"].transform(lambda x: x.shift(1).expanding().max())
    df["prev_abs_speed_idx"] = horse_group["abs_speed_index"].shift(1)
    
    df["avg_abs_speed_idx"] = df["avg_abs_speed_idx"].fillna(85)
    df["max_abs_speed_idx"] = df["max_abs_speed_idx"].fillna(85)
    df["prev_abs_speed_idx"] = df["prev_abs_speed_idx"].fillna(85)
    
    df["sum_rating"] = race_group["avg_abs_speed_idx"].transform("sum")
    df["count_rating"] = race_group["avg_abs_speed_idx"].transform("count")
    
    df["race_level_index"] = (df["sum_rating"] - df["avg_abs_speed_idx"]) / (df["count_rating"] - 1)
    df["race_level_index"] = df["race_level_index"].replace([np.inf, -np.inf], 85).fillna(85)
    
    df["relative_competence"] = df["avg_abs_speed_idx"] - df["race_level_index"]
    
    df["prev_rank"] = horse_group["rank"].shift(1).fillna(10)
    df["prev_last_3f"] = horse_group["last_3f"].shift(1).fillna(36.0)
    
    drop_cols = ["sum_rating", "count_rating", "temp_speed_idx", "race_class_weight_base"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    
    return df

def train_final_model(df):
    df = df.reset_index(drop=True)
    df = df.dropna(subset=["rank", "race_level_index"]).copy()
    
    features = [
        "bracket", "horse_number", "burden_weight", "age", "weight", "weight_diff",
        "distance", "interval_days",
        "jockey_win_rate", "horse_win_rate",
        "jockey_place_win_rate", "jockey_dist_win_rate", "jockey_surface_win_rate",
        "horse_course_win_rate", "bracket_by_course_win_rate",
        "jockey_recent_win_rate", "horse_recent_avg_rank",
        "prev_rank", "prev_last_3f",
        "prev_abs_speed_idx", "avg_abs_speed_idx", "max_abs_speed_idx",
        "race_level_index", "relative_competence",
        "place", "surface", "weather", "condition", "race_class", "dist_cat",
        "month_sin", "month_cos"
    ]
    
    cat_cols = ["place", "surface", "weather", "condition", "race_class", "dist_cat"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    use_features = [f for f in features if f in df.columns]
    target = (df["rank"] == 1).astype(int)
    
    print(f"Features: {use_features}")
    
    # Best Params from Trial 0
    params = {
        'learning_rate': 0.019783049312554545,
        'num_leaves': 134,
        'max_depth': 5,
        'min_child_samples': 76,
        'subsample': 0.9944085111693457,
        'colsample_bytree': 0.6683466662054883,
        'reg_alpha': 0.04664228567544981,
        'reg_lambda': 0.5709268510659403,
        'n_estimators': 2000,
        'random_state': 42,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'importance_type': 'gain'
    }
    
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    train_scores = []
    val_scores = []
    train_sizes = []
    
    print("Starting final training loop...")
    
    for i, (train_index, val_index) in enumerate(tscv.split(df)):
        X_train, X_val = df.iloc[train_index][use_features], df.iloc[val_index][use_features]
        y_train, y_val = target.iloc[train_index], target.iloc[val_index]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            categorical_feature=[c for c in cat_cols if c in use_features]
        )
        
        val_preds = model.predict_proba(X_val)[:, 1]
        try: val_auc = roc_auc_score(y_val, val_preds)
        except: val_auc = 0.5
            
        train_preds = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_preds)
        
        train_scores.append(train_auc)
        val_scores.append(val_auc)
        train_sizes.append(len(X_train))
        
        print(f"Split {i+1}: TrainAUC={train_auc:.4f}, ValAUC={val_auc:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', color="r", label="Train")
    plt.plot(train_sizes, val_scores, 'o-', color="green", label="Val")
    plt.title(f"Learning Curve (Manual V3)")
    plt.xlabel("Train Size")
    plt.ylabel("AUC Score")
    plt.legend()
    plt.grid()
    plt.savefig("learning_curve_v3.png")
    print("Saved learning_curve_v3.png")
    
    lgb.plot_importance(model, max_num_features=20, importance_type='gain', figsize=(10, 8))
    plt.tight_layout()
    plt.savefig("feature_importance_v3.png")

if __name__ == "__main__":
    df = load_historical_data()
    if not df.empty:
        df = preprocess_data(df)
        train_final_model(df)
