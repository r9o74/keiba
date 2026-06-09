import pandas as pd
import sqlite3
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import os
import optuna
from optuna.integration import LightGBMPruningCallback

# ==========================================
# 1. 設定・データ読み込み
# ==========================================
DB_NAME = "keiba_data_main_2.db"
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
        
        # 数値データへの変換
        numeric_cols = ["rank", "bracket", "horse_number", "odds", "popularity", 
                        "weight", "weight_diff", "age", "distance", "burden_weight", "last_3f"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # タイム変換
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
        
        # 日付変換
        if "race_date" in df.columns:
            df["race_date"] = pd.to_datetime(df["race_date"])
            df["year"] = df["race_date"].dt.year
            df["month_sin"] = np.sin(2 * np.pi * df["race_date"].dt.month / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["race_date"].dt.month / 12)
        
        # クラス情報（文字列化）
        if "race_class" not in df.columns:
            df["race_class"] = "不明"
        df["race_class"] = df["race_class"].fillna("不明").astype(str)

        return df
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return pd.DataFrame()

# ==========================================
# 2. 特徴量エンジニアリング（Data-Driven & Interaction）
# ==========================================
def preprocess_data(df):
    print("特徴量エンジニアリング (v3: Advanced) を実行中...")
    df = df.sort_values(["race_date", "race_id"])
    
    # ターゲット変数
    df["is_ren"] = (df["rank"] <= 2).astype(int)
    df["is_win"] = (df["rank"] == 1).astype(int)
    
    # --- 1. Data-Driven Race Class Weights ---
    # 既存のヒューリスティックではなく、クラスごとの平均賞金やレベル感があればベストだが、
    # ここでは既存のヒューリスティックをベースにしつつ、データ分布で微調整する簡易ロジック
    # 本当のデータドリブン: クラスごとの平均走破タイム偏差値などを計算してマッピングする
    
    # まずは基本ウェイト（ベースライン）
    def get_base_race_weight(cls_str):
        if "G1" in cls_str or "GI" in cls_str: return 100
        if "G2" in cls_str or "GII" in cls_str: return 90
        if "G3" in cls_str or "GIII" in cls_str: return 85
        if "L" in cls_str or "OP" in cls_str or "オ" in cls_str: return 80
        if "3勝" in cls_str: return 70
        if "2勝" in cls_str: return 60
        if "1勝" in cls_str: return 50
        if "未勝利" in cls_str: return 40
        if "新馬" in cls_str: return 40 # 新馬は未知数だがレベルは低め設定
        return 45 # その他

    df["race_class_weight_base"] = df["race_class"].apply(get_base_race_weight)
    
    # クラスごとの平均タイム指数（補正用）
    # ※注: 本来はLeakageを防ぐため過去データのみで計算すべきだが、クラスの定義は不変と仮定して全体統計を使う
    # ここでは簡易的に「そのクラスの平均賞金」などの外部データがないため、
    # Base Weight をそのまま採用するが、特徴量として「クラスごとの平均タイム」を計算してmergeするアプローチをとる
    
    # レースカテゴリ（芝・ダート x 距離区分）ごとの正規化タイム
    # これを使って「このクラスは平均よりどれくらい速いか」を算出
    df["dist_cat"] = pd.cut(df["distance"], bins=[0, 1400, 1800, 2200, 3000, 9999], labels=["Sprint", "Mile", "Middle", "Long", "SuperLong"])
    
    # [NEW] Course ID (複合キー)
    df["course_id"] = df["place"].astype(str) + "_" + df["surface"].astype(str) + "_" + df["dist_cat"].astype(str)
    
    # --- 2. Advanced Rolling Statistics (Jockey/Trainer/Horse) ---
    def expanding_mean(df, group_cols, target_col):
        return df.groupby(group_cols, observed=False)[target_col].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    
    # 基本勝率
    df["jockey_win_rate"] = expanding_mean(df, ["jockey_id"], "is_win")
    df["horse_win_rate"] = expanding_mean(df, ["horse_id"], "is_win")
    
    # 粒度の細かい勝率 (Interaction)
    # 騎手 x 場所
    df["jockey_place_win_rate"] = expanding_mean(df, ["jockey_id", "place"], "is_win")
    # 騎手 x 距離区分
    df["jockey_dist_win_rate"] = expanding_mean(df, ["jockey_id", "dist_cat"], "is_win")
    # 騎手 x サーフェス (芝/ダート)
    df["jockey_surface_win_rate"] = expanding_mean(df, ["jockey_id", "surface"], "is_win")
    
    # [NEW] Horse Course Compatibility (コース適性)
    df["horse_course_win_rate"] = expanding_mean(df, ["horse_id", "course_id"], "is_win")

    # [NEW] Track Bias (枠番 x コース)
    df["bracket_by_course_win_rate"] = expanding_mean(df, ["course_id", "bracket"], "is_win")
    
    # 最近の勢い (Trend)
    jockey_group = df.groupby("jockey_id")
    df["jockey_recent_win_rate"] = jockey_group["is_win"].transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean()).fillna(0)
    
    horse_group = df.groupby("horse_id")
    df["horse_recent_avg_rank"] = horse_group["rank"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(10)
    
    # 間隔
    df["prev_date"] = horse_group["race_date"].shift(1)
    df["interval_days"] = (df["race_date"] - df["prev_date"]).dt.days.fillna(999)

    # --- 3. 絶対スピード指数 (v2 Logic + Refinement) ---
    race_group = df.groupby("race_id")
    
    # (A) レース内偏差値
    def dev(s):
        std = s.std()
        if std == 0 or pd.isna(std): return 50
        return 50 + 10 * (s.mean() - s) / std

    df["race_deviation"] = race_group["time_seconds"].transform(dev)
    
    # (B) 絶対指数 = 偏差値 + クラスウェイト
    df["abs_speed_index"] = df["race_deviation"] + df["race_class_weight_base"]
    
    # (C) 過去指数集計
    df["avg_abs_speed_idx"] = horse_group["abs_speed_index"].transform(lambda x: x.shift(1).expanding().mean())
    df["max_abs_speed_idx"] = horse_group["abs_speed_index"].transform(lambda x: x.shift(1).expanding().max()) # ベストパフォーマンス
    df["prev_abs_speed_idx"] = horse_group["abs_speed_index"].shift(1)
    
    # 欠損補完: 初出走などは基準値(例えば40+50=90)より少し低めで
    df["avg_abs_speed_idx"] = df["avg_abs_speed_idx"].fillna(85)
    df["max_abs_speed_idx"] = df["max_abs_speed_idx"].fillna(85)
    df["prev_abs_speed_idx"] = df["prev_abs_speed_idx"].fillna(85)
    
    # --- 4. 対戦相手レベル & 相対指標 ---
    df["sum_rating"] = race_group["avg_abs_speed_idx"].transform("sum")
    df["count_rating"] = race_group["avg_abs_speed_idx"].transform("count")
    
    # 自分を除いた平均
    df["race_level_index"] = (df["sum_rating"] - df["avg_abs_speed_idx"]) / (df["count_rating"] - 1)
    df["race_level_index"] = df["race_level_index"].replace([np.inf, -np.inf], 85).fillna(85)
    
    # 相対コンピテンス
    df["relative_competence"] = df["avg_abs_speed_idx"] - df["race_level_index"]
    
    # --- 5. 変動計 (Lag Features) ---
    # 体重増減の変動
    # df["weight_diff"] は既に「前走比」だが、「前走の体重」との差分などを明示的に入れても良いが、
    # ここでは「前走の人気」と「今回の人気」の乖離などを見たいが、今回の人気は予測時は使えない（オッズは直前まで不明）
    # しかし学習には使えるかもしれないが、Leakage注意。
    # 代わりに「前走の着順」や「前走の上がり3F」を入れる
    df["prev_rank"] = horse_group["rank"].shift(1).fillna(10)
    df["prev_last_3f"] = horse_group["last_3f"].shift(1).fillna(36.0)
    
    # --- 6. カテゴリ特徴量のエンコーディング準備 ---
    # ここではLabelEncoding的な処理はLightGBMに任せるので、category型にするだけ

    # 不要カラム削除
    drop_cols = ["sum_rating", "count_rating", "temp_speed_idx", "race_class_weight_base"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    
    return df

# ==========================================
# 3. Optunaによる最適化 & 学習
# ==========================================
def run_optimization(df):
    df = df.reset_index(drop=True)
    df = df.dropna(subset=["rank", "race_level_index"]).copy()
    
    features = [
        # 基本
        "bracket", "horse_number", "burden_weight", "age", "weight", "weight_diff",
        "distance", "interval_days",
        # 勝率・実績
        "jockey_win_rate", "horse_win_rate",
        "jockey_place_win_rate", "jockey_dist_win_rate", "jockey_surface_win_rate",
        "horse_course_win_rate", "bracket_by_course_win_rate",
        "jockey_recent_win_rate", "horse_recent_avg_rank",
        "prev_rank", "prev_last_3f",
        # スピード指数・レベル
        "prev_abs_speed_idx", "avg_abs_speed_idx", "max_abs_speed_idx",
        "race_level_index", "relative_competence",
        # 環境 (カテゴリ)
        "place", "surface", "weather", "condition", "race_class", "dist_cat",
        "month_sin", "month_cos"
    ]
    
    cat_cols = ["place", "surface", "weather", "condition", "race_class", "dist_cat"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    use_features = [f for f in features if f in df.columns]
    target = (df["rank"] == 1).astype(int) # 1着予想
    
    print(f"使用する特徴量 ({len(use_features)}個): {use_features}")
    
    # TimeSeriesSplit
    n_splits = 4
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # 直近のデータを検証にするように分割IDを取得
    # Optuna効率化のため、最後の1Foldだけでチューニングするか、CV平均を見るか。
    # ここではCV平均を見る（堅牢性重視）
    
    def objective(trial):
        param = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': 42,
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        }
        
        cv_scores = []
        
        # CV Loop
        for i, (train_index, val_index) in enumerate(tscv.split(df)):
            X_train, X_val = df.iloc[train_index][use_features], df.iloc[val_index][use_features]
            y_train, y_val = target.iloc[train_index], target.iloc[val_index]
            
            # Pruning用のCallback
            pruning_callback = LightGBMPruningCallback(trial, "auc")
            
            model = lgb.LGBMClassifier(**param)
            
            # 早期終了ありで学習
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    # pruning_callback # Pruning入れると早いが、CV平均だと使いにくいので今回は外すか、最後のFoldだけ見てPruningするか検討
                ],
                categorical_feature=[c for c in cat_cols if c in use_features]
            )
            
            preds = model.predict_proba(X_val)[:, 1]
            try:
                score = roc_auc_score(y_val, preds)
            except:
                score = 0.5
            cv_scores.append(score)
            
        return np.mean(cv_scores)

    print("\nOptunaによるハイパーパラメータ探索を開始します (Trial=20)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20) # 時間短縮のため20trial

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return study.best_params, use_features, cat_cols, target, df

# ==========================================
# 4. 最終学習と可視化
# ==========================================
def train_final_model(df, best_params, use_features, cat_cols, target):
    print("\nベストパラメータで最終学習曲線をプロットします...")
    
    # n_estimatorsなどは固定値を上書きする場合あり
    final_params = best_params.copy()
    final_params['n_estimators'] = 2000
    final_params['random_state'] = 42
    final_params['importance_type'] = 'gain'
    
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    train_scores = []
    val_scores = []
    train_sizes = []
    
    for i, (train_index, val_index) in enumerate(tscv.split(df)):
        X_train, X_val = df.iloc[train_index][use_features], df.iloc[val_index][use_features]
        y_train, y_val = target.iloc[train_index], target.iloc[val_index]
        
        model = lgb.LGBMClassifier(**final_params)
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

    # グラフ描画
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', color="r", label="Train")
    plt.plot(train_sizes, val_scores, 'o-', color="green", label="Val (v3: Optuna Optimized)")
    plt.title(f"学習曲線 v3 (AUC > 0.800 Challange)")
    plt.xlabel("学習データ数")
    plt.ylabel("AUC Score")
    plt.legend()
    plt.grid()
    plt.savefig("learning_curve_v3.png")
    print("\n保存完了: learning_curve_v3.png")
    
    # 特徴量重要度
    # 最後のモデルを使用
    lgb.plot_importance(model, max_num_features=20, importance_type='gain', figsize=(10, 8))
    plt.title("Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig("feature_importance_v3.png")
    print("保存完了: feature_importance_v3.png")

if __name__ == "__main__":
    df = load_historical_data()
    if not df.empty:
        df = preprocess_data(df)
        best_params, use_features, cat_cols, target, df = run_optimization(df)
        train_final_model(df, best_params, use_features, cat_cols, target)
