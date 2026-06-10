import pandas as pd
import sqlite3
import numpy as np
import lightgbm as lgb
import joblib
import os

# ==========================================
# 設定
# ==========================================
DB_NAME = "/Users/ryota/programs/keiba/keiba_data_main_2.db"
BEST_PARAMS_PATH = "best_params.joblib"
TRAIN_END_YEAR = 2024  # 学習に使用する最終年
TEST_START_YEAR = 2025 # シミュレーションを開始する年

# ==========================================
# 1. データ読み込み・前処理
# ==========================================
def load_and_preprocess():
    print("データベースから全データを読み込んでいます...")
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql("SELECT * FROM race_results", conn)
        conn.close()
    except Exception as e:
        print(f"DB読み込みエラー: {e}")
        return pd.DataFrame()

    # --- 数値変換 ---
    numeric_cols = ["rank", "bracket", "horse_number", "odds", "popularity", 
                    "weight", "weight_diff", "age", "distance", "burden_weight", "last_3f"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- タイム変換 ---
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

    # --- 日付処理 ---
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["year"] = df["race_date"].dt.year
    df["month"] = df["race_date"].dt.month
    
    # 月のsin/cos変換
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- クラス ---
    if "race_class" not in df.columns: df["race_class"] = "不明"
    df["race_class"] = df["race_class"].fillna("不明").astype(str)

    # --- 特徴量生成 ---
    print("特徴量を生成中...")
    df = df.sort_values(["race_id", "horse_number"])
    
    race_group = df.groupby("race_id")
    def calculate_deviation(series):
        mean = series.mean()
        std = series.std()
        if std == 0 or pd.isna(std): return 50
        return 50 + 10 * (mean - series) / std

    df["speed_index"] = race_group["time_seconds"].transform(calculate_deviation).fillna(50)
    df["last_3f_index"] = race_group["last_3f"].transform(calculate_deviation).fillna(50)

    df["time_first_part"] = df["time_seconds"] - df["last_3f"]
    df["pci"] = (df["last_3f"] / df["time_first_part"]) * 100
    df["pci"] = df["pci"].replace([np.inf, -np.inf], np.nan).fillna(0)

    df["is_ren"] = (df["rank"] <= 2).astype(int)
    df["pair_ren_rate"] = df.groupby(["horse_id", "jockey_id"])["is_ren"].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(0)

    df["is_win"] = (df["rank"] == 1).astype(int)
    df["bracket_win_rate"] = df.groupby(["place", "surface", "distance", "bracket"])["is_win"].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(0)
    
    # 過去成績
    df = df.sort_values(["horse_id", "race_date"])
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

    if "sex" in df.columns and "age" in df.columns:
        df["sex_age_cat"] = df["sex"] + df["age"].astype(str)

    return df

# ==========================================
# 2. シミュレーション実行
# ==========================================
def run_simulation():
    # データロード
    df = load_and_preprocess()
    if df.empty: return

    # --- データの分割 (2024年まで vs 2025年以降) ---
    train_df = df[df["year"] <= TRAIN_END_YEAR].copy()
    test_df = df[df["year"] >= TEST_START_YEAR].copy()
    
    # 学習データのクリーニング
    train_df = train_df.dropna(subset=["rank", "odds"])
    # テストデータも結果が必要なのでdropna
    test_df = test_df.dropna(subset=["rank", "odds"])

    if test_df.empty:
        print(f"エラー: {TEST_START_YEAR}年以降のデータがデータベースに存在しません。")
        return
    
    print(f"\n学習データ期間: 〜{TRAIN_END_YEAR}年 ({len(train_df)}件)")
    print(f"シミュレーション期間: {TEST_START_YEAR}年〜 ({len(test_df)}件 / {len(test_df['race_id'].unique())}レース)")

    # --- 特徴量設定 ---
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
    
    cat_cols = ["jockey_id", "place", "surface", "weather", "condition", "sex_age_cat", "race_class"]
    for col in cat_cols:
        train_df[col] = train_df[col].astype("category")
        test_df[col] = test_df[col].astype("category")

    use_features = [f for f in features if f in train_df.columns]

    # --- 学習 ---
    print("\n[モデル学習中 (LightGBM)]...")

    # best_params.joblib があれば Optuna 最適パラメータを使用、なければデフォルト
    default_params = {
        'random_state': 42,
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'min_child_samples': 100,
        'num_leaves': 31,
        'importance_type': 'gain'
    }
    if os.path.exists(BEST_PARAMS_PATH):
        try:
            payload = joblib.load(BEST_PARAMS_PATH)
            loaded = payload.get("params", {})
            # シミュレーション用に固定すべき項目だけ上書き
            params = {**loaded, 'random_state': 42, 'importance_type': 'gain'}
            print(f"✓ Optuna 最適パラメータをロード → {BEST_PARAMS_PATH}")
        except Exception as e:
            print(f"⚠ best_params.joblib のロードに失敗 ({e}) → デフォルトを使用")
            params = default_params
    else:
        params = default_params

    model = lgb.LGBMClassifier(**params)
    model.fit(
        train_df[use_features], 
        (train_df["rank"] == 1).astype(int),
        categorical_feature=[c for c in cat_cols if c in use_features],
        callbacks=[lgb.log_evaluation(0)]
    )

    # --- 予測 ---
    print(f"\n[予測実行 ({TEST_START_YEAR}年以降の全レース)]...")
    test_df["score"] = model.predict_proba(test_df[use_features])[:, 1]
    
    # --- 結果集計 ---
    print("収支計算中...")
    results = []
    race_ids = test_df["race_id"].unique()
    
    for rid in race_ids:
        race_data = test_df[test_df["race_id"] == rid].sort_values("score", ascending=False)
        if len(race_data) < 2: continue
        
        # AIの本命馬 (Score 1位)
        top_horse = race_data.iloc[0]
        second_horse = race_data.iloc[1]
        
        diff = top_horse["score"] - second_horse["score"]
        score = top_horse["score"]
        
        # ランク判定
        rank_char = "C"
        if score >= 0.40 and diff >= 0.15: rank_char = "A"
        elif score >= 0.25 and diff >= 0.05: rank_char = "B"
        
        is_hit = (top_horse["rank"] == 1)
        return_amount = top_horse["odds"] * 100 if is_hit else 0
        
        results.append({
            "race_id": rid,
            "date": top_horse["race_date"],
            "rank_char": rank_char,
            "bet_amount": 100,
            "return_amount": return_amount,
            "is_hit": is_hit,
            "odds": top_horse["odds"]
        })
        
    results_df = pd.DataFrame(results)
    
    # --- パターン別シミュレーション出力 ---
    patterns = [
        (["A"], "パターン1: A評価 (鉄板級) のみ購入"),
        (["A", "B"], "パターン2: A・B評価 (有力級) を購入"),
        (["A", "B", "C"], "パターン3: 全レース購入 (混戦含む)")
    ]
    
    print("\n" + "="*70)
    print(f"【収支シミュレーション結果】 期間: {TEST_START_YEAR}年 〜 現在")
    print("※ AIの本命馬の単勝を100円均等買いした場合")
    print("="*70)
    
    for target_ranks, label in patterns:
        target_df = results_df[results_df["rank_char"].isin(target_ranks)]
        count = len(target_df)
        
        if count == 0:
            print(f"\n{label}\n  -> 該当レースなし")
            continue
            
        total_bet = target_df["bet_amount"].sum()
        total_return = target_df["return_amount"].sum()
        profit = total_return - total_bet
        roi = (total_return / total_bet) * 100
        accuracy = target_df["is_hit"].mean()
        
        # 最高配当
        max_return = target_df[target_df["is_hit"]]["odds"].max() if target_df["is_hit"].any() else 0

        print(f"\n{label}")
        print(f"  購入レース数 : {count:,} R")
        print(f"  的中率       : {accuracy:.1%} ({target_df['is_hit'].sum()}/{count})")
        print(f"  回収率       : {roi:.1f}%")
        print(f"  収支         : {profit:+,.0f}円 (投資: {total_bet:,}円 -> 回収: {total_return:,.0f}円)")
        print(f"  最高的中単勝 : {max_return}倍")

if __name__ == "__main__":
    run_simulation()