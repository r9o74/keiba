import pandas as pd
import sqlite3
import numpy as np
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup
import re
import datetime
from sklearn.metrics import roc_auc_score
import argparse
import os
import joblib

# ==========================================
# 1. 設定
# ==========================================
DB_NAME = "/Users/ryota/programs/keiba/keiba_data_main_2.db"
MODEL_PATH = "model_lgbm.joblib"
BEST_PARAMS_PATH = "best_params.joblib"

TARGET_RACE_IDS = [
    "202605010211"
]

# デフォルトパラメータ（best_params.joblib がなければこれを使用）
DEFAULT_MODEL_PARAMS = {
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

# learning.py で生成された best_params.joblib があれば読み込む
if os.path.exists(BEST_PARAMS_PATH):
    try:
        best_params_payload = joblib.load(BEST_PARAMS_PATH)
        MODEL_PARAMS = best_params_payload.get("params", DEFAULT_MODEL_PARAMS)
        print(f"✓ Optuna 最適パラメータをロード → {BEST_PARAMS_PATH}")
    except Exception as e:
        print(f"⚠ best_params.joblib のロードに失敗 ({e}) → デフォルトを使用")
        MODEL_PARAMS = DEFAULT_MODEL_PARAMS
else:
    MODEL_PARAMS = DEFAULT_MODEL_PARAMS

# ==========================================
# 2. 過去データ読み込み
# ==========================================
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
        
        if "race_class" not in df.columns:
            df["race_class"] = "不明"
        df["race_class"] = df["race_class"].fillna("不明").astype(str)

        return df
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return pd.DataFrame()

# ==========================================
# 3. 当日データスクレイピング (出馬表)
# ==========================================
def get_today_race_ids(target_date_str=None):
    """
    netkeibaから対象日のレース情報を取得。
    戻り値: {race_id: {"name": str}} の dict
    ※ race_list_sub.html を使用（race_list.html は JS シェルのみ）
    """

    if target_date_str is None:
        target_date_str = datetime.datetime.now().strftime("%Y%m%d")

    # race_list.html は Ajax シェルなので、実データが入る _sub を使う
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={target_date_str}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://race.netkeiba.com/",
    }

    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.encoding = "UTF-8"  # ページ宣言も UTF-8
        soup = BeautifulSoup(res.text, "html.parser")

        race_info = {}

        # dl.RaceList_DataList ごと（＝開催場ごと）に li を処理
        for li in soup.find_all("li", class_=re.compile(r"RaceList_DataItem")):
            # race_id を含む最初の <a> を取得（movie リンクは除外）
            a = li.find("a", href=re.compile(r"race_id=\d{12}"),
                        class_=lambda c: c != "LinkIconRaceMovie")
            if not a:
                continue
            href = a.get("href", "")
            m = re.search(r"race_id=(\d{12})", href)
            if not m:
                continue
            rid = m.group(1)

            # レース名: span.ItemTitle に格納されている
            name_span = li.find("span", class_="ItemTitle")
            name = name_span.get_text(strip=True) if name_span else ""

            # 取れなければ <a> テキスト全体からレース番号(「11R」等)を除いて使う
            if not name:
                full_text = a.get_text(separator=" ", strip=True)
                name = re.sub(r'^\d+R\s*', '', full_text).strip()

            # 重賞・OP/Listed 判定: Icon_GradeType1/2/3/5 を持つ span があれば True
            # Icon_GradeType17 = 条件特別（平場扱い）
            # GRADE_TYPES: 1=G1, 2=G2, 3=G3, 4=JpnI等, 5=OP/L
            GRADE_TYPES = {"Icon_GradeType1", "Icon_GradeType2", "Icon_GradeType3",
                           "Icon_GradeType4", "Icon_GradeType5"}
            grade_span = li.find("span", class_=re.compile(r"Icon_GradeType\d"))
            is_grade = False
            if grade_span:
                span_classes = set(grade_span.get("class", []))
                is_grade = bool(span_classes & GRADE_TYPES)

            race_info[rid] = {"name": name, "is_grade": is_grade}

        print(f"  {target_date_str}: {len(race_info)} レース取得")
        return race_info

    except Exception as e:
        print(f"レース一覧取得エラー ({target_date_str}): {e}")
        return {}

def safe_get_text(element, selector=None, class_=None, default=""):
    """要素からテキストを安全に取得するヘルパー"""
    if element is None:
        return default
    
    target = element
    if selector:
        target = element.find(selector, class_=class_)
    
    if target:
        return target.get_text(strip=True)
    return default

def get_shutuba_table(race_id):
    """
    指定されたrace_idの出馬表を取得・パースする (v2 logic)
    URL: https://race.netkeiba.com/race/shutuba.html?race_id=...
    """
    print(f"レースID {race_id} の出馬表を取得中...")
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        res = requests.get(url, headers=headers)
        res.encoding = "EUC-JP"
        soup = BeautifulSoup(res.text, "html.parser")
        
        # --- レース基本情報 ---
        data_intro = soup.find("div", class_="RaceData01")
        if not data_intro:
            print("レース情報が見つかりません。")
            return None
        intro_text = data_intro.get_text(strip=True)
        
        # 場所・日付
        place_code = race_id[4:6]
        place_map = {"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京","06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"} 
        # 府中、中山、新潟の順番なんか違和感ある
        
        # 海外レースが該当するはず
        place = place_map.get(place_code, "その他")
        
        # 日付
        race_date = pd.to_datetime(datetime.datetime.now().date())
        
        surface = "芝" if "芝" in intro_text else "ダ" if "ダ" in intro_text else "障害"
        distance = 1600
        dist_match = re.search(r'(\d+)m', intro_text)
        if dist_match: distance = int(dist_match.group(1))
        
        weather = "晴"
        if "天候:" in intro_text: weather = re.search(r'天候:(\w+)', intro_text).group(1)
        
        condition = "良"
        if "馬場:" in intro_text: condition = re.search(r'馬場:(\w+)', intro_text).group(1)
        
        # クラス判定
        race_name_div = soup.find("div", class_="RaceName")
        race_name = race_name_div.get_text(strip=True) if race_name_div else ""
        intro_full = soup.find("div", class_="RaceList_Item02").get_text(strip=True) if soup.find("div", class_="RaceList_Item02") else ""
        search_text = race_name + intro_full
        
        race_class = "OP"
        # netkeibaがローマ数字の1の代わりに大文字のI使ってるせいで最初反応しなかった。どっちでも対応できるよう修正
        mapping = {"GI":"G1", "GII":"G2", "GIII":"G3"}
        for c in ["G1","GI","G2","GII","G3","GIII","L","OP","3勝","2勝","1勝","未勝利","新馬"]:
            if c in search_text:
                race_class = mapping.get(c, c)
                if "勝" in race_class: race_class += "クラス"
                break
                
        # --- 馬データ ---
        table = soup.find("table", class_="RaceTable01")
        rows = table.find_all("tr", class_="HorseList")
        
        data_list = []
        for row in rows:
            try:
                # 枠・番
                bracket_val = safe_get_text(row, "td", re.compile("Waku"))
                h_num_val = safe_get_text(row, "td", re.compile("Umaban"))
                
                # 馬名・ID
                h_info = row.find("td", class_="HorseInfo")
                h_name = ""
                horse_id = ""
                if h_info:
                    a_tag = h_info.find("a")
                    if a_tag:
                        h_name = a_tag.get_text(strip=True)
                        h_url = a_tag.get("href")
                        if h_url:
                            match = re.search(r'horse/(\d+)', h_url)
                            if match:
                                horse_id = match.group(1)
                
                # 必須情報がない場合はスキップ
                if not h_name:
                    continue

                # 騎手
                j_td = row.find("td", class_="Jockey")
                jockey_id = ""
                if j_td and j_td.find("a"):
                    j_url = j_td.find("a").get("href")
                    if j_url:
                        match = re.search(r'recent/(\w+)', j_url)
                        if match:
                            jockey_id = match.group(1)

                # 斤量
                burden = safe_get_text(row, "td", "JockeyWeight", default="57.0")
                
                # その他 (性齢, オッズ, etc)
                sex_age = safe_get_text(row, "td", "Barei", default="牡3")
                if len(sex_age) >= 2:
                    sex, age = sex_age[0], sex_age[1:]
                else:
                    sex, age = "牡", "3"
                
                odds_text = safe_get_text(row, "td", re.compile("Odds"))
                odds = float(odds_text) if odds_text and odds_text != "--" and odds_text.replace('.', '', 1).isdigit() else np.nan
                
                weight_td = safe_get_text(row, "td", "Weight")
                weight = np.nan
                weight_diff = 0
                if weight_td and weight_td != "--":
                    match = re.match(r'(\d+)\((.+)\)', weight_td)
                    if match:
                        weight = int(match.group(1))
                        try: weight_diff = int(match.group(2))
                        except: pass
                
                # 数値変換 (エラー時はNaN)
                try: bracket = int(bracket_val)
                except: bracket = np.nan
                
                try: h_num = int(h_num_val)
                except: h_num = np.nan
                
                burden_val = float(burden) if burden.replace('.', '', 1).isdigit() else 55.0
                age_val = int(age) if age.isdigit() else 3

                data_list.append({
                    "race_id": race_id,
                    "race_date": race_date,
                    "place": place,
                    "surface": surface,
                    "distance": distance,
                    "race_class": race_class,
                    "weather": weather,
                    "condition": condition,
                    
                    "bracket": bracket,
                    "horse_number": h_num,
                    "horse_name": h_name,
                    "horse_id": horse_id,
                    "jockey_id": jockey_id,
                    "burden_weight": burden_val,
                    "sex": sex,
                    "age": age_val,
                    "odds": odds,
                    "weight": weight,
                    "weight_diff": weight_diff,
                    
                    # ターゲット用ダミー
                    "rank": np.nan,
                    "is_target": 1
                })
            except Exception as e:
                print(f"Row Error: {e}")
                
        return pd.DataFrame(data_list)
        
    except Exception as e:
        print(f"Error fetching race {race_id}: {e}")
        return None

# ==========================================
# 4. 前処理 (History + Today)
# ==========================================
def preprocess_wrapper(history_df, today_df):
    print("Combining history and today's data for feature engineering...")
    
    # 型合わせ
    today_df["rank"] = np.nan # ターゲットなし
    today_df["time_seconds"] = np.nan
    today_df["is_win"] = np.nan
    today_df["last_3f"] = np.nan
    
    # 必要な列を確保
    cols = ["bracket", "horse_number", "burden_weight", "weight", "weight_diff", "distance"]
    for c in cols:
        today_df[c] = pd.to_numeric(today_df[c], errors='coerce')
        
    # sex, age 分解 (もしカラムになければ)
    if "sex" not in today_df.columns or "age" not in today_df.columns:
        if "sex_age" in today_df.columns:
            today_df["sex"] = today_df["sex_age"].str[0]
            today_df["age"] = pd.to_numeric(today_df["sex_age"].str[1:], errors='coerce')
    else:
        today_df["age"] = pd.to_numeric(today_df["age"], errors='coerce')
    
    # 日付型
    today_df["race_date"] = pd.to_datetime(today_df["race_date"])
    
    # 結合 (History -> Today の順)
    # Historyは既にsortされていると仮定するが、念のため再度concat後にsort
    combined_df = pd.concat([history_df, today_df], ignore_index=True)
    combined_df = combined_df.sort_values(["race_date", "race_id"])
    
    # 特徴量エンジニアリング実行
    combined_df = feature_engineering(combined_df)
    
    return combined_df


def feature_engineering(df):
    
    # ターゲット予備計算
    df["is_ren"] = (df["rank"] <= 2).astype(int)
    df["is_win"] = (df["rank"] == 1).astype(int)
    
    # クラスごとの重み付け
    # 現在は簡易的にクラス分類だけで数値化
    # 同じクラスでもレースレベルが大きく異なるケースがあるため、将来的には過去のタイムや出走馬のレベルなども考慮して重み付けすることを検討
    # 牝限と牡馬混合が同じウエイトなのもおかしいので後々修正する予定
    
    def get_base_race_weight(cls_str):
        if pd.isna(cls_str): return 45
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
    df["course_id"] = df["place"].astype(str) + "_" + df["surface"].astype(str) + "_" + df["dist_cat"].astype(str)
    
    if "month_sin" not in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["race_date"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["race_date"].dt.month / 12)

    # Expanding Mean
    def expanding_mean(df, group_cols, target_col):
        # shift(1) しているので、今日のデータには前走までの平均が入る
        return df.groupby(group_cols, observed=False)[target_col].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    
    df["jockey_win_rate"] = expanding_mean(df, ["jockey_id"], "is_win")
    df["horse_win_rate"] = expanding_mean(df, ["horse_id"], "is_win")
    df["jockey_place_win_rate"] = expanding_mean(df, ["jockey_id", "place"], "is_win")
    df["jockey_dist_win_rate"] = expanding_mean(df, ["jockey_id", "dist_cat"], "is_win")
    df["jockey_surface_win_rate"] = expanding_mean(df, ["jockey_id", "surface"], "is_win")
    df["horse_course_win_rate"] = expanding_mean(df, ["horse_id", "course_id"], "is_win")
    df["bracket_by_course_win_rate"] = expanding_mean(df, ["course_id", "bracket"], "is_win")
    
    # Recent Trends
    jockey_group = df.groupby("jockey_id")
    df["jockey_recent_win_rate"] = jockey_group["is_win"].transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean()).fillna(0)
    
    horse_group = df.groupby("horse_id")
    df["horse_recent_avg_rank"] = horse_group["rank"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(10)
    
    df["prev_date"] = horse_group["race_date"].shift(1)
    df["interval_days"] = (df["race_date"] - df["prev_date"]).dt.days.fillna(999)

    # Speed Index
    race_group = df.groupby("race_id")
    
    def dev(s):
        # 今日のレースはタイムがないのでNaNになるはず -> 結果 NaN
        if s.isna().all(): return np.nan
        std = s.std()
        if std == 0 or pd.isna(std): return 50
        # s が NaN (今日のレース) だとここも NaN
        return 50 + 10 * (s.mean() - s) / std

    df["race_deviation"] = race_group["time_seconds"].transform(dev)
    
    # 今日のデータは race_deviation が NaN になる。
    # しかし、必要なのは「過去の」指数なので、
    # avg_abs_speed_idx などを計算するときは shift(1) するのでOK。
    # ただし、shift(1)した先がNaNだと困るが、過去データは埋まっているはず。
    
    df["abs_speed_index"] = df["race_deviation"] + df["race_class_weight_base"]
    
    df["avg_abs_speed_idx"] = horse_group["abs_speed_index"].transform(lambda x: x.shift(1).expanding().mean())
    df["max_abs_speed_idx"] = horse_group["abs_speed_index"].transform(lambda x: x.shift(1).expanding().max())
    df["prev_abs_speed_idx"] = horse_group["abs_speed_index"].shift(1)
    
    df["avg_abs_speed_idx"] = df["avg_abs_speed_idx"].fillna(85)
    df["max_abs_speed_idx"] = df["max_abs_speed_idx"].fillna(85)
    df["prev_abs_speed_idx"] = df["prev_abs_speed_idx"].fillna(85)
    
    # レベル判定など (相手関係)
    # ここは「今回のメンツ」での平均などを出す必要がある
    # 今日のレースメンバーの avg_abs_speed_idx は計算できている（過去の実績から）
    # なので、sum_rating / count_rating は今日のレースIDでも計算可能！
    
    df["sum_rating"] = race_group["avg_abs_speed_idx"].transform("sum")
    df["count_rating"] = race_group["avg_abs_speed_idx"].transform("count")
    
    df["race_level_index"] = (df["sum_rating"] - df["avg_abs_speed_idx"]) / (df["count_rating"] - 1)
    df["race_level_index"] = df["race_level_index"].replace([np.inf, -np.inf], 85).fillna(85)
    
    df["relative_competence"] = df["avg_abs_speed_idx"] - df["race_level_index"]
    
    df["prev_rank"] = horse_group["rank"].shift(1).fillna(10)
    df["prev_last_3f"] = horse_group["last_3f"].shift(1).fillna(36.0)
    
    drop_cols = ["sum_rating", "count_rating", "temp_speed_idx", "race_class_weight_base", "prev_date"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    
    return df

# ==========================================
# 5. モデル保存・ロード
# ==========================================

def save_model(model, use_features, cat_cols, n_train):
    """モデルと学習メタ情報を joblib で保存する"""
    db_mtime = os.path.getmtime(DB_NAME) if os.path.exists(DB_NAME) else 0
    payload = {
        "model":        model,
        "use_features": use_features,
        "cat_cols":     cat_cols,
        "n_train":      n_train,
        "trained_at":   datetime.datetime.now().isoformat(timespec="seconds"),
        "db_mtime":     db_mtime,
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"モデルを保存しました → {MODEL_PATH}  (学習サンプル数: {n_train:,})")


def load_model_if_valid():
    """
    保存済みモデルをロードして返す。
    以下の場合は None を返し、呼び出し元で再学習する:
      - モデルファイルが存在しない
      - DB がモデル学習後に更新されている
      - ロードに失敗した
    """
    if not os.path.exists(MODEL_PATH):
        print("保存済みモデルなし → 新規学習します")
        return None

    try:
        payload = joblib.load(MODEL_PATH)
        db_mtime = os.path.getmtime(DB_NAME) if os.path.exists(DB_NAME) else 0

        if db_mtime > payload.get("db_mtime", 0):
            print("DB が更新されています → 再学習します")
            return None

        print(f"保存済みモデルをロードしました"
              f"  (学習日時: {payload['trained_at']}, サンプル数: {payload['n_train']:,})")
        return payload

    except Exception as e:
        print(f"モデルロードエラー: {e} → 再学習します")
        return None


# ==========================================
# 6. インタラクティブ選択
# ==========================================

PLACE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉"
}

WEEKDAY_JP = ["月", "火", "水", "木", "金", "土", "日"]


def get_upcoming_kaisai_dates(weeks_ahead=3):
    """今日から数週間先の土日＋今日のうちレースがある日付を返す {date_str: [race_ids]}"""
    today = datetime.date.today()
    candidates = []

    # 今日を含め直近1日も確認
    for delta in range(-1, weeks_ahead * 7 + 1):
        d = today + datetime.timedelta(days=delta)
        if d == today or d.weekday() in (5, 6):  # 土=5, 日=6
            candidates.append(d)

    candidates = sorted(set(candidates))
    dates_with_races = {}

    print("開催日程を確認中...")
    for d in candidates:
        date_str = d.strftime("%Y%m%d")
        ids = get_today_race_ids(date_str)
        if ids:
            dates_with_races[date_str] = ids

    return dates_with_races


def parse_races_by_venue(race_ids):
    """race_idリストを場所ごとに整理して返す {place_code: [race_num_int, ...]}"""
    venues = {}
    for rid in race_ids:
        place_code = rid[4:6]
        race_num = int(rid[10:12])
        venues.setdefault(place_code, []).append(race_num)
    for pc in venues:
        venues[pc] = sorted(set(venues[pc]))
    return venues


def select_race_interactive(dates_with_races):
    """対話形式でrace_idを1つ選んで返す"""

    # --- ステップ1: 日付選択 ---
    date_list = sorted(dates_with_races.keys())
    print("\n開催日を選択してください:")
    for i, ds in enumerate(date_list, 1):
        d = datetime.date(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
        wd = WEEKDAY_JP[d.weekday()]
        print(f"  {i}. {d.strftime('%Y-%m-%d')} ({wd})")

    while True:
        try:
            choice = int(input("> ")) - 1
            if 0 <= choice < len(date_list):
                selected_date = date_list[choice]
                break
        except (ValueError, KeyboardInterrupt):
            pass
        print("番号を正しく入力してください。")

    race_ids_for_date = dates_with_races[selected_date]
    venues = parse_races_by_venue(race_ids_for_date)

    # --- ステップ2: 重賞/平場の選択 ---
    # is_grade フラグ（Icon_GradeType1/2/3/4/5）で重賞・OP/Listed を判定
    options = []  # (label, race_id or None, heiba_info or None)

    for pc in sorted(venues.keys()):
        place_name = PLACE_MAP.get(pc, f"不明({pc})")

        venue_race_ids = sorted(
            [rid for rid in race_ids_for_date if rid[4:6] == pc],
            key=lambda r: int(r[10:12])
        )
        grade_races = []
        heiba_nums = []

        for rid in venue_race_ids:
            r_num = int(rid[10:12])
            info = race_ids_for_date.get(rid, {})
            race_name = info.get("name", "")

            # is_grade フラグで重賞・OP/Listed を判定（レース名依存を廃止）
            if info.get("is_grade", False):
                grade_races.append((rid, r_num, race_name))
            else:
                heiba_nums.append(r_num)

        # 重賞・OP は個別に選択肢として追加
        for rid, r_num, name in grade_races:
            options.append((f"{place_name}{r_num}R {name}", rid, None))

        # 平場（重賞以外）をまとめて選択肢に追加
        if heiba_nums:
            options.append((f"{place_name} 平場", None, (pc, heiba_nums)))

    print(f"\nレース種別を選択してください ({selected_date[:4]}/{selected_date[4:6]}/{selected_date[6:8]}):")
    for i, (label, _, _) in enumerate(options, 1):
        print(f"  {i}. {label}")

    while True:
        try:
            choice = int(input("> ")) - 1
            if 0 <= choice < len(options):
                _, main_race_id, heiba_info = options[choice]
                break
        except (ValueError, KeyboardInterrupt):
            pass
        print("番号を正しく入力してください。")

    if main_race_id is not None:
        return [main_race_id]

    # --- ステップ3: 平場のレース番号選択 ---
    pc, heiba_nums = heiba_info
    place_name = PLACE_MAP.get(pc, f"不明({pc})")
    
    heiba_list = []
    for rid in race_ids_for_date.keys():
        if rid[4:6] == pc:
            r_num = int(rid[10:12])
            if r_num in heiba_nums:
                race_name = race_ids_for_date[rid].get("name", "")
                heiba_list.append((rid, r_num, race_name))
    
    # レース番号順にソート
    heiba_list.sort(key=lambda x: x[1])

    print(f"\n{place_name}の平場レースを選択してください:")
    for i, (rid, r_num, name) in enumerate(heiba_list, 1):
        print(f"  {i}. {r_num}R {name}")

    while True:
        try:
            choice = int(input("> ")) - 1
            if 0 <= choice < len(heiba_list):
                return [heiba_list[choice][0]]
        except (ValueError, KeyboardInterrupt):
            pass
        print("番号を正しく入力してください。")


# ==========================================
# 6. メイン処理
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict race results.")
    parser.add_argument("--race_id", type=str, help="Specific Valid Race ID to predict")
    parser.add_argument("--retrain", action="store_true", help="強制的に再学習してモデルを上書き保存する")
    args = parser.parse_args()

    # 1. 過去DB読み込み
    history_df = load_historical_data()
    if history_df.empty:
        print("過去データがないため終了します。")
        exit()

    # 2. レースID決定
    if args.race_id:
        print(f"引数で指定されたレースID: {args.race_id}")
        race_ids = [args.race_id]
    else:
        dates_with_races = get_upcoming_kaisai_dates()
        if not dates_with_races:
            print("直近の開催日程が取得できませんでした。")
            exit()
        race_ids = select_race_interactive(dates_with_races)
    
    
    if not race_ids:
        print("予測対象のレースが見つかりませんでした。終了します。")
        exit()
    
    # 3. 出馬表取得
    today_dfs = []
    for rid in race_ids:
        # ridが数値型などで渡ってきても大丈夫なように文字列化
        rid_str = str(rid)
        print(f"Scraping race {rid_str}...")
        try:
            tmp = get_shutuba_table(rid_str)
            if tmp is not None and not tmp.empty:
                today_dfs.append(tmp)
            else:
                print(f"Race {rid_str} のデータが取得できませんでした (ページ構造変更や無効なIDの可能性)")
        except Exception as e:
            print(f"Error processing {rid_str}: {e}")
            
    if not today_dfs:
        print("有効なレースデータが取得できませんでした。")
        exit()
        
    today_df = pd.concat(today_dfs, ignore_index=True)
    today_df["is_prediction_target"] = True
    print(f"Today's loaded rows: {len(today_df)}")
    
    # history_df に is_prediction_target = False を付与
    history_df["is_prediction_target"] = False
    
    # 重複回避: 今回取得したレースIDが過去データに含まれている場合、過去データ側を除外
    fetched_ids = today_df["race_id"].unique()
    history_df = history_df[~history_df["race_id"].isin(fetched_ids)]

    # 4. 結合＆特徴量生成
    full_df = preprocess_wrapper(history_df, today_df)
    
    # 5. 学習データの準備
    train_df = full_df[full_df["is_prediction_target"] == False].dropna(subset=["rank"]).copy()
    predict_df = full_df[full_df["is_prediction_target"] == True].copy()
    
    if predict_df.empty:
        print("予測対象のデータ抽出に失敗しました")
        exit()

    # ... (これ以降の特徴量リストや学習・予測処理は変更なし) ...

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

    # 5. モデルのロードまたは学習
    saved = None if args.retrain else load_model_if_valid()

    if saved is not None:
        # ── 保存済みモデルを再利用 ──
        full_model   = saved["model"]
        use_features = saved["use_features"]
        cat_cols     = saved["cat_cols"]
    else:
        # ── 新規学習 ──
        for c in cat_cols:
            if c in train_df.columns:
                train_df[c] = train_df[c].astype("category")

        use_features = [f for f in features if f in train_df.columns]
        target = (train_df["rank"] == 1).astype(int)

        print("Training model on full history...")
        full_model = lgb.LGBMClassifier(**MODEL_PARAMS)
        full_model.fit(
            train_df[use_features], target,
            categorical_feature=[c for c in cat_cols if c in use_features]
        )
        save_model(full_model, use_features, cat_cols, n_train=len(train_df))

    # predict_df のカテゴリ列を変換
    for c in cat_cols:
        if c in predict_df.columns:
            predict_df[c] = predict_df[c].astype("category")

    # 6. 予測実行
    print("Predicting races...")
    preds = full_model.predict_proba(predict_df[use_features])[:, 1]
    predict_df["pred_score"] = preds
    
    # 7. 結果表示
    results = predict_df.sort_values(["race_id", "pred_score"], ascending=[True, False])
    
    print("\n========= PREDICTION RESULTS =========")
    current_race = None
    for idx, row in results.iterrows():
        if row["race_id"] != current_race:
            r_date = row.get("race_date", "")
            if isinstance(r_date, pd.Timestamp):
                r_date = r_date.strftime("%Y-%m-%d")
            print(f"\n--- Race {row['race_id']} ({r_date}) ---")
            current_race = row["race_id"]
        
        h_name = row.get("horse_name", "Unknown")
        print(f"  #{row['horse_number']} {h_name} (ID:{row['horse_id']}) Score:{row['pred_score']:.4f}")