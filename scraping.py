import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import sqlite3
import datetime
import numpy as np
from io import StringIO
from tqdm import tqdm
import os

# ==========================================
# 設定項目
# ==========================================

# 実行スクリプトの場所（upload内）から見て、親ディレクトリにあるDBを指すように修正
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_NAME = os.path.join(BASE_DIR, "keiba_data_main_2.db")

# is_already_saved, get_race_ids_by_month は変更不要のため省略
# (元のコードのまま使用してください)
def is_already_saved(race_id, db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='race_results'")
        if cursor.fetchone() is None:
            conn.close()
            return False
        cursor.execute("SELECT 1 FROM race_results WHERE race_id = ? LIMIT 1", (race_id,))
        exists = cursor.fetchone() is not None
    except sqlite3.OperationalError:
        exists = False
    conn.close()
    return exists

def get_latest_race_date(db_name):
    """DB内の最新のレース開催日を取得する"""
    if not os.path.exists(db_name):
        return None
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='race_results'")
        if cursor.fetchone() is None:
            return None
        cursor.execute("SELECT MAX(race_date) FROM race_results")
        res = cursor.fetchone()
        if res and res[0]:
            # "YYYY-MM-DD" 形式を想定してパース
            return datetime.datetime.strptime(res[0], "%Y-%m-%d").date()
    except Exception as e:
        print(f"最新日付取得エラー: {e}")
    finally:
        conn.close()
    return None

def get_race_ids_by_month(year, month):
    url = f"https://db.netkeiba.com/race/list/{year}{str(month).zfill(2)}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        res.encoding = "EUC-JP"
        soup = BeautifulSoup(res.text, "html.parser")
        day_links = soup.find_all('a', href=re.compile(r'/race/list/\d{8}/'))
        day_urls = list(set([f"https://db.netkeiba.com{link.get('href')}" for link in day_links]))
        
        race_ids = []
        for day_url in day_urls:
            time.sleep(1)
            res_day = requests.get(day_url, headers=headers)
            res_day.encoding = "EUC-JP"
            soup_day = BeautifulSoup(res_day.text, "html.parser")
            links = soup_day.find_all('a', href=re.compile(r'/race/\d{12}/'))
            for link in links:
                r_id = re.findall(r'\d{12}', link.get('href'))[0]
                race_ids.append(r_id)
        return sorted(list(set(race_ids)))
    except Exception as e:
        print(f"\n[Error] ID取得エラー ({year}/{month}): {e}")
        return []


# ==========================================
# 【修正版】データ取得関数
# ==========================================
def get_race_result(race_id):
    url = f"https://db.netkeiba.com/race/{race_id}/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.encoding = "EUC-JP"
        soup = BeautifulSoup(res.text, "html.parser")

        # --- レース情報の抽出 (修正部分) ---
        data_intro = soup.find("div", class_="data_intro")
        
        surface, distance, weather, condition = "不明", 0, "不明", "不明"
        race_class, weight_type = "不明", "不明"
        meeting_day = 0
        race_date = "1900-01-01" # デフォルト値設定

        if data_intro:
            # --- 1. テキスト情報の全取得 ---
            # クラス判別精度向上のため、h1(レース名)とdata_intro全体のテキストを結合して検索対象にする
            race_name_elem = soup.find("h1")
            race_name_text = race_name_elem.get_text(strip=True) if race_name_elem else ""
            
            # data_intro内の全テキストを取得（改行削除）
            full_intro_text = data_intro.get_text().replace("\n", "").replace(u'\xa0', u' ')
            
            # 結合テキスト（ここから各種情報を探す）
            search_text = race_name_text + " " + full_intro_text

            # --- 基本情報の抽出 ---
            if "芝" in full_intro_text: surface = "芝"
            elif "ダ" in full_intro_text: surface = "ダ"
            elif "障" in full_intro_text: surface = "障害"
            
            dist_match = re.search(r'(\d+)m', full_intro_text)
            if dist_match: distance = int(dist_match.group(1))
            
            weather_match = re.search(r'天候\s*:\s*(\w+)', full_intro_text)
            if weather_match: weather = weather_match.group(1)
            
            cond_match = re.search(r'(?:芝|ダート|馬場)\s*:\s*(\w+)', full_intro_text)
            if cond_match: condition = cond_match.group(1)

            # --- 2. 日付・開催情報の取得 ---
            small_txt = data_intro.find("p", class_="smalltxt")
            text_header = small_txt.get_text().replace("\n", "").replace(u'\xa0', u' ') if small_txt else ""

            date_match = re.search(r'(\d+)年(\d+)月(\d+)日', text_header)
            if date_match:
                race_date = f"{date_match.group(1)}-{date_match.group(2).zfill(2)}-{date_match.group(3).zfill(2)}"
            
            day_match = re.search(r'\d+回[^\d]+(\d+)日(?:目)?', text_header)
            if day_match:
                meeting_day = int(day_match.group(1))

            # --- 3. クラス情報の抽出（優先順位付き判定） ---
            # 検索対象: search_text (タイトル + 詳細テキスト)
            
            # 【重要】判定の優先順位定義マップ
            # 上にあるものほど優先されます。キーが検索語句、値が保存されるクラス名です。
            # タイトルに含まれる "(GI)" などを先に検知し、下の詳細にある "オープン" を無視するようにします。
            class_priority_map = {
                # --- 優先度 1: 障害重賞 (JGI等はGIを含むため先に判定) ---
                "JGIII": "JG3", "JG3": "JG3",
                "JGII": "JG2", "JG2": "JG2",
                "JGI": "JG1", "JG1": "JG1",
                
                # --- 優先度 2: 地方交流重賞 (Jpn表記) ---
                "JpnIII": "G3", "Jpn3": "G3",
                "JpnII": "G2", "Jpn2": "G2",
                "JpnI": "G1", "Jpn1": "G1",
                
                # --- 優先度 3: 平地重賞 (GIIIがGIに誤検知されないようGIIIから先に) ---
                "GIII": "G3", "G3": "G3",
                "GII": "G2", "G2": "G2",
                "GI": "G1", "G1": "G1",
                
                # --- 優先度 4: リステッド・OP ---
                "(L)": "L", "リステッド": "L", 
                "オープン": "OP", "OP": "OP",
                
                # --- 優先度 5: 条件戦 (3勝が1勝に誤検知されないよう上位から) ---
                "3勝": "3勝クラス", "1600万": "3勝クラス",
                "2勝": "2勝クラス", "1000万": "2勝クラス",
                "1勝": "1勝クラス", "500万": "1勝クラス",
                
                # --- 優先度 6: 新馬・未勝利 ---
                "新馬": "新馬", 
                "未勝利": "未勝利"
            }
            
            found_class = False
            for key, val in class_priority_map.items():
                if key in search_text:
                    race_class = val
                    found_class = True
                    break
            
            # マップで見つからなかった場合のバックアップ（通常はここには来ないはずですが念のため）
            if not found_class:
                if "障害" in search_text and "オープン" in search_text:
                    race_class = "OP" # 障害オープン
                # その他の単純なクラス名検索
                elif "1勝" in search_text: race_class = "1勝クラス"
                elif "2勝" in search_text: race_class = "2勝クラス"
                elif "3勝" in search_text: race_class = "3勝クラス"

            # 斤量種別の判定 (変更なし)
            if "ハンデ" in search_text: weight_type = "ハンデ"
            elif "別定" in search_text: weight_type = "別定"
            elif "定量" in search_text: weight_type = "定量"
            elif "馬齢" in search_text: weight_type = "馬齢"


        # --- ラップタイム (修正部分) ---
        # 画像のクラス名を直接ターゲットにする
        lap_time_str = ""
        lap_cell = soup.find("td", class_="race_lap_cell")
        if lap_cell:
            lap_time_str = lap_cell.get_text(strip=True)
        else:
            # バックアップ: 以前のロジック（th検索）も残しておく
            lap_header = soup.find("th", string=re.compile("ラップタイム"))
            if lap_header:
                lap_td = lap_header.find_next_sibling("td")
                if lap_td: lap_time_str = lap_td.get_text(strip=True)

        # --- テーブル抽出 (変更なし) ---
        table = soup.find("table", class_="race_table_01")
        if not table: return None
        rows = table.find_all("tr")
        header_texts = [c.get_text(strip=True) for c in rows[0].find_all("th")]
        
        col_map = {}
        expected_cols = ["rank", "bracket", "h_num", "h_name", "sex_age", "weight", "jockey", "time", "margin", "pass", "last_3f", "odds", "pop", "h_weight", "trainer"]
        for c in expected_cols: col_map[c] = -1

        for i, t in enumerate(header_texts):
            if "着順" in t: col_map["rank"]=i
            elif "枠" in t: col_map["bracket"]=i
            elif "馬番" in t: col_map["h_num"]=i
            elif "馬名" in t: col_map["h_name"]=i
            elif "性齢" in t: col_map["sex_age"]=i
            elif "斤量" in t: col_map["weight"]=i
            elif "タイム" in t and "指数" not in t: col_map["time"]=i
            elif "着差" in t: col_map["margin"]=i
            elif "通過" in t: col_map["pass"]=i
            elif "上り" in t or "上がり" in t or "3F" in t: col_map["last_3f"]=i
            elif "単勝" in t: col_map["odds"]=i
            elif "人気" in t: col_map["pop"]=i
            elif "馬体重" in t: col_map["h_weight"]=i
            elif "騎手" in t: col_map["jockey"]=i
            elif "調教師" in t: col_map["trainer"]=i

        data_list = []
        horse_ids, jockey_ids, trainer_ids = [], [], []
        
        for row in rows[1:]:
            cells = row.find_all("td")
            if len(cells) < 5: continue 
            
            def get_c(idx): 
                return cells[idx].get_text(strip=True) if idx >= 0 and idx < len(cells) else ""
            
            row_data = {
                "rank": get_c(col_map["rank"]),
                "bracket": get_c(col_map["bracket"]),
                "horse_number": get_c(col_map["h_num"]),
                "horse_name": get_c(col_map["h_name"]),
                "sex_age": get_c(col_map["sex_age"]),
                "burden_weight": get_c(col_map["weight"]),
                "タイム": get_c(col_map["time"]),
                "margin": get_c(col_map["margin"]),
                "passing_order": get_c(col_map["pass"]),
                "last_3f": get_c(col_map["last_3f"]),
                "odds": get_c(col_map["odds"]),
                "popularity": get_c(col_map["pop"]),
                "馬体重": get_c(col_map["h_weight"])
            }
            data_list.append(row_data)

            h_link = cells[col_map["h_name"]].find("a") if col_map["h_name"] != -1 else None
            j_link = cells[col_map["jockey"]].find("a") if col_map["jockey"] != -1 else None
            t_link = cells[col_map["trainer"]].find("a") if col_map["trainer"] != -1 and col_map["trainer"] < len(cells) else None
            
            horse_ids.append(h_link.get("href").split("/")[-2] if h_link else "")
            jockey_ids.append(j_link.get("href").split("/")[-2] if j_link else "")
            trainer_ids.append(t_link.get("href").split("/")[-2] if t_link else "")

        df = pd.DataFrame(data_list)
        df["horse_id"] = horse_ids
        df["jockey_id"] = jockey_ids
        df["trainer_id"] = trainer_ids
        df["race_id"] = race_id
        
        place_code = race_id[4:6]
        place_map = {"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京","06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"}
        df["place"] = place_map.get(place_code, "その他")
        
        df["race_date"] = race_date
        df["surface"] = surface
        df["distance"] = distance
        df["weather"] = weather
        df["condition"] = condition
        df["race_class"] = race_class
        df["weight_type"] = weight_type
        df["lap_time"] = lap_time_str
        df["meeting_day"] = meeting_day

        df["sex"] = df["sex_age"].str[0]
        df["age"] = df["sex_age"].str[1:]
        
        def split_weight(val):
            if not isinstance(val, str): return np.nan, np.nan
            match = re.match(r'(\d+)\((.+)\)', val)
            if match:
                return match.group(1), match.group(2)
            else:
                if val.isdigit(): return val, np.nan
                return val, np.nan

        weight_data = df["馬体重"].apply(split_weight)
        df["weight"] = [x[0] for x in weight_data]
        df["weight_diff"] = [x[1] for x in weight_data]

        return df
    except Exception as e:
        # print(f"Error in {race_id}: {e}") 
        return None
    
# ==========================================
# 【修正】データ整形関数
# ==========================================
def clean_keiba_data(df):
    if df is None: return None
    df = df.copy()
    
    # タイム変換
    def time_to_seconds(t_str):
        if not isinstance(t_str, str) or ":" not in t_str: return np.nan
        try:
            m, s = t_str.split(":")
            return int(m)*60 + float(s)
        except: return np.nan

    if "タイム" in df.columns:
        df["time_seconds"] = df["タイム"].apply(time_to_seconds)
        # 1着タイムとの差（優勝馬が存在しない場合はNaN）
        try:
            min_time = df["time_seconds"].min()
            df["time_diff"] = round(df["time_seconds"] - min_time, 3)
        except:
            df["time_diff"] = np.nan

    # 数値変換リスト：着順は特殊な文字（取消など）を含むため、強制変換リストから除外
    numeric_cols = ["bracket", "horse_number", "odds", "popularity", "last_3f", 
                    "weight", "weight_diff", "age", "distance", "burden_weight", "meeting_day"]
    
    for col in numeric_cols:
        if col in df.columns:
            # カンマが入っている場合の除去などはここで行うと良い
            # df[col] = df[col].astype(str).str.replace(',', '') 
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 「着順」の処理：元の文字列カラムを残し、計算用の数値カラムを追加
    if "rank" in df.columns:
        df["rank_numeric"] = pd.to_numeric(df["rank"], errors='coerce')
        
    return df

# save_to_db, main_scraping は変更不要のため省略
# (元のコードのまま使用してください)
def save_to_db(df, db_name):
    if df is None or df.empty: return
    conn = sqlite3.connect(db_name)
    df.to_sql("race_results", conn, if_exists="append", index=False)
    conn.close()

def main_scraping():
    # DBの最新日付を確認し、開始地点を決定
    latest_date = get_latest_race_date(DB_NAME)
    now = datetime.datetime.now()
    
    if latest_date:
        start_year, start_month = latest_date.year, latest_date.month
        print(f"DB内の最新データ日付: {latest_date}。ここから更新分をチェックします。")
    else:
        # DBが空の場合のデフォルト開始位置
        start_year, start_month = 2021, 1
        print(f"データベースにデータがないため、{start_year}年から開始します。")

    end_year, end_month = now.year, now.month
    print(f"=== スクレイピング開始: {start_year}年{start_month}月 〜 {end_year}年{end_month}月 ===")
    total_stats = {"success": 0, "error": 0, "skipped": 0}
    
    for year in range(start_year, end_year + 1):
        m_start = start_month if year == start_year else 1
        m_end = end_month if year == end_year else 12
        
        for month in range(m_start, m_end + 1):
            race_ids = get_race_ids_by_month(year, month)
            if not race_ids: continue
            pbar = tqdm(race_ids, desc=f"{year}/{month}", unit="race")
            for r_id in pbar:
                if is_already_saved(r_id, DB_NAME):
                    total_stats["skipped"] += 1
                    continue
                df = get_race_result(r_id)
                if df is not None:
                    df_cleaned = clean_keiba_data(df)
                    save_to_db(df_cleaned, DB_NAME)
                    total_stats["success"] += 1
                else: total_stats["error"] += 1
                pbar.set_postfix(total_stats)
                time.sleep(1.0)
    print(f"\n=== 全工程終了 ===")

if __name__ == "__main__":
    main_scraping()