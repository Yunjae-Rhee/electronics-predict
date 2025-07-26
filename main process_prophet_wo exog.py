import os
import sqlite3
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.seasonal import STL

# ────────────────────────────────
# 0. 경로 설정
# ────────────────────────────────
base_path = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(base_path, "data")

db_path       = os.path.join(data_dir, "cleaned_demand.db")
save_db_path  = os.path.join(data_dir, "event_dummy.db")

# ────────────────────────────────
# 1. 도시·SKU 정의
# ────────────────────────────────
LocationP_dict = {
    'Washington_DC':'USA','New_York':'USA','Chicago':'USA','Dallas':'USA',
    'Berlin':'DEU','Munich':'DEU','Frankfurt':'DEU','Hamburg':'DEU',
    'Paris':'FRA','Lyon':'FRA','Marseille':'FRA','Toulouse':'FRA',
    'Seoul':'KOR','Busan':'KOR','Incheon':'KOR','Gwangju':'KOR',
    'Tokyo':'JPN','Osaka':'JPN','Nagoya':'JPN','Fukuoka':'JPN',
    'Manchester':'GBR','London':'GBR','Birmingham':'GBR','Glasgow':'GBR',
    'Ottawa':'CAN','Toronto':'CAN','Vancouver':'CAN','Montreal':'CAN',
    'Canberra':'AUS','Sydney':'AUS','Melbourne':'AUS','Brisbane':'AUS',
    'Brasilia':'BRA','Sao_Paulo':'BRA','Rio_de_Janeiro':'BRA','Salvador':'BRA',
    'Pretoria':'ZAF','Johannesburg':'ZAF','Cape_Town':'ZAF','Durban':'ZAF'
}
cities = list(LocationP_dict.keys())
skus   = [f"SKU{str(i).zfill(4)}" for i in range(1, 26)]

# ────────────────────────────────
# 2. 수요 데이터 로드
# ────────────────────────────────
def load_demand_data(db_path: str, sku: str, city: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    query = f'''
        SELECT * FROM demand_train
        WHERE sku="{sku}" AND city="{city}"
        ORDER BY date
    '''
    df = pd.read_sql(query, con)
    con.close()

    if df.empty:
        return df
    df['Date'] = pd.to_datetime(df['Date'])
    first_nonzero_idx = (df['Demand'] != 0).idxmax()
    return df.loc[first_nonzero_idx:].reset_index(drop=True)

# ────────────────────────────────
# 3. STL 기반 이벤트 더미 생성
# ────────────────────────────────
def stl_event_eliminate(
    df, value_col="Demand", resample_rule="D",
    stl_period=365, z_thresh=2.0,
    min_cluster=7, event_gap=90, max_events=10
):
    y = df[value_col].resample(resample_rule).mean().interpolate()
    stl       = STL(y, period=stl_period, robust=True).fit()
    resid     = stl.resid
    sigma     = resid.std()

    # --- 연속 구간 탐색 함수 ---
    def clusters(mask):
        idx = resid[mask].index
        if idx.empty:
            return []
        grp   = (idx.to_series().diff().dt.days > 1).cumsum()
        spans = idx.to_series().groupby(grp).agg(['min', 'max'])
        spans = spans[(spans['max'] - spans['min']).dt.days + 1 >= min_cluster]
        return [tuple(x) for x in spans.to_numpy()]

    pos = clusters(resid >  sigma * z_thresh)
    neg = clusters(resid < -sigma * z_thresh)
    all_clusters = sorted(pos + neg, key=lambda x: x[0])

    # 인접 이벤트 병합
    merged = []
    for s, e in all_clusters:
        if not merged:
            merged.append([s, e]); continue
        gap = (s - merged[-1][1]) / np.timedelta64(1, "D")
        if gap < event_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    clusters = pd.DataFrame(merged, columns=['Start', 'End'])

    # 더미 생성
    y_adj = pd.DataFrame(y.copy())
    for i in range(max_events):
        col           = f'event_{i+1}'
        y_adj[col]    = 0
        if i < len(clusters):
            s, e          = clusters.iloc[i]
            y_adj.loc[s:e, col] = 1
    return y_adj

# ────────────────────────────────
# 4. Prophet 예측 (외생 변수 = 이벤트 더미만)
# ────────────────────────────────
def forecast_with_prophet(df: pd.DataFrame, periods: int = 730) -> pd.DataFrame:
    # 기본 y
    prophet_df = df[['Date', 'Demand']].rename(columns={'Date': 'ds', 'Demand': 'y'})

    model = Prophet()

    # 이벤트 더미만 추가
    event_cols = [c for c in df.columns if c.startswith('event_')]
    for col in event_cols:
        model.add_regressor(col)
    prophet_df = pd.concat([prophet_df, df[event_cols].reset_index(drop=True)], axis=1)

    # 학습
    model.fit(prophet_df)

    # 미래 프레임 (외생 변수 모두 0)
    future = model.make_future_dataframe(periods=periods, freq='D')
    for col in event_cols:
        future[col] = 0
        # 과거 구간은 학습 데이터 더미 복사
        past_mask   = future['ds'].isin(df['Date'])
        future.loc[past_mask, col] = df.set_index('Date')[col].values

    forecast = model.predict(future)
    future_fc = forecast[forecast['ds'] > df['Date'].max()]

    return future_fc[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Demand'})

# ────────────────────────────────
# 5. 실행 루프
# ────────────────────────────────
all_forecasts = []

for sku in skus:
    for city in cities:
        raw = load_demand_data(db_path, sku, city)
        if raw.empty:
            print(f"[skip] {sku} / {city}")
            continue

        # 이벤트 더미 생성
        ts   = raw.set_index('Date')
        ts_d = stl_event_eliminate(ts, min_cluster=2)

        ts_d = ts_d.reset_index()
        ts_d['SKU']  = sku
        ts_d['City'] = city

        # DB에 저장 (옵션)
        with sqlite3.connect(save_db_path) as conn:
            ts_d.to_sql('event_dummy', conn, if_exists='append', index=False)

        # 예측
        fc = forecast_with_prophet(ts_d)
        fc['sku'], fc['city'] = sku, city
        fc = fc.rename(columns={'Date':'date', 'Demand':'mean'})
        all_forecasts.append(fc[['date', 'sku', 'city', 'mean']])

    print(f"processed {sku}")

# ────────────────────────────────
# 6. CSV 출력
# ────────────────────────────────
final = pd.concat(all_forecasts, ignore_index=True)
final.to_csv(os.path.join(base_path, 'forecast_submission_template.csv'),
             index=False, date_format='%Y-%m-%d')