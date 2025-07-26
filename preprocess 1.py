import os
import sqlite3
import pandas as pd
import numpy as np

# 1. 경로 정의
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_PATH, "data")
INPUT_DB  = os.path.join(DATA_DIR, "demand_train.db")
OUTPUT_DB = os.path.join(DATA_DIR, "cleaned_demand.db")

# 2. 도시-국가 매핑
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

# 3. SKU 리스트 정의
skus = ['SKU' + str(i).zfill(4) for i in range(1, 26)]

# 4. 수요 데이터 불러오기 함수
def load_demand_data(db_path: str, sku: str, city: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    query = f'''
        SELECT * FROM demand_train
        WHERE sku="{sku}" AND city="{city}"
        ORDER BY date
    '''
    rows = con.execute(query).fetchall()
    con.close()

    df = pd.DataFrame(rows, columns=['Date', 'SKU', 'City', 'Demand'])
    if df.empty:
        return df
    first_nonzero_idx = (df["Demand"] != 0).idxmax()
    df = df.loc[first_nonzero_idx:]
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# 5. Z-score 계산 함수
def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() != 0 else 0,
        raw=False
    )

# 6. 이상치 제거 함수
def remove_outliers_from_timeseries(
    df: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    window: int,
    z_thresh: float,
    resample_freq: str | None = None,
    fill_method: str | None = None,
    method: str = 'nan'
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    if resample_freq:
        df = df.set_index(date_col).asfreq(resample_freq, method=fill_method).reset_index()

    for col in value_cols:
        z = rolling_zscore(df[col], window)
        outlier_mask = z.abs() > z_thresh

        if method == 'nan':
            df.loc[outlier_mask, col] = np.nan
        elif method == 'interpolate':
            df.loc[outlier_mask, col] = np.nan
            df[col] = df[col].interpolate()
        else:
            raise ValueError("method must be 'nan' or 'interpolate'")
    return df

# 7. 이상치 플래그 계산 함수 (선택적으로 사용 가능)
def compute_time_series_flags(
    df: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    window: int,
    z_thresh: float,
    resample_freq: str | None = None,
    fill_method: str | None = None,
    filter_flags: bool = False
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    if resample_freq:
        orig_dates = set(df[date_col].dt.strftime('%Y-%m-%d'))
        df = df.set_index(date_col).asfreq(resample_freq, method=fill_method).reset_index()
        df['is_filled'] = (~df[date_col].dt.strftime('%Y-%m-%d').isin(orig_dates)).astype(int)
    
    out = pd.DataFrame({date_col: df[date_col]})
    for col in value_cols:
        z = rolling_zscore(df[col], window)
        out[f'{col}_flag'] = (z.abs() > z_thresh).astype(int)
    
    if filter_flags:
        flag_cols = [f'{col}_flag' for col in value_cols]
        out = out[out[flag_cols].any(axis=1)].reset_index(drop=True)
    
    return out

# 8. 전체 처리 루프
for sku in skus:
    for city in cities:
        demand_df = load_demand_data(INPUT_DB, sku, city)
        if demand_df.empty:
            print(f"No data for {sku} in {city}, skipping.")
            continue

        cleaned_df = remove_outliers_from_timeseries(
            df=demand_df,
            date_col='Date',
            value_cols=['Demand'],
            window=7,
            z_thresh=2,
            resample_freq='D',
            fill_method='ffill',
            method='interpolate'
        )

        with sqlite3.connect(OUTPUT_DB) as con:
            con.execute("DELETE FROM demand_train WHERE SKU=? AND City=?", (sku, city))
            cleaned_df.to_sql('demand_train', con, if_exists='append', index=False)

        print(f"Processed {sku} in {city}: {len(demand_df)} -> {len(cleaned_df)} rows")