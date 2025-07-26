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
data_dir = os.path.join(base_path, "data")

db_path = os.path.join(data_dir, "cleaned_demand.db")
save_db_path = os.path.join(data_dir, "event_dummy.db")
CONF_CSV = os.path.join(data_dir, "consumer_confidence.csv")
OIL_CSV = os.path.join(data_dir, "oil_price.csv")
PRICE_CSV = os.path.join(data_dir, "price_promo_train.csv")

# ────────────────────────────────
# 1. 기본 정의
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
skus = [f'SKU{str(i).zfill(4)}' for i in range(1, 26)]

# ────────────────────────────────
# 2. 수요 데이터 불러오기 함수
# ────────────────────────────────
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
    df['Date'] = pd.to_datetime(df['Date'])
    first_nonzero_idx = (df["Demand"] != 0).idxmax()
    df = df.loc[first_nonzero_idx:]
    return df

# ────────────────────────────────
# 3. STL + Bass 기반 이벤트 제거
# ────────────────────────────────
def bass_increment(t, p, q, m):
    num = (p + q) ** 2 * np.exp(-(p + q) * t)
    den = p * (1 + (q / p) * np.exp(-(p + q) * t)) ** 2
    return m * (num / den)

def stl_event_eliminate(df, value_col="Demand", resample_rule="D", stl_period=365, min_cluster=7,
                        z_thresh=2.0, event_gap=90, basestart=20, diffusion_horizon=90):
    y = df[value_col].resample(resample_rule).mean().interpolate()
    stl = STL(y, period=stl_period, robust=True).fit()
    resid = stl.resid
    resid_sigma = resid.std()

    def contiguous_clusters(mask):
        idx = resid[mask].index
        if idx.empty:
            return []
        grp = (idx.to_series().diff().dt.days > 1).cumsum()
        spans = idx.to_series().groupby(grp).agg(['min', 'max'])
        spans = spans[(spans['max'] - spans['min']).dt.days + 1 >= min_cluster]
        return [tuple(x) for x in spans[['min', 'max']].to_numpy()]

    pos_clusters = contiguous_clusters(resid > z_thresh * resid_sigma)
    neg_clusters = contiguous_clusters(resid < -z_thresh * resid_sigma)
    all_clusters = sorted(pos_clusters + neg_clusters, key=lambda x: x[0])

    merged = []
    for s, e in all_clusters:
        if not merged:
            merged.append([s, e])
            continue
        prev_s, prev_e = merged[-1]
        gap = (s - prev_e) / np.timedelta64(1, "D")
        if gap < event_gap:
            merged[-1][1] = max(prev_e, e)
        else:
            merged.append([s, e])
    clusters = pd.DataFrame(merged, columns=["Start", "End"])

    y_adj = pd.DataFrame(y.copy())
    for i in range(10):
        col = f'event_{i+1}'
        y_adj[col] = 0
        if i < len(clusters):
            s, e = clusters.iloc[i]
            mask = (y_adj.index >= s) & (y_adj.index <= e)
            y_adj.loc[mask, col] = 1

    return y_adj

# ────────────────────────────────
# 4. 외생 변수 로드
# ────────────────────────────────
cci = pd.read_csv(CONF_CSV, parse_dates=['month']) \
    .rename(columns={'month': 'Date', 'confidence_index': 'cci'}) \
    .groupby('Date', as_index=False)['cci'].mean() \
    .set_index('Date').resample('D').ffill()

oil = pd.read_csv(OIL_CSV, parse_dates=['date']) \
    .rename(columns={'date': 'Date', 'brent_usd': 'oil_price'}) \
    .set_index('Date').resample('D').ffill()

price = pd.read_csv(PRICE_CSV, parse_dates=['date']) \
    .rename(columns={'date': 'Date'}) \
    .set_index('Date').groupby(['sku', 'city'])['unit_price'] \
    .resample('D').mean().reset_index()

date_range = pd.date_range(start=cci.index.min(), end=cci.index.max(), freq='D')
full_index = pd.MultiIndex.from_product([skus, cities, date_range], names=['sku', 'city', 'Date'])
full_df = pd.DataFrame(index=full_index).reset_index()
full_df = full_df.merge(cci.reset_index(), on='Date', how='left')
full_df['country'] = full_df['city'].map(LocationP_dict)

oil_country = oil.reset_index().copy()
oil_country['country'] = 'USA'
oil_country_all = []
for country in set(LocationP_dict.values()):
    temp = oil_country.copy()
    temp['country'] = country
    oil_country_all.append(temp)
oil_country_df = pd.concat(oil_country_all, ignore_index=True)
full_df = full_df.merge(oil_country_df[['Date', 'country', 'oil_price']], on=['Date', 'country'], how='left')

full_df = full_df.merge(price, on=['sku', 'city', 'Date'], how='left')
mode_unit_price = price.groupby(['sku', 'city'])['unit_price'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
full_df['unit_price'] = full_df['unit_price'].fillna(full_df.apply(lambda row: mode_unit_price.get((row['sku'], row['city'])), axis=1))

exog_df = full_df[full_df['Date'] <= pd.Timestamp('2022-12-31')].copy()
future_exog = full_df[full_df['Date'] > pd.Timestamp('2022-12-31')].copy()

# ────────────────────────────────
# 5. Prophet 예측 함수
# ────────────────────────────────
def forecast_with_prophet(df: pd.DataFrame, periods: int = 730) -> pd.DataFrame:
    prophet_df = df[['Date', 'Demand']].rename(columns={'Date': 'ds', 'Demand': 'y'})
    model = Prophet()
    model.add_regressor('cci')
    model.add_regressor('oil_price')
    model.add_regressor('unit_price')

    sku = df['SKU'].iloc[0]
    city = df['City'].iloc[0]
    exog_subset = exog_df[(exog_df['sku'] == sku) & (exog_df['city'] == city) & (exog_df['Date'].isin(df['Date']))]
    prophet_df['cci'] = exog_subset['cci'].values
    prophet_df['oil_price'] = exog_subset['oil_price'].values
    prophet_df['unit_price'] = exog_subset['unit_price'].values

    event_cols = [col for col in df.columns if col.startswith('event_')]
    for col in event_cols:
        model.add_regressor(col)
    prophet_df = pd.concat([prophet_df, df[event_cols].reset_index(drop=True)], axis=1)

    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods, freq='D')

    future_exog_subset = future_exog[(future_exog['sku'] == sku) & (future_exog['city'] == city)]
    for col in ['cci', 'oil_price', 'unit_price']:
        future.loc[future['ds'] > df['Date'].max(), col] = future_exog_subset.set_index('Date').reindex(future.loc[future['ds'] > df['Date'].max(), 'ds'])[col].values
        future[col] = future[col].fillna(method='ffill').fillna(0)

    for col in event_cols:
        future[col] = 0
        future.loc[future['ds'] <= df['Date'].max(), col] = df.set_index('Date')[col].values

    forecast = model.predict(future)
    future_forecast = forecast[forecast['ds'] > df['Date'].max()]
    return future_forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Demand'})

# ────────────────────────────────
# 6. 실행 루프
# ────────────────────────────────
all_forecasts = []
for sk in skus:
    for ct in cities:
        demand_df = load_demand_data(db_path=db_path, sku=sk, city=ct)
        demand_df = demand_df.set_index("Date")
        result_df = stl_event_eliminate(df=demand_df, min_cluster=2, value_col="Demand")

        result_df_reset = result_df.reset_index()
        result_df_reset['SKU'] = sk
        result_df_reset['City'] = ct

        with sqlite3.connect(save_db_path) as conn:
            result_df_reset.to_sql('event_dummy', conn, if_exists='append', index=False)

        forecast_df = forecast_with_prophet(result_df_reset, periods=730)
        forecast_df['sku'] = sk
        forecast_df['city'] = ct
        forecast_df['date'] = forecast_df['Date']
        forecast_df['mean'] = forecast_df['Demand']
        all_forecasts.append(forecast_df[['date', 'sku', 'city', 'mean']])
    print(f"processed {sk}")

# 저장
final_df = pd.concat(all_forecasts, ignore_index=True)
final_df.to_csv(os.path.join(base_path, 'forecast_submission_template.csv'), index=False, date_format='%Y-%m-%d')
