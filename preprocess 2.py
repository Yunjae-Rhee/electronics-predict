import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import timedelta
from statsmodels.tsa.seasonal import STL
from scipy.optimize import curve_fit

# ───────────────────────────────────────────────
# 경로 설정 (프로젝트 루트 기준)
# ───────────────────────────────────────────────
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_PATH, "data")
INPUT_DB  = os.path.join(DATA_DIR, "cleaned_demand.db")
OUTPUT_DB = os.path.join(DATA_DIR, "event_removed_demand.db")

# ───────────────────────────────────────────────
# 도시·SKU 목록 (원본 그대로 유지)
# ───────────────────────────────────────────────
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

# ───────────────────────────────────────────────
# Bass increment & STL 이벤트 제거
# ───────────────────────────────────────────────

def bass_increment(t, p, q, m):
    num = (p + q) ** 2 * np.exp(-(p + q) * t)
    den = p * (1 + (q / p) * np.exp(-(p + q) * t)) ** 2
    return m * (num / den)

def stl_event_eliminate(df: pd.DataFrame,
                        value_col: str = "Demand",
                        resample_rule: str = "D",
                        stl_period: int = 365,
                        min_cluster: int = 3,
                        z_thresh: float = 2.0,
                        event_gap: int = 90,
                        basestart: int = 20,
                        diffusion_horizon: int = 90) -> pd.Series:
    """STL 잔차 기반 이벤트 감지 후 Bass 확산모형으로 영향 제거"""
    y = df[value_col].resample(resample_rule).mean().interpolate()
    stl = STL(y, period=stl_period, robust=True).fit()
    resid = stl.resid
    sigma = resid.std()

    def _clusters(mask):
        idx = resid[mask].index
        if idx.empty:
            return []
        grp = (idx.to_series().diff().dt.days > 1).cumsum()
        spans = idx.to_series().groupby(grp).agg(['min','max'])
        spans = spans[(spans['max'] - spans['min']).dt.days + 1 >= min_cluster]
        return [tuple(x) for x in spans.to_numpy()]

    cand = _clusters(resid >  sigma*z_thresh) + _clusters(resid < -sigma*z_thresh)
    cand.sort(key=lambda x: x[0])

    merged = []
    for s,e in cand:
        if not merged:
            merged.append([s,e]); continue
        prev_s, prev_e = merged[-1]
        # gap 을 일 수(float) 로 환산해 비교
        gap = (s - prev_e) / np.timedelta64(1, "D")   # 일 단위 float
        if gap < event_gap:
            merged[-1][1] = max(prev_e, e)
        else:
            merged.append([s, e])
    clusters = pd.DataFrame(merged, columns=["Start","End"])

    effect = pd.Series(0.0, index=y.index)
    for start, _ in clusters.itertuples(index=False):
        S0 = start - timedelta(days=basestart)
        E0 = S0 + timedelta(days=diffusion_horizon-1)
        msk = (y.index >= S0) & (y.index <= E0)
        baseline = (stl.trend + stl.seasonal).loc[msk]
        delta_y  = (y - baseline).loc[msk].clip(lower=0)
        if delta_y.sum()==0:
            continue
        t = np.arange(1, len(delta_y)+1)
        try:
            params,_ = curve_fit(bass_increment, t, delta_y.values,
                                 p0=[0.03,0.4,delta_y.sum()], bounds=(0, np.inf), maxfev=10000)
            effect.loc[msk] += bass_increment(t,*params)
        except RuntimeError:
            pass
    return (y - effect).clip(lower=0)

# ───────────────────────────────────────────────
# 메인 처리 루프
# ───────────────────────────────────────────────

# 출력 DB 초기화
with sqlite3.connect(OUTPUT_DB) as conn:
    conn.execute("DROP TABLE IF EXISTS demand_train")

for sku in skus:
    for city in cities:
        with sqlite3.connect(INPUT_DB) as conn:
            q = "SELECT * FROM demand_train WHERE SKU=? AND City=? ORDER BY Date"
            df = pd.read_sql(q, conn, params=[sku,city], parse_dates=['Date'])

        if df.empty:
            continue

        df = df.set_index("Date")
        df = df[df["Demand"].ne(0)]  # 0 이후부터

        adj = stl_event_eliminate(df, value_col="Demand")
        result = pd.DataFrame({
            "Date": adj.index,
            "SKU" : sku,
            "City": city,
            "Demand": adj.values
        })
        with sqlite3.connect(OUTPUT_DB) as conn:
            result.to_sql("demand_train", conn, if_exists="append", index=False)
        print(f"{sku} / {city}  ▶  {len(df)} → {len(result)} rows 저장 완료")

print("✅ 모든 도시·SKU 이벤트 보정 완료")
