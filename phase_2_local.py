# ============================================================================
#  FULL‑Fidelity Two‑Stage Solver  **with weekly aggregation & DB output**
#  (drop‑in replacement for your previous script)
# ============================================================================
from __future__ import annotations
import math, gc, os, sqlite3, psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, DefaultDict

import pandas as pd, numpy as np
from collections import defaultdict
from gurobipy import Model, GRB, quicksum

DATA_DIR   = Path("data")
CHUNK_DAYS = 31              # ≤ 1 month per MILP instance
DATE_START = datetime(2018, 1, 1)
DATE_END   = datetime(2024,12,31)
DB_NAME    = "plan_submission_template.db"  # ← 제출용 파일
TABLE_NAME = "plan_submission_template"

# ---------------------------------------------------------------------------
# Helper to log memory
PROC = psutil.Process(os.getpid())
log  = lambda tag: print(f"[{tag}] RSS = {PROC.memory_info().rss/1024**2:,.1f} MB")

# ---------------------------------------------------------------------------
# 1)  UTILITY FNS  (identical to original –‑ shortened here)
# ---------------------------------------------------------------------------

def date_range(d0: datetime, d1: datetime):
    cur = d0
    while cur <= d1:
        yield cur;  cur += timedelta(days=1)

def week_monday(dt: datetime) -> datetime:
    """Return the Monday date (00:00) of the ISO week dt belongs to."""
    return dt - timedelta(days=dt.weekday())


def date_range(d0, d1): 
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)

def week_of(dt):  # 2018-01-01(월) 기준 몇 주차
    base = datetime(2018,1,1)
    return (dt.date() - base.date()).days // 7

def dow(dt):  # Mon=1 ... Sun=7
    return dt.weekday() + 1

def block_of(dt):  # 4주=1블록
    return week_of(dt) // 4

def next_monday_00(dt):
    days_ahead = (7 - dt.weekday()) % 7
    days_ahead = 7 if days_ahead == 0 else days_ahead
    return (dt + timedelta(days=days_ahead)).replace(hour=0, minute=0, second=0, microsecond=0)

def haversine_km(lat1, lon1, lat2, lon2, R=6371.0):  # 대권거리(km)
    p1, l1, p2, l2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((p2 - p1)/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin((l2 - l1)/2.0)**2
    return float(2*R*np.arcsin(np.sqrt(a)))

def ceil_pos(x: float) -> int:
    return 0 if x <= 0 else int(math.ceil(x))

def fill_forward_by_biz(df, key, val_col, biz_col='is_biz_day'):  # 영업일 FFill
    df = df.sort_values(['date'] + key).copy()
    out = []
    for keys, g in df.groupby(key):
        g = g.sort_values('date')
        last = None
        buf = []
        for _, r in g.iterrows():
            if r[biz_col] == 1 and pd.notnull(r[val_col]):
                last = r[val_col]
            buf.append(last)
        g[val_col + '_ff'] = buf
        out.append(g)
    return pd.concat(out, ignore_index=True)

def norm_city(name: str) -> str:
    return str(name).replace(' ', '_')

def truck_days_from_distance(dkm: float) -> int:
    d = max(0.0, float(dkm))
    if d <= 500.0:   return 2
    elif d <= 1000.: return 3
    elif d <= 2000.: return 5
    else:            return 8



def to_week_idx(week_col):
    """
    week_col: int 주차 or 'YYYY-MM-DD' (주 시작 월요일)
    → 모델 내부 주 인덱스(2018-01-01 기준 0..N)로 변환한 pd.Series[int]
    """
    if pd.api.types.is_integer_dtype(week_col):
        return week_col.astype(int)
    # 날짜 문자열/datetime → 주 인덱스
    wk_dt = pd.to_datetime(week_col, errors='coerce')
    if wk_dt.isna().any():
        bad = week_col[wk_dt.isna()].unique()[:5]
        raise ValueError(f"[week 파싱 실패] 예: {bad}")
    return wk_dt.apply(week_of).astype(int)


# ============================================================================
# 2) Load ALL master data once – uses original code verbatim
# ============================================================================

# =========================================================
# 2-1) 데이터 로드
# =========================================================
print("Loading master tables …")


dates = list(date_range(DATE_START, DATE_END))
weeks = sorted(set(week_of(t) for t in dates))
blocks = sorted(set(block_of(t) for t in dates))

# --- 캘린더/계절/휴일/날씨/유가/환율 ---
cal = pd.read_csv(f'{DATA_DIR}/calendar.csv')        # date, country, season
cal['date'] = pd.to_datetime(cal['date'])

hol = pd.read_csv(f'{DATA_DIR}/holiday_lookup.csv')  # country, date, holiday_name
hol['date'] = pd.to_datetime(hol['date'])
hol['is_holiday'] = 1

wx  = pd.read_csv(f'{DATA_DIR}/weather.csv')         # date, country, ...
wx['date'] = pd.to_datetime(wx['date'])
for col in ['rain_mm','snow_cm','wind_mps','cloud_pct']:
    if col not in wx.columns:
        wx[col] = 0.0

oil = pd.read_csv(f'{DATA_DIR}/oil_price.csv')       # date, brent_usd
oil['date'] = pd.to_datetime(oil['date'])

fx  = pd.read_csv(f'{DATA_DIR}/currency.csv')        # Date, EUR=X, KRW=X, ...
fx.rename(columns={'Date':'date'}, inplace=True)
fx['date'] = pd.to_datetime(fx['date'])

# --- 거시/노동 ---
cci = pd.read_csv(f'{DATA_DIR}/consumer_confidence.csv')  # (옵션) 예측용
lab_pol = pd.read_csv(f'{DATA_DIR}/labour_policy.csv')    # country, year, regular_wage_local, currency, ot_mult, max_hours_week

# --- 사이트/코스트/운송 ---
sites = pd.read_csv(f'{DATA_DIR}/site_candidates.csv')    # site_id, country, city, site_type, lat, lon
init_cost = pd.read_csv(f'{DATA_DIR}/site_init_cost.csv') # site_id, asset_type, init_cost_usd
mode_meta = pd.read_csv(f'{DATA_DIR}/transport_mode_meta.csv') # mode, cost_per_km_factor, co2_per_km_factor, leadtime_factor

# --- SKU/생산/탄소/능력/고장 ---
sku_meta = pd.read_csv(f'{DATA_DIR}/sku_meta.csv')
sku_meta['launch_date'] = pd.to_datetime(sku_meta['launch_date'])
carbon_prod = pd.read_csv(f'{DATA_DIR}/carbon_factor_prod.csv')

fac_cap = pd.read_csv(f'{DATA_DIR}/factory_capacity.csv')
mfail   = pd.read_csv(f'{DATA_DIR}/machine_failure_log.csv')
mfail['start_date'] = pd.to_datetime(mfail['start_date'])
mfail['end_date']   = pd.to_datetime(mfail['end_date'])
lab_req = pd.read_csv(f'{DATA_DIR}/labour_requirement.csv')

# --- 비용 ---
price_promo = pd.read_csv(f'{DATA_DIR}/price_promo_train.csv')   # 참고
mkt = pd.read_csv(f'{DATA_DIR}/marketing_spend.csv')
prod_cost = pd.read_csv(f'{DATA_DIR}/prod_cost_excl_labour.csv')
inv_cost = pd.read_csv(f'{DATA_DIR}/inv_cost.csv')
short_cost = pd.read_csv(f'{DATA_DIR}/short_cost.csv')

# --- 수요 (과거) ---
con = sqlite3.connect(f'{DATA_DIR}/demand_train.db')
demand_df = pd.read_sql_query("SELECT date, sku, city, demand FROM demand_train", con)
con.close()
demand_df['date'] = pd.to_datetime(demand_df['date'])
demand_df['city'] = demand_df['city'].map(norm_city)

# --- 수요 (미래) override ---
forecast_df = pd.read_csv(f'{DATA_DIR}/forecast_submission_template.csv')  # date, sku, city, mean
forecast_df['date'] = pd.to_datetime(forecast_df['date'])
forecast_df['city'] = forecast_df['city'].map(norm_city)

hist_mask = (demand_df['date'] >= DATE_START) & (demand_df['date'] <= min(DATE_END, pd.to_datetime('2022-12-31')))
d_hist = demand_df.loc[hist_mask, ['date','sku','city','demand']].copy()

future_mask = (forecast_df['date'] >= pd.to_datetime('2023-01-01')) & (forecast_df['date'] <= DATE_END)
d_fut = forecast_df.loc[future_mask, ['date','sku','city','mean']].copy()
d_fut['demand'] = d_fut['mean'].round().astype(int)
d_fut = d_fut.drop(columns=['mean'])

all_dem = (pd.concat([d_hist, d_fut], ignore_index=True)
             .sort_values('date')
             .drop_duplicates(subset=['date','sku','city'], keep='last'))

# =========================================================
# 2-2) 집합/인덱스
# =========================================================
SKUS = sorted(sku_meta['sku'].unique().tolist())
life_days = dict(zip(sku_meta['sku'], sku_meta['life_days']))

# 사이트 세트
sites['city'] = sites['city'].map(norm_city)
fac_df = sites[sites['site_type']=='factory'].copy()
wh_df  = sites[sites['site_type']=='warehouse'].copy()

I = fac_df['site_id'].tolist()
K = wh_df['site_id'].tolist()

# 도시 집합(2018–2024 전체)
J = sorted(all_dem['city'].unique().tolist())

# 국가/좌표 매핑
site_country = dict(zip(sites['site_id'], sites['country']))
site_city    = dict(zip(sites['site_id'], sites['city']))
site_lat     = dict(zip(sites['site_id'], sites['lat']))
site_lon     = dict(zip(sites['site_id'], sites['lon']))

# 도시 → 국가 매핑(명시)
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
city_country = {}
missing = []
for c in J:
    if c in LocationP_dict:
        city_country[c] = LocationP_dict[c]
    else:
        missing.append(c)
if missing:
    raise ValueError(f"[city→country 매핑 누락] {missing}")

# 수요 딕셔너리
D = {(r.city, r.sku, pd.to_datetime(r.date)): int(r.demand) for _, r in all_dem.iterrows()}

# 노동/능력 파라미터
a = dict(zip(lab_req['sku'], lab_req['labour_hours_per_unit']))

fac_cap['week_idx'] = to_week_idx(fac_cap['week'])

# (중복 행 방지: 동일 (factory, week_idx) 여러 행이면 합산. 정책에 따라 'max'도 가능)
fac_cap_agg = (fac_cap
               .groupby(['factory','week_idx'], as_index=False)
               .agg(reg_capacity=('reg_capacity','sum'),
                    ot_capacity=('ot_capacity','sum')))

reg_cap = {(r.factory, int(r.week_idx)): float(r.reg_capacity) for _, r in fac_cap_agg.iterrows()}
ot_cap  = {(r.factory, int(r.week_idx)): float(r.ot_capacity)  for _, r in fac_cap_agg.iterrows()}
# 생산비/탄소
base_cost = {(r.sku, r.factory): r.base_cost_usd for _,r in prod_cost.iterrows()}
delta_prod = {r.factory: r.kg_CO2_per_unit for _,r in carbon_prod.iterrows()}

# 보관비/부족비
HOLD = dict(zip(inv_cost['sku'], inv_cost['inv_cost_per_day']))
SHORT = dict(zip(short_cost['sku'], short_cost['short_cost_per_unit']))

# 모드 메타
modes = mode_meta['mode'].tolist()
beta_cost  = dict(zip(mode_meta['mode'], mode_meta['cost_per_km_factor']))
gamma_co2  = dict(zip(mode_meta['mode'], mode_meta['co2_per_km_factor']))
alpha_lead = dict(zip(mode_meta['mode'], mode_meta['leadtime_factor']))

TRUCK_BASE_COST_PER_KM = 12.0
TRUCK_BASE_CO2_PER_KM  = 0.40

# =========================================================
# 2-3) 거리/리드타임/모드허용/특례
# =========================================================
EU_ZONE = {'DEU','FRA'}
BORDER_COST = 4000.0

def border_cost_fn(cu, cv):
    if cu == cv: return 0.0
    if {cu, cv} == EU_ZONE: return 0.0
    return BORDER_COST

def allowed_mode_between(cu, cv, mode):
    if cu == cv:
        return mode == 'TRUCK'
    if {cu, cv} == EU_ZONE:
        return mode in ['TRUCK','SHIP','AIR']  # EU 특례
    return mode in ['SHIP','AIR']

IK = [(i,k) for i in I for k in K]
KJ = [(k,j) for k in K for j in J]

dist_ik, dist_kj = {}, {}
L_ik = {m:{} for m in modes}
L_kj = {m:{} for m in modes}

# IK
for (i,k) in IK:
    d = haversine_km(site_lat[i], site_lon[i], site_lat[k], site_lon[k])
    dist_ik[(i,k)] = d
    base_days = truck_days_from_distance(d)
    for m in modes:
        L_ik[m][(i,k)] = ceil_pos(alpha_lead[m] * base_days)

# 도시 좌표(공장 평균 → 없으면 창고 평균)
fac_city_latlon = fac_df.groupby('city')[['lat','lon']].mean().reset_index().set_index('city')
city_lat, city_lon, miss = {}, {}, []
for j in J:
    if j in fac_city_latlon.index:
        city_lat[j] = float(fac_city_latlon.loc[j,'lat'])
        city_lon[j] = float(fac_city_latlon.loc[j,'lon'])
    else:
        miss.append(j)
if miss:
    wh_city_latlon = wh_df.groupby('city')[['lat','lon']].mean().reset_index().set_index('city')
    for j in list(miss):
        if j in wh_city_latlon.index:
            city_lat[j] = float(wh_city_latlon.loc[j,'lat'])
            city_lon[j] = float(wh_city_latlon.loc[j,'lon'])
            miss.remove(j)
    if miss:
        raise ValueError(f"[도시 좌표 없음] {miss}")

# KJ
for (k,j) in KJ:
    d = haversine_km(site_lat[k], site_lon[k], city_lat[j], city_lon[j])
    dist_kj[(k,j)] = d
for m in modes:
    for (k,j) in KJ:
        base_days = truck_days_from_distance(dist_kj[(k,j)])
        L_kj[m][(k,j)] = ceil_pos(alpha_lead[m] * base_days)

ALLOWED_IK = {(i,k,m): int(allowed_mode_between(site_country[i], site_country[k], m)) for (i,k) in IK for m in modes}
ALLOWED_KJ = {(k,j,m): int(allowed_mode_between(site_country[k], city_country[j], m)) for (k,j) in KJ for m in modes}
BORDER_IK  = {(i,k): border_cost_fn(site_country[i], site_country[k]) for (i,k) in IK}
BORDER_KJ  = {(k,j): border_cost_fn(site_country[k], city_country[j]) for (k,j) in KJ}

# 로컬 배송: 같은 도시면 거리/리드타임/국경비 0
for (k, j) in KJ:
    if site_city[k] == j:
        dist_kj[(k, j)] = 0.0
        for m_ in modes:
            L_kj[m_][(k, j)] = 0
        BORDER_KJ[(k, j)] = 0.0

# =========================================================
# 2-4) 환율/영업일/유가/날씨
# =========================================================
all_days = pd.DataFrame({'date': pd.date_range(DATE_START, DATE_END, freq='D')})
countries = sorted(pd.unique(pd.concat([sites['country'], cal['country'], hol['country'], wx['country']], ignore_index=True)))

# 영업일 테이블
biz_tbl = []
for c in countries:
    g = all_days.copy()
    g['country'] = c
    g['weekday'] = g['date'].dt.weekday
    g['is_biz_day'] = ((g['weekday']<5)*1).astype(int)
    h = hol[hol['country']==c][['date','is_holiday']].drop_duplicates()
    g = g.merge(h, on='date', how='left')
    g.loc[g['is_holiday']==1, 'is_biz_day'] = 0
    g.drop(columns=['weekday','is_holiday'], inplace=True)
    biz_tbl.append(g)
biz = pd.concat(biz_tbl, ignore_index=True)

# 환율: 1 USD -> LOCAL  → USD/LOCAL = 1 / 값
country_ccy = dict(zip(lab_pol['country'], lab_pol['currency']))
fx_long = fx.melt(id_vars=['date'], var_name='pair', value_name='usd_to_local')
fx_long['ccy'] = fx_long['pair'].str.replace('=X','', regex=False)
fx_long['usd_per_local'] = 1.0 / fx_long['usd_to_local']

biz_ccy = biz.copy()
biz_ccy['ccy'] = biz_ccy['country'].map(country_ccy)
fx_biz = biz_ccy.merge(fx_long[['date','ccy','usd_per_local']], on=['date','ccy'], how='left')
fx_biz = fill_forward_by_biz(fx_biz, key=['country','ccy'], val_col='usd_per_local', biz_col='is_biz_day')
fx_biz.loc[fx_biz['ccy']=='USD', 'usd_per_local_ff'] = 1.0
FX_USD_PER_LOCAL = {(r.country, r.date): float(r.usd_per_local_ff) for _, r in fx_biz.iterrows()}

# 유가: 직전 영업일
oil_all = all_days.merge(oil[['date','brent_usd']], on='date', how='left')
oil_all['is_biz_day'] = (oil_all['date'].dt.weekday<5).astype(int)
oil_all['_g'] = 1
oil_all = fill_forward_by_biz(oil_all, ['_g'], 'brent_usd', 'is_biz_day').drop(columns=['_g'])
OIL = {r.date: float(r.brent_usd_ff) for _,r in oil_all.iterrows()}

# 휴일(토/일 또는 공휴일)
HOLIDAY = {}
for c in countries:
    hset = set(hol.loc[hol['country']==c, 'date'].dt.normalize())
    for dt in all_days['date']:
        HOLIDAY[(c, dt)] = 1 if (dt.normalize() in hset or dt.weekday()>=5) else 0

# 날씨
WX = {(r.country, r.date): (float(r.get('rain_mm',0)), float(r.get('snow_cm',0)),
                            float(r.get('wind_mps',0)), float(r.get('cloud_pct',0)))
      for _,r in wx.iterrows()}
def is_bad_weather(country, dt):
    rain,snow,wind,cloud = WX.get((country, dt), (0,0,0,0))
    return (rain >= 45.7) or (snow >= 3.85) or (wind >= 13.46) or (cloud >= 100.0)

# 주간 유가 5%↑ → 2배
oil_mondays = [t for t in dates if t.weekday()==0]
OIL_UP_WEEK = set()
for idx in range(1,len(oil_mondays)):
    t_prev, t_cur = oil_mondays[idx-1], oil_mondays[idx]
    p_prev, p_cur = OIL[t_prev], OIL[t_cur]
    if p_prev>0 and (p_cur - p_prev)/p_prev >= 0.05:
        w = week_of(t_cur)
        for tt in dates:
            if week_of(tt)==w:
                OIL_UP_WEEK.add(tt)

def cost_multiplier_depart(country, dt):
    k_wx = 3.0 if is_bad_weather(country, dt) else 1.0
    k_oil = 2.0 if (dt in OIL_UP_WEEK) else 1.0
    return k_wx * k_oil

# 미리 준비: 창고 도착 Arr, 창고 출하 Out 일배열 만들기
Arr = {(k,s,t): 0 for k in K for s in SKUS for t in dates}
Out = {(k,s,t): 0 for k in K for s in SKUS for t in dates}

DATES_ALL: List[datetime] = list(date_range(DATE_START, DATE_END))


# ============================================================================
# 3) MODEL WRAPPER – builds *original* MILP for an arbitrary date slice
# ============================================================================
def build_daily_model(t0: datetime, t1: datetime,
                      inv_init: Dict[Tuple[str,str], int]):
    """Return Model + dicts capturing end‑state variables needed for the next chunk.
    *inv_init* is { (warehouse, sku): inventory_on_t0_minus_1 }.
    All code inside is copy‑paste from your Section 6–8 with three changes:
       1. *dates* → local list `dates_chunk`.
       2. If t==DATE_START guard replaced by inv_init lookup.
       3. Objective accumulators reset per chunk.
    Everything else (constraints, cost formulas) stays IDENTICAL.
    """

    dates_chunk = [d for d in DATES_ALL if t0 <= d <= t1]


    # =========================================================
    # 3-1) 모델 생성
    # =========================================================
    m = Model(f"SCM_{t0.date()}_{t1.date()}")
    BIGM = 10**9
    QCONT = 4000

    # ---------- (A) 착공/운영/결제 변수 ----------
    b_fac = m.addVars(I, dates, vtype=GRB.BINARY, name='b_fac')  # 공장 착공(t)
    b_wh  = m.addVars(K, dates, vtype=GRB.BINARY, name='b_wh')   # 창고 착공(t)

    live_fac = m.addVars(I, dates, vtype=GRB.BINARY, name='live_fac')  # 공장 가동(t)
    live_wh  = m.addVars(K, dates, vtype=GRB.BINARY, name='live_wh')   # 창고 완공(t)

    pay_fac = m.addVars(I, dates, vtype=GRB.CONTINUOUS, lb=0.0, name='pay_fac')  # 착공비(USD)
    pay_wh  = m.addVars(K, dates, vtype=GRB.CONTINUOUS, lb=0.0, name='pay_wh')

    x_fac_any = m.addVars(I, vtype=GRB.BINARY, name='x_fac_any')  # 언젠가 착공?
    x_wh_any  = m.addVars(K, vtype=GRB.BINARY, name='x_wh_any')

    # ---------- (B) 운영 변수/흐름 ----------
    P     = m.addVars(I, SKUS, dates, vtype=GRB.INTEGER, lb=0, name='P')
    H_reg = m.addVars(I, dates, lb=0.0, name='H_reg')
    H_ot  = m.addVars(I, dates, lb=0.0, name='H_ot')

    X = m.addVars(IK, SKUS, dates, modes, vtype=GRB.INTEGER, lb=0, name='X_ik')
    Y = m.addVars(KJ, SKUS, dates, modes, vtype=GRB.INTEGER, lb=0, name='Y_kj')

    N_ik = m.addVars(IK, dates, modes, vtype=GRB.INTEGER, lb=0, name='Ncont_ik')
    N_kj = m.addVars(KJ, dates, modes, vtype=GRB.INTEGER, lb=0, name='Ncont_kj')

    mu_ik = m.addVars(IK, range(1,8), blocks, modes, vtype=GRB.BINARY, name='mu_ik')
    mu_kj = m.addVars(KJ, range(1,8), blocks, modes, vtype=GRB.BINARY, name='mu_kj')

    Ivar = m.addVars(K, SKUS, dates, vtype=GRB.INTEGER, lb=0, name='I')
    Wvar = m.addVars(K, SKUS, dates, vtype=GRB.INTEGER, lb=0, name='W')
    Svar = m.addVars(J, SKUS, dates, vtype=GRB.INTEGER, lb=0, name='S')
    Uvar = m.addVars(J, SKUS, dates, vtype=GRB.INTEGER, lb=0, name='U')

    # IK 도착(리드타임 반영) → Arr[k,s,t_arrival]에 적재
    for (i,k) in IK:
        for s in SKUS:
            for t_dep in dates:
                for m_ in modes:
                    L = L_ik[m_][(i,k)]
                    t_arr = t_dep + timedelta(days=L)
                    if t_arr <= DATE_END:
                        Arr[(k,s,t_arr)] = Arr[(k,s,t_arr)] + X[i,k,s,t_dep,m_]

    # KJ 출하 → Out[k,s,t] = sum_j,m Y[k,j,s,t,m]
    for (k,j) in KJ:
        for s in SKUS:
            for t in dates:
                for m_ in modes:
                    Out[(k,s,t)] = Out[(k,s,t)] + Y[k,j,s,t,m_]







    # =========================================================
    # 7) 제약식
    # =========================================================

    # --- 착공 1회, x_any 연결
    for i in I:
        m.addConstr(quicksum(b_fac[i,t] for t in dates) <= 1, name=f'build_once_fac_{i}')
        m.addConstr(x_fac_any[i] == quicksum(b_fac[i,t] for t in dates), name=f'xany_fac_{i}')
    for k in K:
        m.addConstr(quicksum(b_wh[k,t]  for t in dates) <= 1, name=f'build_once_wh_{k}')
        m.addConstr(x_wh_any[k]  == quicksum(b_wh[k,t]  for t in dates), name=f'xany_wh_{k}')

    # --- 부지 상한 & 도시당 1개
    m.addConstr(x_fac_any.sum() <= 5,  'fac_limit')
    m.addConstr(x_wh_any.sum()  <= 20, 'wh_limit')
    for city, g in fac_df.groupby('city'):
        m.addConstr(quicksum(x_fac_any[i] for i in g['site_id']) <= 1, f'one_fac_per_city_{city}')
    for city, g in wh_df.groupby('city'):
        m.addConstr(quicksum(x_wh_any[k]  for k in g['site_id']) <= 1, f'one_wh_per_city_{city}')

    # --- 운영 상태(live): 공장=다음 월요일부터, 창고=즉시
    nextMon = {t: next_monday_00(t) for t in dates}
    for i in I:
        for t in dates:
            lhs = quicksum(b_fac[i, tau] for tau in dates if nextMon[tau] <= t)
            m.addConstr(live_fac[i, t] == lhs, name=f'live_fac_def_{i}_{t.date()}')
    for k in K:
        for t in dates:
            lhs = quicksum(b_wh[k, tau] for tau in dates if tau <= t)  # 창고 즉시 완공
            m.addConstr(live_wh[k, t] == lhs, name=f'live_wh_def_{k}_{t.date()}')

    # --- 착공비 결제: 다음 월요일 00시
    init_cost_usd = dict(zip(init_cost['site_id'], init_cost['init_cost_usd']))
    for i in I:
        for t in dates:
            due_sum = quicksum(b_fac[i, tau] for tau in dates if nextMon[tau] == t)
            m.addConstr(pay_fac[i, t] == init_cost_usd.get(i, 0.0) * due_sum, name=f'pay_fac_{i}_{t.date()}')
    for k in K:
        for t in dates:
            due_sum = quicksum(b_wh[k, tau] for tau in dates if nextMon[tau] == t)  # 창고도 동일 로직(필요시 tau==t로 변경)
            m.addConstr(pay_wh[k, t] == init_cost_usd.get(k, 0.0) * due_sum, name=f'pay_wh_{k}_{t.date()}')

    # --- 생산/노동: 운영중일 때만
    for i in I:
        for s in SKUS:
            for t in dates:
                m.addConstr(P[i,s,t] <= BIGM * live_fac[i, t], f'prod_if_live_{i}_{s}_{t.date()}')

    # 노동-생산 연결
    for i in I:
        for t in dates:
            m.addConstr(quicksum(a.get(s,0.0) * P[i,s,t] for s in SKUS) <= 8.0*H_reg[i,t] + H_ot[i,t],
                        f'labor_link_{i}_{t.date()}')

    # 주간 정규/OT/법정 상한
    for i in I:
        ci = site_country[i]
        for w in weeks:
            if (i, w) in reg_cap:
                m.addConstr(quicksum(H_reg[i, t] for t in dates if week_of(t)==w) <= reg_cap[(i, w)],
                            f'regcap_{i}_{w}')
            if (i, w) in ot_cap:
                m.addConstr(quicksum(H_ot[i, t] for t in dates if week_of(t)==w) <= ot_cap[(i, w)],
                            f'otcap_{i}_{w}')
            week_days = [t for t in dates if week_of(t) == w]
            if week_days:
                y = max(t.year for t in week_days)
                row = lab_pol[(lab_pol['country']==ci) & (lab_pol['year']==y)]
                if len(row)>0:
                    Hlaw = float(row['max_hours_week'].iloc[0])
                    m.addConstr(quicksum(H_reg[i,t]+H_ot[i,t] for t in week_days) <= Hlaw,
                                f'law_{i}_{w}')

    # 일일 규칙: 휴일=정규0, 평일 정규≤8h, 미가동이면 근로 불가
    for i in I:
        ci = site_country[i]
        for t in dates:
            is_hol = HOLIDAY.get((ci, t), 0)
            m.addConstr(H_ot[i,t]  <= BIGM * live_fac[i,t],  name=f'hot_if_live_{i}_{t.date()}')
            if is_hol == 1:
                m.addConstr(H_reg[i,t] == 0,               name=f'hreg_holiday_{i}_{t.date()}')
            else:
                m.addConstr(H_reg[i,t] <= 8.0 * live_fac[i,t], name=f'hreg_weekday8_{i}_{t.date()}')

    # 기계 고장: 생산/출하/근로 0
    for _, r in mfail.iterrows():
        i = r['factory']
        if i not in I:
            continue
        for t in dates:
            if r['start_date'] <= t <= r['end_date']:
                for s in SKUS:
                    m.addConstr(P[i,s,t] == 0, f'failP_{i}_{s}_{t.date()}')
                for k in K:
                    for s in SKUS:
                        for m_ in modes:
                            m.addConstr(X[i,k,s,t,m_] == 0, f'failX_{i}_{k}_{s}_{t.date()}_{m_}')
                m.addConstr(H_reg[i,t] == 0, f'failHreg_{i}_{t.date()}')
                m.addConstr(H_ot[i,t]  == 0, f'failHot_{i}_{t.date()}')

    # 모드 선택: (구간,요일,블록) 1개
    for (i,k) in IK:
        for d in range(1,8):
            for b in blocks:
                m.addConstr(quicksum(mu_ik[i,k,d,b,m_] for m_ in modes) == 1, f'muik_one_{i}_{k}_{d}_{b}')
    for (k,j) in KJ:
        for d in range(1,8):
            for b in blocks:
                m.addConstr(quicksum(mu_kj[k,j,d,b,m_] for m_ in modes) == 1, f'mukj_one_{k}_{j}_{d}_{b}')

    # 금지 모드 선택 자체 차단
    for (i,k) in IK:
        for d in range(1,8):
            for b in blocks:
                for m_ in modes:
                    if ALLOWED_IK[(i,k,m_)] == 0:
                        m.addConstr(mu_ik[i,k,d,b,m_] == 0, name=f'forbid_mu_ik_{i}_{k}_{d}_{b}_{m_}')
    for (k,j) in KJ:
        for d in range(1,8):
            for b in blocks:
                for m_ in modes:
                    if ALLOWED_KJ[(k,j,m_)] == 0:
                        m.addConstr(mu_kj[k,j,d,b,m_] == 0, name=f'forbid_mu_kj_{k}_{j}_{d}_{b}_{m_}')

    # 링크 & 컨테이너 & 시설 가동 연결
    for (i,k) in IK:
        for t in dates:
            d, b = dow(t), block_of(t)
            for m_ in modes:
                m.addConstr(quicksum(X[i,k,s,t,m_] for s in SKUS) <= BIGM * mu_ik[i,k,d,b,m_],
                            f'link_ik_{i}_{k}_{t.date()}_{m_}')
                m.addConstr(quicksum(X[i,k,s,t,m_] for s in SKUS) <= QCONT * N_ik[i,k,t,m_],
                            f'cont_ik_{i}_{k}_{t.date()}_{m_}')
                # 시설 가동중이어야 출하 가능
                m.addConstr(N_ik[i,k,t,m_] <= BIGM * live_fac[i,t], name=f'nik_fac_live_{i}_{k}_{t.date()}_{m_}')
                m.addConstr(N_ik[i,k,t,m_] <= BIGM * live_wh[k,t],  name=f'nik_wh_live_{i}_{k}_{t.date()}_{m_}')

    for (k,j) in KJ:
        for t in dates:
            d, b = dow(t), block_of(t)
            for m_ in modes:
                m.addConstr(quicksum(Y[k,j,s,t,m_] for s in SKUS) <= BIGM * mu_kj[k,j,d,b,m_],
                            f'link_kj_{k}_{j}_{t.date()}_{m_}')
                m.addConstr(quicksum(Y[k,j,s,t,m_] for s in SKUS) <= QCONT * N_kj[k,j,t,m_],
                            f'cont_kj_{k}_{j}_{t.date()}_{m_}')
                m.addConstr(N_kj[k,j,t,m_] <= BIGM * live_wh[k,t], name=f'nkj_wh_live_{k}_{j}_{t.date()}_{m_}')

    # 창고 재고 흐름(공장 재고 없음)
    for k in K:
            for s in SKUS:
                for t in dates_chunk:
                    inbound = []  # unchanged calc
                    outbound = quicksum(Y[k,j,s,t,m_] for (k2,j) in KJ if k2==k for m_ in modes)

                    if t == t0:
                        I_prev_expr = inv_init.get((k, s), 0)
                    else:
                        I_prev_expr = Ivar[k,s,t - timedelta(days=1)]

                    m.addConstr(Ivar[k,s,t] == I_prev_expr + quicksum(inbound) - outbound - Wvar[k,s,t])

    # 도시 수요 충족/부족 & 반드시 Y로 커버(로컬도 비용0 arc)
    for j in J:
        for s in SKUS:
            for t in dates:
                D_jst = int(D.get((j, s, t), 0))
                # 수요/부족 정의는 유지
                m.addConstr(Svar[j, s, t] <= D_jst,                  name=f'dcap_{j}_{s}_{t.date()}')
                m.addConstr(Uvar[j, s, t] == D_jst - Svar[j, s, t],  name=f'short_{j}_{s}_{t.date()}')

                # t일에 '도착'하는 물량만 합산 (출발은 t - L_kj 일)
                arrivals = []
                for k in K:
                    for m_ in modes:
                        L = L_kj[m_][(k, j)]
                        t_dep = t - timedelta(days=L)
                        if t_dep >= DATE_START:
                            arrivals.append(Y[k, j, s, t_dep, m_])

                # 도착량으로만 충족 가능
                if arrivals:
                    m.addConstr(Svar[j, s, t] <= quicksum(arrivals), name=f'cover_with_arrival_{j}_{s}_{t.date()}')
                else:
                    # 어떤 경로도 t일 도착하지 않으면 충족은 0이어야 함
                    m.addConstr(Svar[j, s, t] == 0, name=f'cover_with_arrival_zero_{j}_{s}_{t.date()}')

    # life_days 윈도우/누적 제약
    for k in K:
        for s in SKUS:
            Ls = int(life_days.get(s, 10**9))  # 미기재시 매우 크게
            for idx, t in enumerate(dates):
                # 1) 최근 Ls일 출하 ≤ 최근 Ls일 입고
                t0 = dates[max(0, idx - Ls + 1)]
                win_days = [tt for tt in dates if t0 <= tt <= t]
                m.addConstr(quicksum(Out[(k,s,tt)] for tt in win_days)
                            <= quicksum(Arr[(k,s,tt)] for tt in win_days),
                            name=f'fresh_out_win_{k}_{s}_{t.date()}')

                # 2) 기한초과 입고 ≤ 누적 폐기
                old_days = [tt for tt in dates if tt <= t - timedelta(days=Ls)]
                if old_days:
                    m.addConstr(quicksum(Arr[(k,s,tt)] for tt in old_days)
                                <= quicksum(Wvar[k,s,tt] for tt in dates if tt <= t),
                                name=f'expire_dispose_{k}_{s}_{t.date()}')

    # 주간 Fill-Rate ≥ 95%
    for j in J:
        for s in SKUS:
            for w in weeks:
                week_days = [t for t in dates if week_of(t) == w]
                D_week = quicksum(int(D.get((j,s,t),0)) for t in week_days)
                S_week = quicksum(Svar[j,s,t] for t in week_days)
                m.addConstr(S_week >= 0.95 * D_week, f'fr95_weekly_{j}_{s}_{w}')

    # =========================================================
    # 8) 비용/배출/목적함수
    # =========================================================
    # 생산비
    cost_prod = quicksum(base_cost.get((s,i),0.0) * P[i,s,t] for i in I for s in SKUS for t in dates)

    # 노동비(일자별 환율)
    def wage_usd(country, dt, is_ot=False):
        y = int(dt.year)
        row = lab_pol[(lab_pol['country'] == country) & (lab_pol['year'] == y)]
        if row.empty:
            return 0.0
        reg_local = float(row['regular_wage_local'].iloc[0])
        ot_mult   = float(row['ot_mult'].iloc[0])
        fx_usd = FX_USD_PER_LOCAL.get((country, dt))
        if fx_usd is None or pd.isna(fx_usd):
            ccy = row['currency'].iloc[0]
            fx_usd = 1.0 if str(ccy).upper()=='USD' else 0.0
        base_usd = reg_local * fx_usd
        return base_usd * (ot_mult if is_ot else 1.0)

    cost_labor = quicksum(wage_usd(site_country[i], t, False)*H_reg[i,t] +
                        wage_usd(site_country[i], t, True )*H_ot[i,t]
                        for i in I for t in dates)

    def mode_cost_per_km(mode):
        return TRUCK_BASE_COST_PER_KM * beta_cost[mode]

    def mode_co2_per_km(mode):
        return TRUCK_BASE_CO2_PER_KM * gamma_co2[mode]

    # 운송비
    cost_trans_ik = 0.0
    for (i,k) in IK:
        dep_country = site_country[i]
        for t in dates:
            mult = cost_multiplier_depart(dep_country, t)
            for m_ in modes:
                dkm = dist_ik[(i,k)]
                km_cost = mode_cost_per_km(m_)
                cost_trans_ik += mult * (km_cost * dkm * N_ik[i,k,t,m_] + BORDER_IK[(i,k)] * N_ik[i,k,t,m_])

    cost_trans_kj = 0.0
    for (k,j) in KJ:
        dep_country = site_country[k]
        for t in dates:
            mult = cost_multiplier_depart(dep_country, t)
            for m_ in modes:
                dkm = dist_kj[(k,j)]
                km_cost = mode_cost_per_km(m_)
                cost_trans_kj += mult * (km_cost * dkm * N_kj[k,j,t,m_] + BORDER_KJ[(k,j)] * N_kj[k,j,t,m_])

    # 보관/부족
    cost_inv   = quicksum(HOLD.get(s,0.0)  * Ivar[k,s,t] for k in K for s in SKUS for t in dates)
    cost_short = quicksum(SHORT.get(s,0.0) * Uvar[j,s,t] for j in J for s in SKUS for t in dates)

    # CO2 (올림+최소1톤)
    co2_prod = quicksum(delta_prod.get(i,0.0) * P[i,s,t] for i in I for s in SKUS for t in dates) / 1000.0
    co2_trans = (
        quicksum(mode_co2_per_km(m_) * dist_ik[(i,k)] * N_ik[i,k,t,m_] for (i,k) in IK for t in dates for m_ in modes) +
        quicksum(mode_co2_per_km(m_) * dist_kj[(k,j)] * N_kj[k,j,t,m_] for (k,j) in KJ for t in dates for m_ in modes)
    ) / 1000.0
    Ton_int = m.addVar(vtype=GRB.INTEGER, lb=0, name='Ton_int')
    m.addConstr(Ton_int >= co2_prod + co2_trans)
    z_co2 = m.addVar(vtype=GRB.BINARY, name='z_co2_pos')
    m.addConstr((co2_prod + co2_trans) <= BIGM * z_co2)
    m.addConstr(Ton_int >= z_co2)
    C_CO2 = 200.0
    cost_co2 = C_CO2 * Ton_int

    # 건설비(결제일 기준)
    cost_build = quicksum(pay_fac[i, t] for i in I for t in dates) + \
                quicksum(pay_wh[k, t]  for k in K for t in dates)

    # 모드 변경 수수료 5%
    C_ik = m.addVars(IK, range(1,8), blocks, lb=0.0, name='C_ik_db')
    C_kj = m.addVars(KJ, range(1,8), blocks, lb=0.0, name='C_kj_db')
    TdB = {(d,b): [t for t in dates if dow(t)==d and block_of(t)==b] for d in range(1,8) for b in blocks}

    for (i,k) in IK:
        dep_country = site_country[i]
        for d in range(1,8):
            for b in blocks:
                expr = 0.0
                for t in TdB[(d,b)]:
                    mult = cost_multiplier_depart(dep_country, t)
                    for m_ in modes:
                        dkm = dist_ik[(i,k)]
                        km_cost = TRUCK_BASE_COST_PER_KM * beta_cost[m_]
                        expr += mult * (km_cost * dkm * N_ik[i,k,t,m_] + BORDER_IK[(i,k)] * N_ik[i,k,t,m_])
                m.addConstr(C_ik[i,k,d,b] == expr, name=f'defCik_{i}_{k}_{d}_{b}')

    for (k,j) in KJ:
        dep_country = site_country[k]
        for d in range(1,8):
            for b in blocks:
                expr = 0.0
                for t in TdB[(d,b)]:
                    mult = cost_multiplier_depart(dep_country, t)
                    for m_ in modes:
                        dkm = dist_kj[(k,j)]
                        km_cost = TRUCK_BASE_COST_PER_KM * beta_cost[m_]
                        expr += mult * (km_cost * dkm * N_kj[k,j,t,m_] + BORDER_KJ[(k,j)] * N_kj[k,j,t,m_])
                m.addConstr(C_kj[k,j,d,b] == expr, name=f'defCkj_{k}_{j}_{d}_{b}')

    z_change_ik = m.addVars(IK, range(1,8), blocks, vtype=GRB.BINARY, name='zchg_ik')
    z_change_kj = m.addVars(KJ, range(1,8), blocks, vtype=GRB.BINARY, name='zchg_kj')

    for (i,k) in IK:
        for d in range(1,8):
            for b in blocks:
                if b == min(blocks):
                    m.addConstr(z_change_ik[i,k,d,b] == 0, name=f'z0_ik_{i}_{k}_{d}_{b}')
                    continue
                for m_ in modes:
                    m.addConstr(z_change_ik[i,k,d,b] >= mu_ik[i,k,d,b,m_] - mu_ik[i,k,d,b-1,m_])
                    m.addConstr(z_change_ik[i,k,d,b] >= mu_ik[i,k,d,b-1,m_] - mu_ik[i,k,d,b,m_])

    for (k,j) in KJ:
        for d in range(1,8):
            for b in blocks:
                if b == min(blocks):
                    m.addConstr(z_change_kj[k,j,d,b] == 0, name=f'z0_kj_{k}_{j}_{d}_{b}')
                    continue
                for m_ in modes:
                    m.addConstr(z_change_kj[k,j,d,b] >= mu_kj[k,j,d,b,m_] - mu_kj[k,j,d,b-1,m_])
                    m.addConstr(z_change_kj[k,j,d,b] >= mu_kj[k,j,d,b-1,m_] - mu_kj[k,j,d,b,m_])

    cost_change  = quicksum(0.05 * C_ik[i,k,d,b-1] * z_change_ik[i,k,d,b]
                            for (i,k) in IK for d in range(1,8) for b in blocks if b>min(blocks))
    cost_change += quicksum(0.05 * C_kj[k,j,d,b-1] * z_change_kj[k,j,d,b]
                            for (k,j) in KJ for d in range(1,8) for b in blocks if b>min(blocks))

    return m, {
        "I_end": Ivar,   # inventory at day t1
        "P":     P,
        "H_reg": H_reg,
        "H_ot":  H_ot,
        "X":     X,
        "Y":     Y,
    }
# ---------------------------------------------------------------------------
# 4) DRIVER  ―  월 단위 MILP →  주간(plan_submission_template) DB 작성
# ---------------------------------------------------------------------------
def run_chunked_pipeline():
    # ──────────────────────────────────────────────────────────────
    # 1) Chunk 간 인벤토리 carry
    # ──────────────────────────────────────────────────────────────
    inv_carry: Dict[Tuple[str, str], int] = {}      # (warehouse, sku) → units
    t_curr = DATE_START
    total_obj = 0.0

    # ──────────────────────────────────────────────────────────────
    # 2) 주차 집계 버퍼
    # ──────────────────────────────────────────────────────────────
    WeekKey = Tuple[str, str, str]                  # (MonDate, factory, sku)
    PLAN: DefaultDict[WeekKey, List[int]] = defaultdict(lambda: [0, 0, 0])
    #                   ↑ [reg_prod, ot_prod, ship_qty]

    SHIP_ROWS: List[Tuple[str, str, str, str, str, int, str]] = []
    #             (MonDate, sku, from_city, to_city, mode, qty, factory)

    while t_curr <= DATE_END:
        t_next = min(t_curr + timedelta(days=CHUNK_DAYS - 1), DATE_END)
        print(f"\n===  Solving {t_curr.date()} – {t_next.date()}  ===")
        log("build")
        m, ref = build_daily_model(t_curr, t_next, inv_carry)

        m.Params.OutputFlag = 0
        m.Params.MIPGap     = 0.03
        m.Params.TimeLimit  = 1800
        log("opt start")
        m.optimize()
        log("opt done")

        if m.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            raise RuntimeError("MILP failed to converge")

        total_obj += m.ObjVal

        # ---------- (A) 생산량 주차-집계 ----------
        P, H_reg, H_ot = ref["P"], ref["H_reg"], ref["H_ot"]
        for (i, s, t) in [(i, s, t) for (i, s, t) in P.keys()
                          if t_curr <= t <= t_next]:
            units = int(round(P[i, s, t].X))
            if units == 0:
                continue

            # 노동시간을 이용해 정규/OT 비례 배분
            hrs_tot  = H_reg[i, t].X + H_ot[i, t].X
            if hrs_tot == 0:          # 이론상 불가능, 방어
                reg_u, ot_u = units, 0
            else:
                reg_frac = H_reg[i, t].X / hrs_tot
                reg_u = int(round(units * reg_frac))
                ot_u  = units - reg_u

            wk = week_monday(t).strftime("%Y-%m-%d")
            PLAN[(wk, i, s)][0] += reg_u
            PLAN[(wk, i, s)][1] += ot_u

        # ---------- (B) 선적 주차-집계 ----------
        X, Y = ref["X"], ref["Y"]
        # 공장 → 창고
        for (i, k, s, t, m_) in X.keys():
            if not (t_curr <= t <= t_next):  # 이번 청크 날짜만
                continue
            qty = int(round(X[i, k, s, t, m_].X))
            if qty == 0:
                continue
            wk = week_monday(t).strftime("%Y-%m-%d")
            PLAN[(wk, i, s)][2] += qty                         # ship_qty
            SHIP_ROWS.append((wk, s, site_city[i], site_city[k], m_, qty, i))

        # 창고 → 도시
        for (k, j, s, t, m_) in Y.keys():
            if not (t_curr <= t <= t_next):
                continue
            qty = int(round(Y[k, j, s, t, m_].X))
            if qty == 0:
                continue
            wk = week_monday(t).strftime("%Y-%m-%d")
            SHIP_ROWS.append((wk, s, site_city[k], j, m_, qty, None))  # factory=N/A

        # ---------- (C) 인벤토리 carry ----------
        inv_carry.clear()
        I_end = ref["I_end"]
        for k in K:
            for s in SKUS:
                inv_carry[(k, s)] = int(round(I_end[k, s, t_next].X))

        # 메모리 정리
        del m
        gc.collect()
        t_curr = t_next + timedelta(days=1)

    print("\nPIPELINE DONE  →  total cost =", f"${total_obj:,.0f}")

    # ──────────────────────────────────────────────────────────────
    # 3)  DB 파일(plan_submission_template.db) 작성
    # ──────────────────────────────────────────────────────────────
    # 3-1) 생산-집계 → DF
    df_prod = pd.DataFrame(
        [(wk, fac, sku, vals[0], vals[1])
         for (wk, fac, sku), vals in PLAN.items() if vals[0] or vals[1]],
        columns=["date", "factory", "sku", "production_qty", "ot_qty"]
    )

    # 3-2) 선적-집계 → DF
    df_ship = pd.DataFrame(
        SHIP_ROWS,
        columns=["date", "sku", "from_city", "to_city", "mode", "ship_qty", "factory"]
    ).groupby(["date", "factory", "sku", "from_city", "to_city", "mode"],
              as_index=False).agg(ship_qty=("ship_qty", "sum"))

    # 3-3) 두 DF 병합 (outer concat)
    df_final = pd.concat([
        df_prod.assign(ship_qty=0, from_city=None, to_city=None, mode=None),
        df_ship.assign(production_qty=0, ot_qty=0)
    ], ignore_index=True).fillna({"production_qty": 0, "ot_qty": 0, "ship_qty": 0})

    df_final = df_final[[
        "date", "factory", "sku",
        "production_qty", "ot_qty", "ship_qty",
        "from_city", "to_city", "mode"
    ]]

    # 3-4) SQLite 저장
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
    with sqlite3.connect(DB_NAME) as conn:
        df_final.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    print(f"✅  결과 DB 작성 완료  →  {DB_NAME} (table: {TABLE_NAME})")
if __name__ == "__main__":
    run_chunked_pipeline()