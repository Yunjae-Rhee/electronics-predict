# -*- coding: utf-8 -*-
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import sqlite3
from gurobipy import Model, GRB, quicksum

# =========================================================
# 0) 경로/기간 설정
# =========================================================
DATA_DIR = 'data'

DATE_START = datetime(2018, 1, 1)
DATE_END   = datetime(2024,12,31)

# 필요 시 디버그용 축소
# DATE_END = datetime(2018,3,31)

# =========================================================
# 1) 유틸
# =========================================================
def date_range(d0, d1):
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)

def week_of(dt):
    base = datetime(2018,1,1)  # 문제 조건: 월요일
    return (dt.date() - base.date()).days // 7

def dow(dt):  # Mon=1 ... Sun=7
    return dt.weekday() + 1

def block_of(dt):  # 4주=1블록
    return week_of(dt) // 4

def next_monday_00(dt):
    days_ahead = (7 - dt.weekday()) % 7
    days_ahead = 7 if days_ahead == 0 else days_ahead
    return (dt + timedelta(days=days_ahead)).replace(hour=0, minute=0, second=0, microsecond=0)

def haversine_km(lat1, lon1, lat2, lon2, R=6371.0):
    p1, l1, p2, l2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((p2 - p1)/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin((l2 - l1)/2.0)**2
    return float(2*R*np.arcsin(np.sqrt(a)))

def ceil_pos(x):
    if x <= 0: 
        return 0
    eps = 1e-10
    return int(np.ceil(x - eps))

def fill_forward_by_biz(df, key, val_col, biz_col='is_biz_day'):
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

# =========================================================
# 2) 데이터 로드
# =========================================================
dates = list(date_range(DATE_START, DATE_END))
weeks = sorted(set(week_of(t) for t in dates))
blocks = sorted(set(block_of(t) for t in dates))

# --- 캘린더/계절/휴일/날씨/유가/환율 ---
cal = pd.read_csv(f'{DATA_DIR}/calendar.csv')        # date, country, season
cal['date'] = pd.to_datetime(cal['date'])

hol = pd.read_csv(f'{DATA_DIR}/holiday_lookup.csv')  # country, date, holiday_name
hol['date'] = pd.to_datetime(hol['date'])
hol['is_holiday'] = 1

wx  = pd.read_csv(f'{DATA_DIR}/weather.csv')         # date, country, avg_temp, ... (threshold는 문제 정의 사용)
wx['date'] = pd.to_datetime(wx['date'])
# 비/눈/풍속/운량 등 실제 열명이 다르면 여기서 rename 하세요.
# 예시 가정:
# wx columns 예: date,country,rain_mm,snow_cm,wind_mps,cloud_pct
for col in ['rain_mm','snow_cm','wind_mps','cloud_pct']:
    if col not in wx.columns:
        wx[col] = 0.0

oil = pd.read_csv(f'{DATA_DIR}/oil_price.csv')       # date, brent_usd
oil['date'] = pd.to_datetime(oil['date'])

fx  = pd.read_csv(f'{DATA_DIR}/currency.csv')        # Date, EUR=X, KRW=X, ...
fx.rename(columns={'Date':'date'}, inplace=True)
fx['date'] = pd.to_datetime(fx['date'])

# --- 거시/노동 ---
cci = pd.read_csv(f'{DATA_DIR}/consumer_confidence.csv')  # month, country, confidence_index
# 필요 시 수요 보정에 사용

lab_pol = pd.read_csv(f'{DATA_DIR}/labour_policy.csv')    # country, year, regular_wage_local, currency, ot_mult, max_hours_week

# --- 사이트/코스트/운송 ---
sites = pd.read_csv(f'{DATA_DIR}/site_candidates.csv')    # site_id, country, city, site_type, lat, lon
init_cost = pd.read_csv(f'{DATA_DIR}/site_init_cost.csv') # site_id, asset_type, init_cost_usd
mode_meta = pd.read_csv(f'{DATA_DIR}/transport_mode_meta.csv') # mode, cost_per_km_factor, co2_per_km_factor, leadtime_factor

# --- SKU/생산/탄소/능력/고장 ---
sku_meta = pd.read_csv(f'{DATA_DIR}/sku_meta.csv')        # sku, family, storage_gb, colour, life_days, launch_date
sku_meta['launch_date'] = pd.to_datetime(sku_meta['launch_date'])
carbon_prod = pd.read_csv(f'{DATA_DIR}/carbon_factor_prod.csv')  # factory, kg_CO2_per_unit
fac_cap = pd.read_csv(f'{DATA_DIR}/factory_capacity.csv')        # week, factory, reg_capacity, ot_capacity
mfail   = pd.read_csv(f'{DATA_DIR}/machine_failure_log.csv')     # factory, start_date, end_date, machine_id
mfail['start_date'] = pd.to_datetime(mfail['start_date'])
mfail['end_date']   = pd.to_datetime(mfail['end_date'])
lab_req = pd.read_csv(f'{DATA_DIR}/labour_requirement.csv')      # sku, labour_hours_per_unit

# --- 비용 ---
price_promo = pd.read_csv(f'{DATA_DIR}/price_promo_train.csv')   # 2018-2022 (참고)
mkt = pd.read_csv(f'{DATA_DIR}/marketing_spend.csv')
prod_cost = pd.read_csv(f'{DATA_DIR}/prod_cost_excl_labour.csv') # sku, factory, base_cost_usd
inv_cost = pd.read_csv(f'{DATA_DIR}/inv_cost.csv')               # sku, inv_cost_per_day (USD)
short_cost = pd.read_csv(f'{DATA_DIR}/short_cost.csv')           # sku, short_cost_per_unit (USD)

# --- 수요 (sqlite) ---
con = sqlite3.connect(f'{DATA_DIR}/demand_train.db')
demand_df = pd.read_sql_query("SELECT date, sku, city, demand FROM demand_train", con)
con.close()
demand_df['date'] = pd.to_datetime(demand_df['date'])

# =========================================================
# 3) 집합/인덱스
# =========================================================
SKUS = sorted(sku_meta['sku'].unique().tolist())
life_days = dict(zip(sku_meta['sku'], sku_meta['life_days']))

# 사이트 세트
fac_df = sites[sites['site_type']=='factory'].copy()
wh_df  = sites[sites['site_type']=='warehouse'].copy()

I = fac_df['site_id'].tolist()
K = wh_df['site_id'].tolist()

# 도시 세트는 demand의 city를 사용
J = sorted(demand_df['city'].unique().tolist())

# 국가/좌표 매핑
site_country = dict(zip(sites['site_id'], sites['country']))
site_city    = dict(zip(sites['site_id'], sites['city']))
site_lat     = dict(zip(sites['site_id'], sites['lat']))
site_lon     = dict(zip(sites['site_id'], sites['lon']))

# 도시 → 국가 (price_promo/demand 기준으로 별도 매핑 필요 시 city→country 테이블 추가)
# --- 3-x) 도시 → 국가 매핑 주입 ---------------------------------------
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

def norm_city(name: str) -> str:
    """도시명 정규화: 공백→언더스코어, 기타 필요 규칙 추가 가능"""
    return str(name).replace(' ', '_')

# demand_df, price_promo 등 city 컬럼 정규화 (여기선 demand_df만 사용)
demand_df['city'] = demand_df['city'].map(norm_city)

# 사이트 파일에 있는 창고/공장 도시명도 동일 규칙 적용
sites['city']   = sites['city'].map(norm_city)
fac_df['city']  = fac_df['city'].map(norm_city)
wh_df['city']   = wh_df['city'].map(norm_city)

# 도시 집합 재정의(정규화된 이름 기준)
J = sorted(demand_df['city'].unique().tolist())

# 매핑 생성
city_country = {}
missing = []
for c in J:
    if c in LocationP_dict:
        city_country[c] = LocationP_dict[c]
    else:
        missing.append(c)

if missing:
    # 매핑 누락 도시가 있으면 여기서 명확히 실패하게 하거나, 임시 기본값을 넣고 경고
    raise ValueError(f"[city→country 매핑 누락] 다음 도시들에 대한 매핑을 추가하세요: {missing}")

# --- 3-y) KJ(창고→도시) 관련 허용모드/국경비 재계산 ---------------------
def allowed_mode_between(cu, cv, mode):
    if cu == cv:
        return mode == 'TRUCK'
    # DEU↔FRA EU 특례: TRUCK/SHIP/AIR 모두 허용
    if {cu, cv} == {'DEU','FRA'}:
        return mode in ['TRUCK','SHIP','AIR']
    # 그 외 국제: SHIP/AIR만
    return mode in ['SHIP','AIR']

def border_cost(cu, cv, base=4000.0):
    if cu == cv:
        return 0.0
    if {cu, cv} == {'DEU','FRA'}:
        return 0.0
    return base

# KJ 세트는 창고 K × 도시 J 고정. city_country[j]를 사용하도록 다시 정의
ALLOWED_KJ = {(k,j,m): int(allowed_mode_between(site_country[k], city_country[j], m))
              for (k,j) in KJ for m in modes}

BORDER_KJ  = {(k,j): border_cost(site_country[k], city_country[j], base=BORDER_COST)
              for (k,j) in KJ}


# 수요 사전
D = {(r.city, r.sku, pd.to_datetime(r.date)): int(r.demand) for _,r in demand_df.iterrows()}

# 노동/능력 파라미터
a = dict(zip(lab_req['sku'], lab_req['labour_hours_per_unit']))  # h/unit
reg_cap = {(r.factory, int(r.week)): r.reg_capacity for _,r in fac_cap.iterrows()}
ot_cap  = {(r.factory, int(r.week)): r.ot_capacity  for _,r in fac_cap.iterrows()}

# 생산비/탄소
base_cost = {(r.sku, r.factory): r.base_cost_usd for _,r in prod_cost.iterrows()}
delta_prod = {(r.factory): r.kg_CO2_per_unit for _,r in carbon_prod.iterrows()}

# 보관비/부족비
HOLD = dict(zip(inv_cost['sku'], inv_cost['inv_cost_per_day']))
SHORT = dict(zip(short_cost['sku'], short_cost['short_cost_per_unit']))

# 모드 메타
modes = mode_meta['mode'].tolist()
beta_cost = dict(zip(mode_meta['mode'], mode_meta['cost_per_km_factor']))
gamma_co2 = dict(zip(mode_meta['mode'], mode_meta['co2_per_km_factor']))
alpha_lead = dict(zip(mode_meta['mode'], mode_meta['leadtime_factor']))

TRUCK_BASE_COST_PER_KM = 12.0
TRUCK_BASE_CO2_PER_KM  = 0.40  # 필요 시 gamma에 이미 반영되었다면 중복 주의

# =========================================================
# 4) 거리/리드타임/모드허용/특례
# =========================================================
EU_ZONE = set(['DEU','FRA'])
BORDER_COST = 4000.0

def border_cost(cu, cv):
    if cu == cv:
        return 0.0
    if {cu, cv} == EU_ZONE:
        return 0.0
    return BORDER_COST

def allowed_mode_between(cu, cv, mode):
    if cu == cv:
        return mode == 'TRUCK'
    # 국제 기본: SHIP/AIR
    if {cu, cv} == EU_ZONE:
        return mode in ['TRUCK','SHIP','AIR']  # EU 특례
    return mode in ['SHIP','AIR']

# 모든 공장-창고, 창고-도시 후보 구간 생성 (완전연결은 크면 큼)
IK = [(i,k) for i in I for k in K]
KJ = [(k,j) for k in K for j in J]

# 거리/리드타임
dist_ik = {}
dist_kj = {}
L_ik = {m:{} for m in modes}
L_kj = {m:{} for m in modes}

def truck_days_from_distance(dkm: float) -> int:
    """
    하버사인 거리 dkm(km)에 대한 트럭 리드타임(일) 반환.
    구간표:
      [0, 500]   -> 2일
      (500,1000] -> 3일
      (1000,2000]-> 5일
      > 2000     -> 8일
    """
    d = max(0.0, float(dkm))  # 음수 방지
    if d <= 500.0:
        return 2
    elif d <= 1000.0:  # 500km 초과 ~ 1000km 이하
        return 3
    elif d <= 2000.0:  # 1000km 초과 ~ 2000km 이하
        return 5
    else:              # 2000km 초과
        return 8

for (i,k) in IK:
    d = haversine_km(site_lat[i], site_lon[i], site_lat[k], site_lon[k])
    dist_ik[(i,k)] = d
    base_days = truck_days_from_distance(d)
    for m in modes:
        L_ik[m][(i,k)] = ceil_pos(alpha_lead[m] * base_days)

# 창고→도시 거리: 도시 좌표가 없다면, 임시로 창고-도시간 평균거리(대체) → 실제로는 city 좌표 파일 필요
# 본 스켈레톤에서는 "창고도시와 동일 도시"를 상정하려면 별도 city 좌표 입력이 필요합니다.
# 임시: 창고->도시 거리 = 100km (데모). 실서비스에서는 city 좌표 파일을 추가하세요.
# ==== 도시 좌표: 공장(site_type='factory') 좌표로 정의 ====
def norm_city(name: str) -> str:
    return str(name).replace(' ', '_')

# (이미 했다면 중복 제거)
sites['city']   = sites['city'].map(norm_city)
fac_df['city']  = fac_df['city'].map(norm_city)
wh_df['city']   = wh_df['city'].map(norm_city)
demand_df['city'] = demand_df['city'].map(norm_city)

J = sorted(demand_df['city'].unique().tolist())

fac_city_latlon = (
    fac_df.groupby('city')[['lat','lon']]
          .mean()
          .reset_index()
          .set_index('city')
)

city_lat, city_lon, missing = {}, {}, []
for j in J:
    if j in fac_city_latlon.index:
        city_lat[j] = float(fac_city_latlon.loc[j, 'lat'])
        city_lon[j] = float(fac_city_latlon.loc[j, 'lon'])
    else:
        missing.append(j)

if missing:
    # 필요 시 창고 좌표로 보강:
    wh_city_latlon = wh_df.groupby('city')[['lat','lon']].mean().reset_index().set_index('city')
    for j in list(missing):
        if j in wh_city_latlon.index:
            city_lat[j] = float(wh_city_latlon.loc[j, 'lat'])
            city_lon[j] = float(wh_city_latlon.loc[j, 'lon'])
            missing.remove(j)
    if missing:
        raise ValueError(f"[도시 좌표 없음] {missing} → 해당 도시에 공장/창고 좌표를 추가하세요.")

# ==== 창고→도시 거리/리드타임 계산 ====
dist_kj.clear()
for (k, j) in KJ:
    d = haversine_km(site_lat[k], site_lon[k], city_lat[j], city_lon[j])
    dist_kj[(k, j)] = d

for m in modes:
    for (k, j) in KJ:
        base_days = truck_days_from_distance(dist_kj[(k, j)])  # 구간표 사용
        L_kj[m][(k, j)] = ceil_pos(alpha_lead[m] * base_days)

# 모드 허용성
ALLOWED_IK = {(i,k,m): int(allowed_mode_between(site_country[i], site_country[k], m)) for (i,k) in IK for m in modes}
ALLOWED_KJ = {(k,j,m): int(allowed_mode_between(site_country[k], city_country[j], m)) for (k,j) in KJ for m in modes}

BORDER_IK = {(i,k): border_cost(site_country[i], site_country[k]) for (i,k) in IK}
BORDER_KJ = {(k,j): border_cost(site_country[k], city_country[j]) for (k,j) in KJ}

# =========================================================
# 5) 환율/영업일/유가/날씨 (국가 단위)
# =========================================================
# 영업일 플래그: 나라별로 달력에 영업일 열이 없다 → 간단히 토/일만 비영업일로 처리 후 휴일 merge
all_days = pd.DataFrame({'date': pd.date_range(DATE_START, DATE_END, freq='D')})
countries = sorted(pd.unique(pd.concat([sites['country'], cal['country'], hol['country'], wx['country']], ignore_index=True)))

biz_tbl = []
for c in countries:
    g = all_days.copy()
    g['country'] = c
    g['weekday'] = g['date'].dt.weekday
    g['is_biz_day'] = ((g['weekday']<5)*1).astype(int)  # Mon~Fri
    # 실제 공휴일 0 처리
    h = hol[hol['country']==c][['date','is_holiday']].drop_duplicates()
    g = g.merge(h, on='date', how='left')
    g.loc[g['is_holiday']==1, 'is_biz_day'] = 0
    g.drop(columns=['weekday','is_holiday'], inplace=True)
    biz_tbl.append(g)
biz = pd.concat(biz_tbl, ignore_index=True)
# --- 환율 전처리: currency.csv 는 "1 USD -> LOCAL" ---
fx = pd.read_csv(f'{DATA_DIR}/currency.csv')        # Date, EUR=X, KRW=X, ...
fx.rename(columns={'Date':'date'}, inplace=True)
fx['date'] = pd.to_datetime(fx['date'])

# 국가→통화 매핑 (labour_policy.csv 기준)
country_ccy = dict(zip(lab_pol['country'], lab_pol['currency']))

# long 형태로 변환: 1 USD -> LOCAL
fx_long = fx.melt(id_vars=['date'], var_name='pair', value_name='usd_to_local')
fx_long['ccy'] = fx_long['pair'].str.replace('=X','', regex=False)

# USD/LOCAL = 1 / (1 USD -> LOCAL)
fx_long['usd_per_local'] = 1.0 / fx_long['usd_to_local']

# 국가-통화 붙여서 해당 통화의 환율만 조인
biz_ccy = biz.copy()                   # biz: ['date','country','is_biz_day']
biz_ccy['ccy'] = biz_ccy['country'].map(country_ccy)

fx_biz = biz_ccy.merge(
    fx_long[['date','ccy','usd_per_local']],
    on=['date','ccy'], how='left'
)

# 주말/공휴일은 직전 영업일 값으로 채움
fx_biz = fill_forward_by_biz(fx_biz, key=['country','ccy'], val_col='usd_per_local', biz_col='is_biz_day')

# USD 국가(통화가 USD)는 1.0로 고정
fx_biz.loc[fx_biz['ccy']=='USD', 'usd_per_local_ff'] = 1.0

# 조회용 딕셔너리
FX_USD_PER_LOCAL = {(r.country, r.date): float(r.usd_per_local_ff) for _, r in fx_biz.iterrows()}


# 각 국가-날짜에 해당 통화의 usd_per_local만 남기고 직전 영업일로 채움
fx_biz = fx_biz.dropna(subset=['ccy'])
fx_biz = fx_biz.merge(fx_long[['date','ccy','usd_per_local']], on=['date','ccy'], how='left', suffixes=('','_raw'))
fx_biz = fill_forward_by_biz(fx_biz, ['country','ccy'], 'usd_per_local', 'is_biz_day')
FX_USD_PER_LOCAL = {(r.country, r.date): float(r.usd_per_local_ff) for _,r in fx_biz.iterrows()}

# 유가: 직전 영업일 채움(글로벌 지표라 국가무관)
oil_all = all_days.merge(oil[['date','brent_usd']], on='date', how='left')
# 글로벌 영업일: 월~금 기준
oil_all['is_biz_day'] = (oil_all['date'].dt.weekday<5).astype(int)
oil_all = fill_forward_by_biz(oil_all, [], 'brent_usd', 'is_biz_day')
OIL = {r.date: float(r.brent_usd_ff) for _,r in oil_all.iterrows()}

# 날씨: 출발국가 기준의 악천후 트리거
# 임계값 (문제 명시): 비 ≥45.7, 눈 ≥3.85, 풍속 ≥13.46, 구름 =100
WX = {(r.country, r.date): (float(r.get('rain_mm',0)), float(r.get('snow_cm',0)),
                            float(r.get('wind_mps',0)), float(r.get('cloud_pct',0)))
      for _,r in wx.iterrows()}

def is_bad_weather(country, dt):
    rain,snow,wind,cloud = WX.get((country, dt), (0,0,0,0))
    return (rain >= 45.7) or (snow >= 3.85) or (wind >= 13.46) or (cloud >= 100.0)

# 주간 유가 5%↑ → 주간 내내 2배
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

# =========================================================
# 6) 모델 생성
# =========================================================
m = Model('SCM_MILP')

BIGM = 10**9
QCONT = 4000

# -----------------------
# 변수
# -----------------------
# 시설 선택
x_fac = m.addVars(I, vtype=GRB.BINARY, name='x_fac')
x_wh  = m.addVars(K, vtype=GRB.BINARY, name='x_wh')

# 생산/노동
P     = m.addVars(I, SKUS, dates, vtype=GRB.INTEGER, lb=0, name='P')
H_reg = m.addVars(I, dates, lb=0.0, name='H_reg')
H_ot  = m.addVars(I, dates, lb=0.0, name='H_ot')

# 선적(유닛) & 컨테이너(정수)
X = m.addVars(IK, SKUS, dates, modes, vtype=GRB.INTEGER, lb=0, name='X_ik')
Y = m.addVars(KJ, SKUS, dates, modes, vtype=GRB.INTEGER, lb=0, name='Y_kj')

N_ik = m.addVars(IK, dates, modes, vtype=GRB.INTEGER, lb=0, name='Ncont_ik')
N_kj = m.addVars(KJ, dates, modes, vtype=GRB.INTEGER, lb=0, name='Ncont_kj')

# 모드 선택(요일/블록): 1개 고정 (변경수수료는 후속)
mu_ik = m.addVars(IK, range(1,8), blocks, modes, vtype=GRB.BINARY, name='mu_ik')
mu_kj = m.addVars(KJ, range(1,8), blocks, modes, vtype=GRB.BINARY, name='mu_kj')

# 재고/폐기/충족/부족
Ivar = m.addVars(K, SKUS, dates, vtype=GRB.INTEGER, lb=0, name='I')
Wvar = m.addVars(K, SKUS, dates, vtype=GRB.INTEGER, lb=0, name='W')
Svar = m.addVars(J, SKUS, dates, vtype=GRB.INTEGER, lb=0, name='S')
Uvar = m.addVars(J, SKUS, dates, vtype=GRB.INTEGER, lb=0, name='U')

# =========================================================
# 7) 제약식
# =========================================================
# (7-1) 시설 상한
m.addConstr(x_fac.sum() <= 5, 'fac_limit')
m.addConstr(x_wh.sum()  <= 20, 'wh_limit')

# (7-2) 공장 1도시 1개, 창고 1도시 1개 (site_candidates가 city 기준이므로 city별로 묶기)
for city, g in fac_df.groupby('city'):
    m.addConstr(quicksum(x_fac[i] for i in g['site_id']) <= 1, f'one_fac_per_city_{city}')
for city, g in wh_df.groupby('city'):
    m.addConstr(quicksum(x_wh[k] for k in g['site_id']) <= 1, f'one_wh_per_city_{city}')

# (7-3) 생산은 공장 선택 시만
for i in I:
    for s in SKUS:
        for t in dates:
            m.addConstr(P[i,s,t] <= BIGM * x_fac[i], f'prod_if_fac_{i}_{s}_{t.date()}')

# (7-4) 노동-생산 연결 (휴일/8h 초과 OT 정밀화는 후속)
for i in I:
    for t in dates:
        m.addConstr(quicksum(a.get(s,0.0) * P[i,s,t] for s in SKUS) <= 8.0*H_reg[i,t] + H_ot[i,t],
                    f'labor_link_{i}_{t.date()}')

# 주간 능력/노동법
for i in I:
    ci = site_country[i]
    for w in weeks:
        # 공장 능력(정규/OT)
        if (i,w) in reg_cap:
            m.addConstr(quicksum(H_reg[i,t] for t in dates if week_of(t)==w) <= reg_cap[(i,w)], f'regcap_{i}_{w}')
        if (i,w) in ot_cap:
            m.addConstr(quicksum(H_ot[i,t] for t in dates if week_of(t)==w) <= ot_cap[(i,w)],  f'otcap_{i}_{w}')
        # 국가 주당 최대 근로시간: labour_policy.year → 해당 주의 연도 찾아 매핑
        # 간단화: 연도별 상수로 보고 적용
        y = int(datetime(2018,1,1).year + (w // 52))
        row = lab_pol[(lab_pol['country']==ci) & (lab_pol['year']==y)]
        if len(row)>0:
            Hlaw = float(row['max_hours_week'].iloc[0])
            m.addConstr(quicksum(H_reg[i,t]+H_ot[i,t] for t in dates if week_of(t)==w) <= Hlaw,
                        f'law_{i}_{w}')

# (7-5) 모드 선택: 요일/블록당 단일
for (i,k) in IK:
    for d in range(1,8):
        for b in blocks:
            m.addConstr(quicksum(mu_ik[i,k,d,b,m_] for m_ in modes) == 1, f'muik_one_{i}_{k}_{d}_{b}')
for (k,j) in KJ:
    for d in range(1,8):
        for b in blocks:
            m.addConstr(quicksum(mu_kj[k,j,d,b,m_] for m_ in modes) == 1, f'mukj_one_{k}_{j}_{d}_{b}')

# (7-6) 출발일의 요일/블록 모드만 출하 & 허용모드 & 컨테이너 용량
for (i,k) in IK:
    for t in dates:
        d, b = dow(t), block_of(t)
        for m_ in modes:
            if ALLOWED_IK[(i,k,m_)] == 0:
                m.addConstr(quicksum(X[i,k,s,t,m_] for s in SKUS) == 0, f'forbid_ik_{i}_{k}_{t.date()}_{m_}')
            m.addConstr(quicksum(X[i,k,s,t,m_] for s in SKUS) <= BIGM * mu_ik[i,k,d,b,m_],
                        f'link_ik_{i}_{k}_{t.date()}_{m_}')
            m.addConstr(quicksum(X[i,k,s,t,m_] for s in SKUS) <= QCONT * N_ik[i,k,t,m_],
                        f'cont_ik_{i}_{k}_{t.date()}_{m_}')

for (k,j) in KJ:
    for t in dates:
        d, b = dow(t), block_of(t)
        for m_ in modes:
            if ALLOWED_KJ[(k,j,m_)] == 0:
                m.addConstr(quicksum(Y[k,j,s,t,m_] for s in SKUS) == 0, f'forbid_kj_{k}_{j}_{t.date()}_{m_}')
            m.addConstr(quicksum(Y[k,j,s,t,m_] for s in SKUS) <= BIGM * mu_kj[k,j,d,b,m_],
                        f'link_kj_{k}_{j}_{t.date()}_{m_}')
            m.addConstr(quicksum(Y[k,j,s,t,m_] for s in SKUS) <= QCONT * N_kj[k,j,t,m_],
                        f'cont_kj_{k}_{j}_{t.date()}_{m_}')

# (7-7) 재고 흐름 (공장 재고 없음, 입고는 리드타임 후 도착)
for k in K:
    for s in SKUS:
        for t in dates:
            inbound = []
            for (ii,kk) in IK:
                if kk != k: 
                    continue
                for m_ in modes:
                    L = L_ik[m_][(ii,k)]
                    t_dep = t - timedelta(days=L)
                    if t_dep >= DATE_START:
                        inbound.append(X[ii,k,s,t_dep,m_])
            outbound = quicksum(Y[k,j,s,t,m_] for (kk,j) in KJ if kk==k for m_ in modes)
            if t == DATE_START:
                I_prev = 2000 * x_wh[k]  # 초기 2000
            else:
                I_prev = Ivar[k,s,t - timedelta(days=1)]
            m.addConstr(Ivar[k,s,t] == I_prev + quicksum(inbound) - outbound - Wvar[k,s,t],
                        f'inv_{k}_{s}_{t.date()}')

# (7-8) 수요 충족/부족 (도시 내 창고 있으면 무운송 소화 가능 → 비용 0으로 처리)
for j in J:
    for s in SKUS:
        for t in dates:
            D_jst = int(D.get((j,s,t), 0))
            m.addConstr(Svar[j,s,t] <= D_jst, f'dcap_{j}_{s}_{t.date()}')
            m.addConstr(Uvar[j,s,t] == D_jst - Svar[j,s,t], f'short_{j}_{s}_{t.date()}')
            # 외부창고 출하량은 충족분 이상의 흐름을 만들어야 함(도시내 창고 존재시 완화)
            out_from_k = quicksum(Y[k,j,s,t,m_] for (k,jj) in KJ if jj==j for m_ in modes)
            # 도시내 창고 존재여부 (같은 city 이름을 가진 창고 선택)
            has_local_wh = quicksum(x_wh[k] for k in K if site_city[k]==j)
            # 선형화를 위해 큰 M
            m.addConstr(out_from_k + BIGM*has_local_wh >= Svar[j,s,t], f'cover_{j}_{s}_{t.date()}')

# (7-9) 기계 고장 기간: 해당 공장 모든 작업 중지 (생산/출하)
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

# =========================================================
# 8) 비용/배출/목적함수
# =========================================================
# 생산비 (USD)
cost_prod = quicksum(base_cost.get((s,i),0.0) * P[i,s,t] for i in I for s in SKUS for t in dates)

# 노동비: 정규/OT 단가(현지통화) × 해당일 FX(USD/LOCAL)로 환산
# labour_policy: regular_wage_local(시간당 최저) + ot_mult
def wage_usd(country, dt, is_ot=False):
    """
    country: 'USA','KOR'...
    dt: datetime (일자별 환율 적용)
    is_ot: 초과근무 시 True
    """
    # 1) 해당 연도 정책 로드
    y = int(dt.year)
    row = lab_pol[(lab_pol['country'] == country) & (lab_pol['year'] == y)]
    if row.empty:
        return 0.0

    reg_local = float(row['regular_wage_local'].iloc[0])  # 현지통화/시간
    ot_mult   = float(row['ot_mult'].iloc[0])

    # 2) 환율(USD/LOCAL) — 주말/공휴일은 직전 영업일로 FFill된 값이어야 함
    fx_usd = FX_USD_PER_LOCAL.get((country, dt))
    if fx_usd is None or pd.isna(fx_usd):
        # 통화가 USD거나 환율 누락 시: USD 국가면 1.0, 아니면 0.0(보수적)
        # (fx_biz 생성 시 USD는 1.0으로 채우는게 베스트)
        ccy = row['currency'].iloc[0]
        if str(ccy).upper() == 'USD':
            fx_usd = 1.0
        else:
            return 0.0

    base_usd = reg_local * fx_usd  # 시간당 USD
    return base_usd * (ot_mult if is_ot else 1.0)


cost_labor = quicksum(wage_usd(site_country[i], t, False)*H_reg[i,t] +
                      wage_usd(site_country[i], t, True )*H_ot[i,t]
                      for i in I for t in dates)

# 운송비: (거리 × 트럭비용 × 모드배수 × 컨테이너수 + 국경비) × (악천후×유가 계수)
def mode_cost_per_km(mode):
    return TRUCK_BASE_COST_PER_KM * beta_cost[mode]

def mode_co2_per_km(mode):
    return TRUCK_BASE_CO2_PER_KM * gamma_co2[mode]

cost_trans_ik = 0.0
for (i,k) in IK:
    dep_country = site_country[i]
    for t in dates:
        mult = cost_multiplier_depart(dep_country, t)
        for m_ in modes:
            dkm = dist_ik[(i,k)]
            km_cost = mode_cost_per_km(m_)
            # 국경비 정확 모델링(1건/출발)은 z이진 필요 → 여기선 컨테이너당으로 근사
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

# (옵션) 모드 변경수수료 5%: 블록 종료 시 변경 감지 + 직전블록 비용 합 → 후속 확장
cost_change = 0.0  # TODO

# 탄소부담금: 생산 + 운송
co2_prod = quicksum(delta_prod.get(i,0.0) * P[i,s,t] for i in I for s in SKUS for t in dates) / 1000.0  # ton
co2_trans = (
    quicksum(mode_co2_per_km(m_) * dist_ik[(i,k)] * N_ik[i,k,t,m_] for (i,k) in IK for t in dates for m_ in modes) +
    quicksum(mode_co2_per_km(m_) * dist_kj[(k,j)] * N_kj[k,j,t,m_] for (k,j) in KJ for t in dates for m_ in modes)
) / 1000.0

# 천장(ceil) 톤 반영: 정수톤 >= 실톤, 실톤>0이면 최소 1톤 비용 적용(정밀 선형화는 z 바이너리 필요)
Ton_int = m.addVar(vtype=GRB.INTEGER, lb=0, name='Ton_int')
m.addConstr(Ton_int >= co2_prod + co2_trans)
# 실톤>0 ⇒ Ton_int >=1
z_co2 = m.addVar(vtype=GRB.BINARY, name='z_co2_pos')
m.addConstr((co2_prod + co2_trans) <= BIGM * z_co2)
m.addConstr(Ton_int >= z_co2)  # >0이면 최소 1톤
C_CO2 = 200.0
cost_co2 = C_CO2 * Ton_int

# 건설비 (USD, 착공일 다음 월요일 결제 상세는 후속): 여기서는 선택 시 즉시 지출
init_cost_usd = dict(zip(init_cost['site_id'], init_cost['init_cost_usd']))
cost_build = quicksum(init_cost_usd.get(i,0.0) * x_fac[i] for i in I) + \
             quicksum(init_cost_usd.get(k,0.0) * x_wh[k]  for k in K)

# 목적함수
m.setObjective(cost_build + cost_prod + cost_labor + cost_trans_ik + cost_trans_kj +
               cost_inv + cost_short + cost_change + cost_co2, GRB.MINIMIZE)

# =========================================================
# 9) 파라미터 & 풀이
# =========================================================
m.Params.OutputFlag = 1
m.Params.MIPGap     = 0.02
m.Params.TimeLimit  = 3600  # 1 hour

m.optimize()

# =========================================================
# 10) 결과 내보내기 (예시)
# =========================================================
if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
    print("Objective:", m.ObjVal)

    # 생산 계획
    prod_rows = []
    for i in I:
        for s in SKUS:
            for t in dates:
                v = P[i,s,t].X
                if v > 0:
                    prod_rows.append([t.strftime('%Y-%m-%d'), 'factory', s, int(v), int(H_ot[i,t].X), 0, i, None, None])
    prod_df = pd.DataFrame(prod_rows, columns=['date','factory/warehouse','sku','production_qty','ot_qty','ship_qty','from','to','mode'])

    # 출하 계획 (ik)
    ship_rows = []
    for (i,k) in IK:
        for s in SKUS:
            for t in dates:
                for m_ in modes:
                    qty = X[i,k,s,t,m_].X
                    if qty > 0:
                        ship_rows.append([t.strftime('%Y-%m-%d'),'factory',s,0,0,int(qty), i, k, m_])
    # 출하 계획 (kj)
    for (k,j) in KJ:
        for s in SKUS:
            for t in dates:
                for m_ in modes:
                    qty = Y[k,j,s,t,m_].X
                    if qty > 0:
                        ship_rows.append([t.strftime('%Y-%m-%d'),'warehouse',s,0,0,int(qty), k, j, m_])

    ship_df = pd.DataFrame(ship_rows, columns=['date','factory/warehouse','sku','production_qty','ot_qty','ship_qty','from','to','mode'])

    # 저장
    prod_df.to_csv('plan_production.csv', index=False)
    ship_df.to_csv('plan_shipping.csv', index=False)