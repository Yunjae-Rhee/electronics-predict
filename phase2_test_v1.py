# ============================================================================
#  FAST SCM Solver (weekly aggregation + DB) — speed-optimized
#  - LP relaxation of integer vars
#  - Pruned arcs (nearest neighbors)
#  - No mode-selection binaries or change penalties
#  - Weekly FR with slack penalty (ramp-up aware)
# ============================================================================
from __future__ import annotations
import math, gc, os, sqlite3, psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, DefaultDict

import pandas as pd, numpy as np
from collections import defaultdict
from gurobipy import Model, GRB, quicksum

# ======================= CONFIG =======================
DATA_DIR   = Path("data")
CHUNK_DAYS = 31
DATE_START = datetime(2018, 1, 1)
DATE_END   = datetime(2024,12,31)
DB_NAME    = "plan_submission_template.db"
TABLE_NAME = "plan_submission_template"

# Fill-rate (soft)
FR_TARGET   = 0.95
FR_PENALTY  = 1e6
RAMP_UP_DAYS_MARGIN = 7

# Arc pruning (speed lever)
KJ_NEAREST = 3   # cities served by nearest K warehouses
IK_NEAREST = 2   # each factory ships to nearest K warehouses

# Base costs
TRUCK_BASE_COST_PER_KM = 12.0
TRUCK_BASE_CO2_PER_KM  = 0.40
QCONT = 4000.0

# ======================================================
PROC = psutil.Process(os.getpid())
log  = lambda tag: print(f"[{tag}] RSS = {PROC.memory_info().rss/1024**2:,.1f} MB")

def date_range(d0, d1):
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)

def week_monday(dt):
    return (dt - timedelta(days=dt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)

def week_of(dt): return (dt.date() - datetime(2018,1,1).date()).days // 7
def dow(dt): return dt.weekday() + 1
def block_of(dt): return week_of(dt) // 4
def next_monday_00(dt):
    d = (7 - dt.weekday()) % 7
    d = 7 if d == 0 else d
    return (dt + timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)

def haversine_km(lat1, lon1, lat2, lon2, R=6371.0):
    p1, l1, p2, l2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((p2 - p1)/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin((l2 - l1)/2.0)**2
    return float(2*R*np.arcsin(np.sqrt(a)))

def ceil_pos(x): return 0 if x<=0 else int(math.ceil(x))

def fill_forward_by_biz(df, key, val_col, biz_col='is_biz_day'):
    df = df.sort_values(['date'] + key).copy()
    out = []
    for keys, g in df.groupby(key):
        g = g.sort_values('date')
        last = None; buf=[]
        for _, r in g.iterrows():
            if r[biz_col] == 1 and pd.notnull(r[val_col]): last = r[val_col]
            buf.append(last)
        g[val_col + '_ff'] = buf; out.append(g)
    return pd.concat(out, ignore_index=True)

def norm_city(name): return str(name).replace(' ', '_')

def truck_days_from_distance(dkm):
    d = max(0.0, float(dkm))
    if d <= 500.0:   return 2
    elif d <= 1000.: return 3
    elif d <= 2000.: return 5
    else:            return 8

def to_week_idx(week_col):
    if pd.api.types.is_integer_dtype(week_col): return week_col.astype(int)
    wk_dt = pd.to_datetime(week_col, errors='coerce')
    if wk_dt.isna().any(): raise ValueError("[week 파싱 실패]")
    return wk_dt.apply(week_of).astype(int)

# =================== LOAD DATA ========================
print("Loading master tables …")
dates_all  = list(date_range(DATE_START, DATE_END))
weeks_all  = sorted(set(week_of(t) for t in dates_all))

cal = pd.read_csv(f'{DATA_DIR}/calendar.csv');        cal['date'] = pd.to_datetime(cal['date'])
hol = pd.read_csv(f'{DATA_DIR}/holiday_lookup.csv');  hol['date'] = pd.to_datetime(hol['date']); hol['is_holiday']=1
wx  = pd.read_csv(f'{DATA_DIR}/weather.csv');         wx['date']  = pd.to_datetime(wx['date'])
for col in ['rain_mm','snow_cm','wind_mps','cloud_pct']:
    if col not in wx.columns: wx[col]=0.0
oil = pd.read_csv(f'{DATA_DIR}/oil_price.csv');       oil['date'] = pd.to_datetime(oil['date'])
fx  = pd.read_csv(f'{DATA_DIR}/currency.csv');        fx.rename(columns={'Date':'date'}, inplace=True); fx['date']=pd.to_datetime(fx['date'])

cci     = pd.read_csv(f'{DATA_DIR}/consumer_confidence.csv')
lab_pol = pd.read_csv(f'{DATA_DIR}/labour_policy.csv')

sites     = pd.read_csv(f'{DATA_DIR}/site_candidates.csv')
init_cost = pd.read_csv(f'{DATA_DIR}/site_init_cost.csv')
mode_meta = pd.read_csv(f'{DATA_DIR}/transport_mode_meta.csv')

sku_meta = pd.read_csv(f'{DATA_DIR}/sku_meta.csv'); sku_meta['launch_date']=pd.to_datetime(sku_meta['launch_date'])
carbon_prod = pd.read_csv(f'{DATA_DIR}/carbon_factor_prod.csv')

fac_cap = pd.read_csv(f'{DATA_DIR}/factory_capacity.csv')
mfail   = pd.read_csv(f'{DATA_DIR}/machine_failure_log.csv'); mfail['start_date']=pd.to_datetime(mfail['start_date']); mfail['end_date']=pd.to_datetime(mfail['end_date'])
lab_req = pd.read_csv(f'{DATA_DIR}/labour_requirement.csv')

price_promo = pd.read_csv(f'{DATA_DIR}/price_promo_train.csv')
mkt         = pd.read_csv(f'{DATA_DIR}/marketing_spend.csv')
prod_cost   = pd.read_csv(f'{DATA_DIR}/prod_cost_excl_labour.csv')
inv_cost    = pd.read_csv(f'{DATA_DIR}/inv_cost.csv')
short_cost  = pd.read_csv(f'{DATA_DIR}/short_cost.csv')

con = sqlite3.connect(f'{DATA_DIR}/demand_train.db')
demand_df = pd.read_sql_query("SELECT date, sku, city, demand FROM demand_train", con)
con.close()
demand_df['date']=pd.to_datetime(demand_df['date']); demand_df['city']=demand_df['city'].map(norm_city)

forecast_df = pd.read_csv(f'{DATA_DIR}/forecast_submission_template.csv')
forecast_df['date']=pd.to_datetime(forecast_df['date']); forecast_df['city']=forecast_df['city'].map(norm_city)

hist_mask = (demand_df['date']>=DATE_START) & (demand_df['date']<=min(DATE_END, pd.to_datetime('2022-12-31')))
d_hist = demand_df.loc[hist_mask, ['date','sku','city','demand']].copy()

future_mask = (forecast_df['date']>=pd.to_datetime('2023-01-01')) & (forecast_df['date']<=DATE_END)
d_fut = forecast_df.loc[future_mask, ['date','sku','city','mean']].copy()
d_fut['demand']=d_fut['mean'].round().astype(int); d_fut.drop(columns=['mean'], inplace=True)

all_dem = (pd.concat([d_hist,d_fut], ignore_index=True)
           .sort_values('date')
           .drop_duplicates(subset=['date','sku','city'], keep='last'))

SKUS = sorted(sku_meta['sku'].unique().tolist())
life_days = dict(zip(sku_meta['sku'], sku_meta['life_days']))

sites['city']=sites['city'].map(norm_city)
fac_df = sites[sites['site_type']=='factory'].copy()
wh_df  = sites[sites['site_type']=='warehouse'].copy()

I = fac_df['site_id'].tolist()
K = wh_df['site_id'].tolist()
J = sorted(all_dem['city'].unique().tolist())

site_country = dict(zip(sites['site_id'], sites['country']))
site_city    = dict(zip(sites['site_id'], sites['city']))
site_lat     = dict(zip(sites['site_id'], sites['lat']))
site_lon     = dict(zip(sites['site_id'], sites['lon']))

LocationP = {
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
city_country = {j: LocationP[j] for j in J if j in LocationP}
missing = [j for j in J if j not in city_country]
if missing: raise ValueError(f"[city→country 매핑 누락] {missing}")

D = {(r.city, r.sku, pd.to_datetime(r.date)): int(r.demand) for _, r in all_dem.iterrows()}

a = dict(zip(lab_req['sku'], lab_req['labour_hours_per_unit']))
fac_cap['week_idx'] = to_week_idx(fac_cap['week'])
fac_cap_agg = (fac_cap.groupby(['factory','week_idx'], as_index=False)
               .agg(reg_capacity=('reg_capacity','sum'),
                    ot_capacity=('ot_capacity','sum')))
reg_cap = {(r.factory, int(r.week_idx)): float(r.reg_capacity) for _, r in fac_cap_agg.iterrows()}
ot_cap  = {(r.factory, int(r.week_idx)): float(r.ot_capacity)  for _, r in fac_cap_agg.iterrows()}

base_cost = {(r.sku, r.factory): r.base_cost_usd for _,r in prod_cost.iterrows()}
delta_prod = {r.factory: r.kg_CO2_per_unit for _,r in carbon_prod.iterrows()}
HOLD  = dict(zip(inv_cost['sku'],  inv_cost['inv_cost_per_day']))
SHORT = dict(zip(short_cost['sku'], short_cost['short_cost_per_unit']))

modes     = mode_meta['mode'].tolist()
beta_cost = dict(zip(mode_meta['mode'], mode_meta['cost_per_km_factor']))
gamma_co2 = dict(zip(mode_meta['mode'], mode_meta['co2_per_km_factor']))
alpha_lead= dict(zip(mode_meta['mode'], mode_meta['leadtime_factor']))

# ---------- distances & leads ----------
IK_all = [(i,k) for i in I for k in K]
KJ_all = [(k,j) for k in K for j in J]

dist_ik, dist_kj = {}, {}
L_ik = {m:{} for m in modes}
L_kj = {m:{} for m in modes}

for (i,k) in IK_all:
    d = haversine_km(site_lat[i], site_lon[i], site_lat[k], site_lon[k])
    dist_ik[(i,k)] = d
    base_days = truck_days_from_distance(d)
    for m in modes:
        L_ik[m][(i,k)] = ceil_pos(alpha_lead[m] * base_days)

# city coords (avg of site city if needed)
fac_city_ll = fac_df.groupby('city')[['lat','lon']].mean().reset_index().set_index('city')
city_lat, city_lon = {}, {}
for j in J:
    if j in fac_city_ll.index:
        city_lat[j]=float(fac_city_ll.loc[j,'lat']); city_lon[j]=float(fac_city_ll.loc[j,'lon'])
    else:
        wh_city_ll = wh_df.groupby('city')[['lat','lon']].mean().reset_index().set_index('city')
        if j in wh_city_ll.index:
            city_lat[j]=float(wh_city_ll.loc[j,'lat']); city_lon[j]=float(wh_city_ll.loc[j,'lon'])
        else:
            raise ValueError(f"[도시 좌표 없음] {j}")

for (k,j) in KJ_all:
    d = haversine_km(site_lat[k], site_lon[k], city_lat[j], city_lon[j])
    dist_kj[(k,j)] = d
for m in modes:
    for (k,j) in KJ_all:
        base_days = truck_days_from_distance(dist_kj[(k,j)])
        L_kj[m][(k,j)] = ceil_pos(alpha_lead[m] * base_days)

EU_ZONE = {'DEU','FRA'}   # simplify
BORDER_COST = 4000.0
def allowed_mode_between(cu, cv, m):
    eu_pair = (cu in EU_ZONE) and (cv in EU_ZONE)
    if cu==cv: return m=='TRUCK'
    if eu_pair: return m in ['TRUCK','SHIP','AIR']
    return m in ['SHIP','AIR']
ALLOWED_IK = {(i,k,m): int(allowed_mode_between(site_country[i], site_country[k], m)) for (i,k) in IK_all for m in modes}
ALLOWED_KJ = {(k,j,m): int(allowed_mode_between(site_country[k], city_country[j], m)) for (k,j) in KJ_all for m in modes}

BORDER_IK = {(i,k): (0.0 if site_country[i]==site_country[k] or ((site_country[i] in EU_ZONE) and (site_country[k] in EU_ZONE)) else BORDER_COST) for (i,k) in IK_all}
BORDER_KJ = {(k,j): (0.0 if site_country[k]==city_country[j] or ((site_country[k] in EU_ZONE) and (city_country[j] in EU_ZONE)) else BORDER_COST) for (k,j) in KJ_all}

# local delivery zeroes
for (k,j) in KJ_all:
    if site_city[k]==j:
        dist_kj[(k,j)]=0.0
        for m in modes: L_kj[m][(k,j)]=0

# ---- ramp-up horizon ----
MAX_L_IK = max(L_ik[m][e] for m in modes for e in L_ik[m]) if L_ik else 0
MAX_L_KJ = max(L_kj[m][e] for m in modes for e in L_kj[m]) if L_kj else 0
RAMP_UP_DAYS = MAX_L_IK + MAX_L_KJ + RAMP_UP_DAYS_MARGIN

# ---------- FX / oil / holidays / weather ----------
all_days = pd.DataFrame({'date': pd.date_range(DATE_START, DATE_END, freq='D')})
countries = sorted(pd.unique(pd.concat([sites['country'], cal['country'], hol['country'], wx['country']], ignore_index=True)))

biz_tbl=[]
for c in countries:
    g=all_days.copy(); g['country']=c; g['weekday']=g['date'].dt.weekday; g['is_biz_day']=((g['weekday']<5)*1).astype(int)
    h=hol[hol['country']==c][['date','is_holiday']].drop_duplicates()
    g=g.merge(h,on='date',how='left'); g.loc[g['is_holiday']==1,'is_biz_day']=0
    g.drop(columns=['weekday','is_holiday'], inplace=True); biz_tbl.append(g)
biz = pd.concat(biz_tbl, ignore_index=True)

country_ccy = dict(zip(lab_pol['country'], lab_pol['currency']))
fx_long = fx.melt(id_vars=['date'], var_name='pair', value_name='usd_to_local')
fx_long['ccy']=fx_long['pair'].replace('=X','',regex=True).str.replace('=X','',regex=False)
fx_long['usd_per_local']=1.0/fx_long['usd_to_local']
biz_ccy=biz.copy(); biz_ccy['ccy']=biz_ccy['country'].map(country_ccy)
fx_biz=biz_ccy.merge(fx_long[['date','ccy','usd_per_local']], on=['date','ccy'], how='left')
def fill_forward_by_biz_simple(df):
    df=df.sort_values(['country','date'])
    df['usd_per_local']=df.groupby('country')['usd_per_local'].ffill()
    return df
fx_biz = fill_forward_by_biz_simple(fx_biz)
fx_biz.loc[fx_biz['ccy']=='USD', 'usd_per_local']=1.0
FX_USD_PER_LOCAL = {(r.country, r.date): float(r.usd_per_local) for _, r in fx_biz.iterrows()}

oil_all = all_days.merge(oil[['date','brent_usd']], on='date', how='left')
oil_all['brent_usd']=oil_all['brent_usd'].ffill()
OIL = {r.date: float(r.brent_usd) for _,r in oil_all.iterrows()}

HOLIDAY = {}
for c in countries:
    hset=set(hol.loc[hol['country']==c,'date'].dt.normalize())
    for dt in all_days['date']:
        HOLIDAY[(c,dt)] = 1 if (dt.normalize() in hset or dt.weekday()>=5) else 0

WX = {(r.country, r.date): (float(r.get('rain_mm',0)), float(r.get('snow_cm',0)),
                            float(r.get('wind_mps',0)), float(r.get('cloud_pct',0)))
      for _,r in wx.iterrows()}
def is_bad_weather(country, dt):
    rain,snow,wind,cloud = WX.get((country, dt), (0,0,0,0))
    return (rain >= 45.7) or (snow >= 3.85) or (wind >= 13.46) or (cloud >= 100.0)

oil_mondays=[t for t in dates_all if t.weekday()==0]
OIL_UP_WEEK=set()
for idx in range(1,len(oil_mondays)):
    t_prev, t_cur = oil_mondays[idx-1], oil_mondays[idx]
    p_prev, p_cur = OIL[t_prev], OIL[t_cur]
    if p_prev>0 and (p_cur-p_prev)/p_prev>=0.05:
        w=week_of(t_cur)
        for tt in dates_all:
            if week_of(tt)==w: OIL_UP_WEEK.add(tt)

def cost_multiplier_depart(country, dt):
    k_wx = 3.0 if is_bad_weather(country, dt) else 1.0
    k_oil= 2.0 if (dt in OIL_UP_WEEK) else 1.0
    return k_wx * k_oil

# ====== PRUNE ARCS ======
# For each city j: keep nearest K warehouses (always include same-city)
KJ = []
for j in J:
    dlist = sorted([(k, dist_kj[(k,j)]) for k in K], key=lambda x: x[1])
    keep = set(k for k,_ in dlist[:KJ_NEAREST])
    keep |= set(k for k in K if site_city[k]==j)
    for k in keep: KJ.append((k,j))
KJ = sorted(set(KJ))

# For each factory i: keep nearest IK_NEAREST warehouses
IK = []
for i in I:
    dlist = sorted([(k, dist_ik[(i,k)]) for k in K], key=lambda x: x[1])
    for k,_ in dlist[:IK_NEAREST]:
        IK.append((i,k))
IK = sorted(set(IK))

# ======================================================
def build_model_chunk(t0, t1, inv_init, carry_arr_k_s_t, carry_arr_j_s_t):
    dates_chunk = [d for d in dates_all if t0 <= d <= t1]
    weeks_chunk = sorted(set(week_of(t) for t in dates_chunk))

    m = Model(f"FAST_SCM_{t0.date()}_{t1.date()}")
    BIGM = 1e9

    # ---- VARIABLES (LP relaxed) ----
    b_fac = m.addVars(I, dates_chunk, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='b_fac')
    b_wh  = m.addVars(K, dates_chunk, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='b_wh')
    live_fac = m.addVars(I, dates_chunk, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='live_fac')
    live_wh  = m.addVars(K, dates_chunk, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='live_wh')
    pay_fac  = m.addVars(I, dates_chunk, lb=0.0, name='pay_fac')
    pay_wh   = m.addVars(K, dates_chunk, lb=0.0, name='pay_wh')
    x_fac_any= m.addVars(I, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x_fac_any')
    x_wh_any = m.addVars(K, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x_wh_any')

    P     = m.addVars(I, SKUS, dates_chunk, lb=0.0, name='P')
    H_reg = m.addVars(I, dates_chunk, lb=0.0, name='H_reg')
    H_ot  = m.addVars(I, dates_chunk, lb=0.0, name='H_ot')

    X = m.addVars(IK, SKUS, dates_chunk, modes, lb=0.0, name='X_ik')
    Y = m.addVars(KJ, SKUS, dates_chunk, modes, lb=0.0, name='Y_kj')

    N_ik = m.addVars(IK, dates_chunk, modes, lb=0.0, name='Ncont_ik')
    N_kj = m.addVars(KJ, dates_chunk, modes, lb=0.0, name='Ncont_kj')

    Ivar = m.addVars(K, SKUS, dates_chunk, lb=0.0, name='I')
    Wvar = m.addVars(K, SKUS, dates_chunk, lb=0.0, name='W')
    Svar = m.addVars(J, SKUS, dates_chunk, lb=0.0, name='S')
    Uvar = m.addVars(J, SKUS, dates_chunk, lb=0.0, name='U')

    FR_slack = m.addVars(J, SKUS, weeks_chunk, lb=0.0, name='FR_slack')

    # ---- CONSTRAINTS ----
    for i in I:
        m.addConstr(quicksum(b_fac[i,t] for t in dates_chunk) <= 1, name=f'build_once_fac_{i}')
        m.addConstr(x_fac_any[i] == quicksum(b_fac[i,t] for t in dates_chunk))
    for k in K:
        m.addConstr(quicksum(b_wh[k,t] for t in dates_chunk) <= 1, name=f'build_once_wh_{k}')
        m.addConstr(x_wh_any[k] == quicksum(b_wh[k,t] for t in dates_chunk))

    m.addConstr(x_fac_any.sum() <= 5,  'fac_limit')
    m.addConstr(x_wh_any.sum()  <= 20, 'wh_limit')
    for city, g in fac_df.groupby('city'):
        m.addConstr(quicksum(x_fac_any[i] for i in g['site_id']) <= 1, f'one_fac_per_city_{city}')
    for city, g in wh_df.groupby('city'):
        m.addConstr(quicksum(x_wh_any[k] for k in g['site_id']) <= 1, f'one_wh_per_city_{city}')

    nextMon = {t: next_monday_00(t) for t in dates_chunk}
    init_cost_usd = dict(zip(init_cost['site_id'], init_cost['init_cost_usd']))

    for i in I:
        for t in dates_chunk:
            lhs = quicksum(b_fac[i, tau] for tau in dates_chunk if nextMon[tau] <= t)
            m.addConstr(live_fac[i, t] == lhs)
            due_sum = quicksum(b_fac[i, tau] for tau in dates_chunk if nextMon[tau] == t)
            m.addConstr(pay_fac[i, t] == init_cost_usd.get(i, 0.0) * due_sum)

    for k in K:
        for t in dates_chunk:
            lhs = quicksum(b_wh[k, tau] for tau in dates_chunk if tau <= t)
            m.addConstr(live_wh[k, t] == lhs)
            due_sum = quicksum(b_wh[k, tau] for tau in dates_chunk if nextMon[tau] == t)
            m.addConstr(pay_wh[k, t] == init_cost_usd.get(k, 0.0) * due_sum)

    for i in I:
        for s in SKUS:
            for t in dates_chunk:
                m.addConstr(P[i,s,t] <= BIGM * live_fac[i, t])

    for i in I:
        for t in dates_chunk:
            m.addConstr(quicksum(a.get(s,0.0) * P[i,s,t] for s in SKUS) <= 8.0*H_reg[i,t] + H_ot[i,t])

    for i in I:
        ci = site_country[i]
        for w in weeks_chunk:
            week_days = [t for t in dates_chunk if week_of(t)==w]
            if (i, w) in reg_cap:
                m.addConstr(quicksum(H_reg[i, t] for t in week_days) <= reg_cap[(i, w)])
            if (i, w) in ot_cap:
                m.addConstr(quicksum(H_ot[i, t] for t in week_days) <= ot_cap[(i, w)])
            if week_days:
                y = max(t.year for t in week_days)
                row = lab_pol[(lab_pol['country']==ci) & (lab_pol['year']==y)]
                if len(row)>0:
                    Hlaw = float(row['max_hours_week'].iloc[0])
                    m.addConstr(quicksum(H_reg[i,t]+H_ot[i,t] for t in week_days) <= Hlaw)

    for i in I:
        ci = site_country[i]
        for t in dates_chunk:
            is_hol = HOLIDAY.get((ci, t), 0)
            m.addConstr(H_ot[i,t]  <= BIGM * live_fac[i,t])
            if is_hol == 1: m.addConstr(H_reg[i,t] == 0)
            else:           m.addConstr(H_reg[i,t] <= 8.0 * live_fac[i,t])

    # Factory failure
    for _, r in mfail.iterrows():
        i=r['factory']
        if i not in I: continue
        for t in dates_chunk:
            if r['start_date'] <= t <= r['end_date']:
                for s in SKUS: m.addConstr(P[i,s,t]==0)
                for (i2,k) in IK:
                    if i2==i:
                        for s in SKUS:
                            for m_ in modes: m.addConstr(X[i,k,s,t,m_]==0)
                m.addConstr(H_reg[i,t]==0); m.addConstr(H_ot[i,t]==0)

    # Flow capacity by containers (continuous)
    for (i,k) in IK:
        for t in dates_chunk:
            for m_ in modes:
                if ALLOWED_IK[(i,k,m_)]==0:
                    m.addConstr(N_ik[i,k,t,m_]==0); 
                    for s in SKUS: m.addConstr(X[i,k,s,t,m_]==0)
                else:
                    m.addConstr(quicksum(X[i,k,s,t,m_] for s in SKUS) <= QCONT * N_ik[i,k,t,m_])
                    m.addConstr(N_ik[i,k,t,m_] <= BIGM * live_fac[i,t])
                    m.addConstr(N_ik[i,k,t,m_] <= BIGM * live_wh[k,t])

    for (k,j) in KJ:
        for t in dates_chunk:
            for m_ in modes:
                if ALLOWED_KJ[(k,j,m_)]==0:
                    m.addConstr(N_kj[k,j,t,m_]==0)
                    for s in SKUS: m.addConstr(Y[k,j,s,t,m_]==0)
                else:
                    m.addConstr(quicksum(Y[k,j,s,t,m_] for s in SKUS) <= QCONT * N_kj[k,j,t,m_])
                    m.addConstr(N_kj[k,j,t,m_] <= BIGM * live_wh[k,t])

    # Inventory flow with inbound arrivals (IK) and outbound (KJ)
    for k in K:
        for s in SKUS:
            for idx, t in enumerate(dates_chunk):
                inbound_parts=[]
                for i in I:
                    for m_ in modes:
                        L = L_ik[m_][(i,k)]
                        t_dep = t - timedelta(days=L)
                        if (i,k) in IK and t_dep in dates_chunk:
                            inbound_parts.append(X[i,k,s,t_dep,m_])
                inbound_expr = quicksum(inbound_parts) + carry_arr_k_s_t.get((k,s,t),0)

                outbound_expr = quicksum(Y[k,j,s,t,m_] for (k2,j) in KJ if k2==k for m_ in modes)
                I_prev = inv_init.get((k,s),0) if idx==0 else Ivar[k,s,dates_chunk[idx-1]]
                m.addConstr(Ivar[k,s,t] == I_prev + inbound_expr - outbound_expr - Wvar[k,s,t])

    # Demand satisfaction by arrivals (KJ)
    for j in J:
        for s in SKUS:
            for t in dates_chunk:
                Djst = int(D.get((j,s,t),0))
                m.addConstr(Svar[j,s,t] <= Djst)
                m.addConstr(Uvar[j,s,t] == Djst - Svar[j,s,t])

                arr_parts=[]
                for (k2,j2) in KJ:
                    if j2!=j: continue
                    for m_ in modes:
                        L=L_kj[m_][(k2,j)]
                        t_dep = t - timedelta(days=L)
                        if t_dep in dates_chunk:
                            arr_parts.append(Y[k2,j,s,t_dep,m_])
                arrivals = quicksum(arr_parts) + carry_arr_j_s_t.get((j,s,t),0)
                m.addConstr(Svar[j,s,t] <= arrivals)

    # Shelf life (window) – chunk-local approximation
    for k in K:
        for s in SKUS:
            Ls = int(life_days.get(s, 10**9))
            for idx, t in enumerate(dates_chunk):
                t0 = dates_chunk[max(0, idx-Ls+1)]
                win = [tt for tt in dates_chunk if t0<=tt<=t]
                in_win=[]
                for tt in win:
                    in_win.append(carry_arr_k_s_t.get((k,s,tt),0))
                    for i in I:
                        for m_ in modes:
                            L=L_ik[m_][(i,k)]; t_dep=tt - timedelta(days=L)
                            if (i,k) in IK and t_dep in dates_chunk:
                                in_win.append(X[i,k,s,t_dep,m_])
                out_win=[]
                for tt in win:
                    for (k2,j) in KJ:
                        if k2==k:
                            for m_ in modes: out_win.append(Y[k,j,s,tt,m_])
                m.addConstr(quicksum(out_win) <= quicksum(in_win))

    # ====== GATED: "기한초과 입고 ≤ 누적 폐기" (t - Ls >= t0일 때만 활성화) ======
    threshold_day = t - timedelta(days=Ls)
    # build 함수 인자로 t0가 있다면 그걸 쓰고, 없다면 청크 시작일을 t0로 사용
    chunk_start = t0 if 't0' in locals() or 't0' in globals() else dates_chunk[0]

    if threshold_day >= chunk_start:
        old_days = [tt for tt in dates_chunk if tt <= threshold_day]
        if old_days:
            old_in = []
            for tt in old_days:
                # (a) 이전 청크에서 tt에 '도착'한 carry-in
                old_in.append(carry_arr_k_s_t.get((k, s, tt), 0))
                # (b) 이번 청크 내 출발→tt 도착분
                for i in I:
                    for m_ in modes:
                        L = L_ik[m_][(i, k)]
                        t_dep = tt - timedelta(days=L)
                        if (i, k) in IK and t_dep in dates_chunk:
                            old_in.append(X[i, k, s, t_dep, m_])

            m.addConstr(
                quicksum(old_in) <= quicksum(Wvar[k, s, tt] for tt in dates_chunk if tt <= t),
                name=f'expire_dispose_gated_{k}_{s}_{t.date()}'
            )
    # =======================================================

    # Weekly FR with slack (ramp-up aware)
    for j in J:
        for s in SKUS:
            for w in weeks_chunk:
                wdays = [t for t in dates_chunk if week_of(t)==w]
                D_week = quicksum(int(D.get((j,s,t),0)) for t in wdays)
                S_week = quicksum(Svar[j,s,t] for t in wdays)
                last_day = max(wdays) if wdays else None
                m.addConstr(S_week + FR_slack[j,s,w] >= FR_TARGET * D_week)

    # ---- COSTS & OBJECTIVE ----
    def wage_usd(country, dt, is_ot=False):
        y=int(dt.year)
        row = lab_pol[(lab_pol['country']==country) & (lab_pol['year']==y)]
        if row.empty: return 0.0
        reg_local = float(row['regular_wage_local'].iloc[0])
        ot_mult   = float(row['ot_mult'].iloc[0])
        fx_usd = FX_USD_PER_LOCAL.get((country, dt), 1.0)
        base_usd = reg_local * fx_usd
        return base_usd * (ot_mult if is_ot else 1.0)

    cost_prod = quicksum(base_cost.get((s,i),0.0)*P[i,s,t] for i in I for s in SKUS for t in dates_chunk)
    cost_labor= quicksum(wage_usd(site_country[i], t, False)*H_reg[i,t] + wage_usd(site_country[i], t, True)*H_ot[i,t]
                          for i in I for t in dates_chunk)

    def mode_cost_per_km(m): return TRUCK_BASE_COST_PER_KM * beta_cost[m]
    def mode_co2_per_km(m):  return TRUCK_BASE_CO2_PER_KM  * gamma_co2[m]

    cost_trans_ik = quicksum(
        cost_multiplier_depart(site_country[i], t) * (mode_cost_per_km(m_)*dist_ik[(i,k)]*N_ik[i,k,t,m_] + BORDER_IK[(i,k)]*N_ik[i,k,t,m_])
        for (i,k) in IK for t in dates_chunk for m_ in modes)

    cost_trans_kj = quicksum(
        cost_multiplier_depart(site_country[k], t) * (mode_cost_per_km(m_)*dist_kj[(k,j)]*N_kj[k,j,t,m_] + BORDER_KJ[(k,j)]*N_kj[k,j,t,m_])
        for (k,j) in KJ for t in dates_chunk for m_ in modes)

    cost_inv   = quicksum(HOLD.get(s,0.0)*Ivar[k,s,t] for k in K for s in SKUS for t in dates_chunk)
    cost_short = quicksum(SHORT.get(s,0.0)*Uvar[j,s,t] for j in J for s in SKUS for t in dates_chunk)

    co2_prod = quicksum(delta_prod.get(i,0.0)*P[i,s,t] for i in I for s in SKUS for t in dates_chunk)/1000.0
    co2_trans= (quicksum(mode_co2_per_km(m_)*dist_ik[(i,k)]*N_ik[i,k,t,m_] for (i,k) in IK for t in dates_chunk for m_ in modes) +
                quicksum(mode_co2_per_km(m_)*dist_kj[(k,j)]*N_kj[k,j,t,m_] for (k,j) in KJ for t in dates_chunk for m_ in modes))/1000.0
    # simple linearized CO2 cost (no ceil/min1)
    cost_co2 = 200.0*(co2_prod + co2_trans)

    cost_build = quicksum(pay_fac[i,t] for i in I for t in dates_chunk) + quicksum(pay_wh[k,t] for k in K for t in dates_chunk)
    cost_frpen = FR_PENALTY*quicksum(FR_slack[j,s,w] for j in J for s in SKUS for w in weeks_chunk)

    m.setObjective(cost_prod + cost_labor + cost_trans_ik + cost_trans_kj + cost_inv + cost_short + cost_co2 + cost_build + cost_frpen, GRB.MINIMIZE)

    # ---- Gurobi speed knobs ----
    m.Params.OutputFlag = 0
    m.Params.Presolve   = 2
    m.Params.Heuristics = 0.5
    m.Params.MIPFocus   = 1
    m.Params.TimeLimit  = 900   # 15분/청크 (필요시 더 줄여도 됨)
    # m.Params.Threads  = 0  # 자동

    return m, {"I_end":Ivar, "P":P, "H_reg":H_reg, "H_ot":H_ot, "X":X, "Y":Y}

# =================== DRIVER ===========================
def run_chunked_pipeline():
    inv_carry: Dict[Tuple[str, str], int] = {}
    carry_arr_k_s_t: Dict[Tuple[str,str,datetime], int] = {}
    carry_arr_j_s_t: Dict[Tuple[str,str,datetime], int] = {}

    t_curr = DATE_START
    total_obj = 0.0

    WeekKey = Tuple[str, str, str]
    PLAN: DefaultDict[WeekKey, List[int]] = defaultdict(lambda: [0,0,0])
    SHIP_ROWS: List[Tuple[str, str, str, str, str, int, str]] = []

    while t_curr <= DATE_END:
        t_next = min(t_curr + timedelta(days=CHUNK_DAYS - 1), DATE_END)
        print(f"\n===  Solving {t_curr.date()} – {t_next.date()}  ===")
        log("build")
        m, ref = build_model_chunk(t_curr, t_next, inv_carry, carry_arr_k_s_t, carry_arr_j_s_t)
        log("opt start"); m.optimize(); log("opt done")

        if m.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            try:
                m.computeIIS(); m.write(f"iis_{t_curr.date()}_{t_next.date()}.ilp")
                m.write(f"model_{t_curr.date()}_{t_next.date()}.lp")
            except: pass
            raise RuntimeError("MILP failed to converge")

        total_obj += m.ObjVal

        P, H_reg, H_ot = ref["P"], ref["H_reg"], ref["H_ot"]
        for (i,s,t) in P.keys():
            if not (t_curr<=t<=t_next): continue
            units = int(round(P[i,s,t].X))
            if units==0: continue
            hrs_tot = H_reg[i,t].X + H_ot[i,t].X
            reg_u = units if hrs_tot==0 else int(round(units * (H_reg[i,t].X / max(1e-9,hrs_tot))))
            ot_u  = units - reg_u
            wk = week_monday(t).strftime("%Y-%m-%d")
            PLAN[(wk,i,s)][0]+=reg_u; PLAN[(wk,i,s)][1]+=ot_u

        X, Y = ref["X"], ref["Y"]
        for (i,k,s,t,m_) in X.keys():
            if not (t_curr<=t<=t_next): continue
            qty = int(round(X[i,k,s,t,m_].X))
            if qty==0: continue
            wk = week_monday(t).strftime("%Y-%m-%d")
            PLAN[(wk,i,s)][2]+=qty
            SHIP_ROWS.append((wk, s, site_city[i], site_city[k], m_, qty, i))
            L=L_ik[m_][(i,k)]; t_arr=t+timedelta(days=L)
            if t_arr>t_next: carry_arr_k_s_t[(k,s,t_arr)] = carry_arr_k_s_t.get((k,s,t_arr),0)+qty

        for (k,j,s,t,m_) in Y.keys():
            if not (t_curr<=t<=t_next): continue
            qty = int(round(Y[k,j,s,t,m_].X))
            if qty==0: continue
            wk = week_monday(t).strftime("%Y-%m-%d")
            SHIP_ROWS.append((wk, s, site_city[k], j, m_, qty, None))
            L=L_kj[m_][(k,j)]; t_arr=t+timedelta(days=L)
            if t_arr>t_next: carry_arr_j_s_t[(j,s,t_arr)] = carry_arr_j_s_t.get((j,s,t_arr),0)+qty

        inv_carry.clear()
        I_end = ref["I_end"]
        for k in K:
            for s in SKUS:
                inv_carry[(k,s)] = int(round(I_end[k,s,t_next].X))

        del m; gc.collect()
        t_curr = t_next + timedelta(days=1)

    print("\nPIPELINE DONE  →  total cost =", f"${total_obj:,.0f}")

    # ---- build DB ----
    df_prod = pd.DataFrame(
        [(wk, fac, sku, vals[0], vals[1]) for (wk, fac, sku), vals in PLAN.items() if vals[0] or vals[1]],
        columns=["date", "factory", "sku", "production_qty", "ot_qty"]
    )
    df_ship = (pd.DataFrame(SHIP_ROWS, columns=["date","sku","from_city","to_city","mode","ship_qty","factory"])
               .groupby(["date","factory","sku","from_city","to_city","mode"], as_index=False)
               .agg(ship_qty=("ship_qty","sum")))
    df_final = pd.concat([
        df_prod.assign(ship_qty=0, from_city=None, to_city=None, mode=None),
        df_ship.assign(production_qty=0, ot_qty=0)
    ], ignore_index=True).fillna({"production_qty":0, "ot_qty":0, "ship_qty":0})
    df_final = df_final[["date","factory","sku","production_qty","ot_qty","ship_qty","from_city","to_city","mode"]]

    if os.path.exists(DB_NAME): os.remove(DB_NAME)
    with sqlite3.connect(DB_NAME) as conn:
        df_final.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    print(f"✅  결과 DB 작성 완료  →  {DB_NAME} (table: {TABLE_NAME})")

# =================== MAIN ============================
if __name__ == "__main__":
    run_chunked_pipeline()