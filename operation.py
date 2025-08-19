# -*- coding: utf-8 -*-
# ============================================================================
#  STAGE 3 OPERATIONAL SIMULATOR (Exact labor split: Regular vs OT) — FIXED
#  - 2단계(tactical_plan_advanced.json) 기반 일일 시뮬레이션
#  - 버그 수정:
#    (A) open_pipeline 이중가산 제거 (선적 시만 +, 입고 시 -)
#    (B) 주간 용량 캘린더 공장별 ffill (빈 주차도 생산 가능)
#    (C) S(목표재고)까지 발주 (ROP 트리거)
#    (D) 고갈예측 ≤ 리드타임 시 AIR 강제
#  - 비용: 생산(정규/OT/비노동), 운송(국경/날씨/유가), 보관, 품절
#  - 지표: Fill-Rate(주간/전체)
#  - 산출물:
#       plan_submission_template.db (table: plan_submission_template)
#       cost_daily.csv, fr_weekly.csv, fr_overall.csv, shortages_log.csv
# ============================================================================

import pandas as pd
import numpy as np
import json, sqlite3, os
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque

# ---------------- Consts & IO ----------------
DATA_DIR   = Path("data")
DATE_START = datetime(2018, 1, 1)
DATE_END   = datetime(2024, 12, 31)
DB_NAME    = "plan_submission_template.db"
TABLE_NAME = "plan_submission_template"

TRUCK_BASE_COST_PER_KM = 12.0
BORDER_FEE_PER_UNIT    = 4000.0      # 국경 통과 단위당 근사 가산(EU 면제)
EU_ZONE = {'DEU','FRA'}

WEATHER_MULT_BAD = 3.0               # 악천후 할증
OIL_JUMP_PCT     = 0.05
OIL_MULT_JUMP    = 2.0               # 유가 급등 주간 할증

def norm_city(x: str) -> str:
    return str(x).replace(' ', '_')

# ---------------- Helpers ----------------
def truck_days_from_distance(dkm: float) -> int:
    d = max(0.0, float(dkm))
    if d <= 500.0:   return 2
    elif d <= 1000.: return 3
    elif d <= 2000.: return 5
    else:            return 8

def haversine_km(lat1, lon1, lat2, lon2, R=6371.0):
    p1, l1, p2, l2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((p2 - p1)/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin((l1 - l2)/2.0)**2
    return float(2*R*np.arcsin(np.sqrt(a)))

def is_cross_border(cu, cv):
    if cu == cv: return False
    if (cu in EU_ZONE) and (cv in EU_ZONE): return False
    return True

def compute_lead_days(site_info, mode_meta, fac_id, wh_id, default_mode="SHIP"):
    """공장→창고 리드타임(일) 및 거리/모드 산정."""
    lat_f, lon_f = site_info.loc[fac_id, ['lat','lon']]
    lat_w, lon_w = site_info.loc[wh_id, ['lat','lon']]
    c_f = site_info.loc[fac_id, 'country']
    c_w = site_info.loc[wh_id, 'country']
    dist_km   = haversine_km(lat_f, lon_f, lat_w, lon_w)
    truck_d   = truck_days_from_distance(dist_km)
    if c_f == c_w:
        mode = "TRUCK" if default_mode != "AIR" else "AIR"
    else:
        mode = default_mode  # 국제: 기본 SHIP, 긴급은 AIR
    factor = float(mode_meta.loc[mode, 'leadtime_factor'])
    lead   = int(np.ceil(truck_d * factor))
    return max(0, lead), mode, dist_km, c_f, c_w

def is_bad_weather(weather_by_cc, country, dt):
    row = weather_by_cc.get((country, dt.date()))
    if row is None: return False
    rain, snow, wind, cloud = row
    return (rain >= 45.7) or (snow >= 3.85) or (wind >= 13.46) or (cloud >= 100.0)

def oil_jump_week(oil_jumps_set, dt):
    return dt.date() in oil_jumps_set

def monday_of(dt): return (dt - timedelta(days=dt.weekday())).date()

# ---------------- Main ----------------
def main_stage3_simulator():
    print("--- 3단계 운영 시뮬레이션 시작 (정규/OT, 비용, FR) ---")

    # ---------- (1) 계획/데이터 로드 ----------
    print("\n[3-1] 데이터 및 전술 계획 로드 중...")
    try:
        with open("tactical_plan_advanced.json", "r", encoding='utf-8') as f:
            plan = json.load(f)
        sourcing_policy  = plan['sourcing_policy']      # {wh: {primary_source:{factory,cost,lead_time}, secondary_source}}
        inventory_policy = plan['inventory_policy']     # {wh: {sku:{safety_stock,reorder_point}}}
    except FileNotFoundError:
        print("오류: 2단계 결과 파일('tactical_plan_advanced.json')이 필요합니다.")
        return

    # 마스터 데이터
    sites_df   = pd.read_csv(DATA_DIR / 'site_candidates.csv')
    sites_df['city'] = sites_df['city'].map(norm_city)
    site_info  = sites_df.set_index('site_id')
    mode_meta  = pd.read_csv(DATA_DIR / 'transport_mode_meta.csv').set_index('mode')

    # 생산/노동 관련
    fac_cap_df = pd.read_csv(DATA_DIR / 'factory_capacity.csv')
    fac_cap_df['week'] = pd.to_datetime(fac_cap_df['week'])
    labour_req = pd.read_csv(DATA_DIR / 'labour_requirement.csv')     # sku, labour_hours_per_unit
    a_hours = dict(zip(labour_req['sku'], labour_req['labour_hours_per_unit']))

    lab_pol   = pd.read_csv(DATA_DIR / 'labour_policy.csv')          # country,year,regular_wage_local,ot_mult,max_hours_week,currency
    fx_df     = pd.read_csv(DATA_DIR / 'currency.csv')               # Date + "<CCY>=X" 열들(USD/Local)
    fx_df.rename(columns={'Date':'date'}, inplace=True)
    fx_df['date'] = pd.to_datetime(fx_df['date'])
    fx_long = fx_df.melt(id_vars=['date'], var_name='pair', value_name='usd_per_local_inv')
    fx_long['ccy'] = fx_long['pair'].str.replace('=X','', regex=False)
    fx_long['usd_per_local'] = 1.0 / fx_long['usd_per_local_inv']
    ctry_ccy = dict(zip(lab_pol['country'], lab_pol['currency']))

    dates_all = pd.date_range(DATE_START, DATE_END, freq='D')
    fx_tbl = []
    for c in lab_pol['country'].unique():
        ccy = ctry_ccy.get(c, 'USD')
        tmp = pd.DataFrame({'date': dates_all})
        sub = fx_long[fx_long['ccy']==ccy][['date','usd_per_local']].sort_values('date')
        tmp = tmp.merge(sub, on='date', how='left').sort_values('date')
        tmp['usd_per_local'] = tmp['usd_per_local'].ffill()
        if ccy == 'USD': tmp['usd_per_local'] = 1.0
        tmp['country'] = c
        fx_tbl.append(tmp)
    fx_tbl = pd.concat(fx_tbl, ignore_index=True)
    FX = {(r['country'], r['date'].date()): float(r['usd_per_local']) for _, r in fx_tbl.iterrows()}

    # 비노동 생산원가(공장×SKU)
    prod_cost = pd.read_csv(DATA_DIR / 'prod_cost_excl_labour.csv')  # sku,factory,base_cost_usd
    base_cost = {(r['sku'], r['factory']): float(r['base_cost_usd']) for _,r in prod_cost.iterrows()}

    # 비용 테이블
    inv_cost_df   = pd.read_csv(DATA_DIR / 'inv_cost.csv')           # sku, inv_cost_per_day
    short_cost_df = pd.read_csv(DATA_DIR / 'short_cost.csv')         # sku, short_cost_per_unit
    HOLD  = dict(zip(inv_cost_df['sku'],   inv_cost_df['inv_cost_per_day']))
    SHORT = dict(zip(short_cost_df['sku'], short_cost_df['short_cost_per_unit']))

    # 기계고장/휴일
    mfail = pd.read_csv(DATA_DIR / 'machine_failure_log.csv')
    if not mfail.empty:
        mfail['start_date'] = pd.to_datetime(mfail['start_date'])
        mfail['end_date']   = pd.to_datetime(mfail['end_date'])

    # holiday_lookup.csv: country,date,holiday_name → is_holiday=1 생성
    try:
        hol = pd.read_csv(DATA_DIR / 'holiday_lookup.csv')
        hol['date'] = pd.to_datetime(hol['date'])
        hol['is_holiday'] = 1
        hol = hol[['country', 'date', 'is_holiday']]
    except FileNotFoundError:
        hol = pd.DataFrame(columns=['country','date','is_holiday'])
    HOL = {(r['country'], r['date'].date()): int(r['is_holiday']) for _, r in hol.iterrows()}

    # 수요 (과거+미래)
    try:
        con = sqlite3.connect(DATA_DIR / 'demand_train.db')
        d_hist = pd.read_sql_query("SELECT date, sku, city, demand FROM demand_train", con, parse_dates=['date'])
        con.close()
    except Exception:
        d_hist = pd.DataFrame(columns=['date','sku','city','demand'])
    d_hist['city'] = d_hist['city'].map(norm_city)

    d_fut = pd.read_csv(DATA_DIR / 'forecast_submission_template.csv', parse_dates=['date'])
    d_fut = d_fut.rename(columns={'mean':'demand'})
    d_fut['city'] = d_fut['city'].map(norm_city)

    all_dem = (pd.concat([d_hist, d_fut], ignore_index=True)
                 .drop_duplicates(subset=['date','sku','city'], keep='last')
                 .sort_values('date'))

    # 도시→창고 매핑(선정된 창고 중 최근접)
    selected_whs = set(sourcing_policy.keys())
    city_ll = sites_df.groupby('city')[['lat','lon']].mean().reset_index().set_index('city')
    def nearest_wh_for_city(city_name):
        if city_name not in city_ll.index: return None
        lat_c, lon_c = city_ll.loc[city_name, ['lat','lon']]
        best, best_d = None, 1e18
        for wh in selected_whs:
            lat_w, lon_w = site_info.loc[wh, ['lat','lon']]
            d = haversine_km(lat_c, lon_c, lat_w, lon_w)
            if d < best_d: best, best_d = wh, d
        return best
    city_to_wh = {c: nearest_wh_for_city(c) for c in all_dem['city'].unique()}

    # (wh,sku)별 일평균 수요 (Order-Up-To S 계산용)
    dem_tmp = all_dem.copy()
    dem_tmp['warehouse'] = dem_tmp['city'].map(city_to_wh)
    dem_tmp = dem_tmp[dem_tmp['warehouse'].notna()]
    mean_demand_wh_sku = (dem_tmp.groupby(['warehouse','sku'])['demand']
                          .mean()
                          .to_dict())

    # 악천후/유가 급등 할증 준비
    wx_df  = pd.read_csv(DATA_DIR / 'weather.csv')
    wx_df['date'] = pd.to_datetime(wx_df['date'])
    for col in ['rain_mm','snow_cm','wind_mps','cloud_pct']:
        if col not in wx_df.columns: wx_df[col] = 0.0
    weather_by_cc = {(r['country'], r['date'].date()):(float(r['rain_mm']), float(r['snow_cm']), float(r['wind_mps']), float(r['cloud_pct']))
                     for _, r in wx_df.iterrows()}
    oil_df = pd.read_csv(DATA_DIR / 'oil_price.csv'); oil_df['date'] = pd.to_datetime(oil_df['date'])
    oil_mons = oil_df[oil_df['date'].dt.weekday==0].sort_values('date')
    oil_jump_weeks = set()
    prev = None
    for _, r in oil_mons.iterrows():
        if prev is not None and prev>0 and (r['brent_usd']-prev)/prev >= OIL_JUMP_PCT:
            start = r['date'].date()
            for k in range(7): oil_jump_weeks.add(start + timedelta(days=k))
        prev = r['brent_usd']

    # ---------- (2) 주간 용량 캘린더 ffill ----------
    print("\n[3-1-2] 주간 공장 용량 시계열 보정(ffill)...")
    all_weeks = pd.date_range(DATE_START, DATE_END, freq='W-MON').date
    cap_raw = (fac_cap_df
               .assign(week=lambda x: pd.to_datetime(x['week']).dt.date)
               .groupby(['factory','week'])[['reg_capacity','ot_capacity']].sum()
               .reset_index())
    week_reg_cap, week_ot_cap = {}, {}
    for fac, g in cap_raw.groupby('factory'):
        g2 = (pd.DataFrame({'week': all_weeks})
              .merge(g, on='week', how='left')
              .sort_values('week'))
        g2[['reg_capacity','ot_capacity']] = g2[['reg_capacity','ot_capacity']].ffill().fillna(0.0)
        for _, r in g2.iterrows():
            week_reg_cap[(fac, r['week'])] = float(r['reg_capacity'])
            week_ot_cap[(fac, r['week'])]  = float(r['ot_capacity'])

    # ---------- (3) 상태변수 ----------
    print("\n[3-2] 시뮬레이션 상태 초기화 중...")
    # 재고
    inventory = defaultdict(float)
    for wh in sourcing_policy.keys():
        for sku in inventory_policy.get(wh, {}).keys():
            inventory[(wh, sku)] = 2000.0   # 초기 버퍼(원하면 0으로)

    open_pipeline = defaultdict(float)             # (wh,sku) 오는 길 위 수량
    in_transit = defaultdict(list)                 # arr_date -> [(wh, sku, qty, fac, mode, dist_km, origin_cty)]
    factory_orders = defaultdict(deque)            # fac -> deque of {'wh','sku','qty','attempt','preferred_mode'}

    # 주간 공장 시간(정규/OT) 사용량 (월요일 키로 구분)
    week_reg_used = defaultdict(float)
    week_ot_used  = defaultdict(float)

    # 비용/지표 집계
    daily_cost_rows = []
    cost_accum = {"transport":0.0, "holding":0.0, "shortage":0.0, "production_labor":0.0, "production_base":0.0}
    shortages_log = []

    # FR 집계: 주간 단위(월요일 시작)
    dem_week  = defaultdict(float)                 # (city, sku, week_monday) -> demand
    ship_week = defaultdict(float)                 # (city, sku, week_monday) -> shipped
    total_dem_all  = 0.0
    total_ship_all = 0.0

    # 제출 DB 행들
    db_rows = []  # date,factory/warehouse,sku,production_qty,ot_qty,ship_qty,from,to,mode

    # ---------------- (4) 일일 루프 ----------------
    print("\n[3-3] 일일 시뮬레이션 실행 (2018-01-01 ~ 2024-12-31)...")
    for t_current in pd.date_range(DATE_START, DATE_END, freq='D'):
        if t_current.day == 1:
            print(f"  -> 진행 중: {t_current.date()}")

        wk_monday = monday_of(t_current)

        # ---- A) 오늘 도착 처리 (공장→창고) ----
        for dest_wh, sku, qty, fac, mode, dist_km, origin_cty in in_transit.pop(t_current.date(), []):
            inventory[(dest_wh, sku)] += qty
            # open_pipeline: 선적 시 +, 입고 시 -
            open_pipeline[(dest_wh, sku)] = max(0.0, open_pipeline[(dest_wh, sku)] - qty)

            # 원한다면 입고 레코드도 별도 테이블/CSV에 남길 수 있음 (여긴 제출 포맷만 기록)

        # ---- B) 수요 처리 (창고→도시) + 품절비/FR ----
        today_dem = all_dem[all_dem['date'] == t_current]
        day_trans_cost = 0.0
        day_hold_cost  = 0.0
        day_short_cost = 0.0
        day_prod_labor = 0.0
        day_prod_base  = 0.0

        for _, r in today_dem.iterrows():
            city = r['city']; sku = r['sku']; dqty = float(r['demand'] or 0.0)
            if dqty <= 0: continue
            wh = city_to_wh.get(city)
            if not wh:   continue

            avail = inventory.get((wh, sku), 0.0)
            ship  = min(avail, dqty)
            shortage = max(0.0, dqty - ship)

            if ship > 0:
                inventory[(wh, sku)] = avail - ship
                db_rows.append({
                    "date": t_current.strftime("%Y-%m-%d"),
                    "factory/warehouse": wh,
                    "sku": sku,
                    "production_qty": 0,
                    "ot_qty": 0,
                    "ship_qty": int(round(ship)),
                    "from": wh,
                    "to": city,
                    "mode": "TRUCK"   # 라스트마일은 트럭
                })

            if shortage > 0:
                unit_short = float(SHORT.get(sku, 0.0))
                day_short_cost += unit_short * shortage
                shortages_log.append({
                    "date": t_current.strftime("%Y-%m-%d"),
                    "city": city,
                    "sku": sku,
                    "short_qty": round(shortage, 2)
                })

            dem_week[(city, sku, wk_monday)]  += dqty
            ship_week[(city, sku, wk_monday)] += ship
            total_dem_all  += dqty
            total_ship_all += ship

        # ---- C) 재고 보충 주문 (ROP 트리거, S까지 보충, 고갈예측 시 AIR) ----
        for wh, sku_dict in inventory_policy.items():
            for sku, pol in sku_dict.items():
                on_hand = inventory.get((wh, sku), 0.0)
                in_pipe = open_pipeline.get((wh, sku), 0.0)
                rop     = float(pol.get('reorder_point', 0))
                ss      = float(pol.get('safety_stock', 0))

                if on_hand + in_pipe <= rop:
                    # 목표재고 S = SS + (일평균수요 * LT * 2)
                    primary_fac = sourcing_policy[wh]['primary_source']['factory']
                    # 리드타임: 계획 파일 값이 있으면 그걸 우선 사용
                    lt_days_planned = sourcing_policy[wh]['primary_source'].get('lead_time', None)
                    if lt_days_planned is None:
                        lt_days_planned = compute_lead_days(site_info, mode_meta, primary_fac, wh, default_mode="SHIP")[0]
                    mean_d = float(mean_demand_wh_sku.get((wh, sku), 0.0))
                    target_S = ss + mean_d * lt_days_planned * 2.0

                    order_qty = max(0.0, target_S - (on_hand + in_pipe))
                    if order_qty > 0:
                        # 기본 모드
                        if site_info.loc[primary_fac,'country'] == site_info.loc[wh,'country']:
                            default_mode = "TRUCK"
                        else:
                            default_mode = "SHIP"
                        # 고갈예측 기준으로 AIR 강제
                        days_to_depletion = (on_hand / max(mean_d, 1e-6)) if mean_d > 0 else 1e9
                        preferred_mode = "AIR" if days_to_depletion <= lt_days_planned else default_mode

                        # (A) open_pipeline 이중가산 방지: 주문 시에는 + 하지 않음
                        factory_orders[primary_fac].append({
                            'wh': wh, 'sku': sku, 'qty': order_qty,
                            'attempt': 1, 'preferred_mode': preferred_mode
                        })

        # ---- D) 생산 & 출고 (주간 정규/OT로 정확히 처리) ----
        fail_today = {}
        if not mfail.empty:
            for _, rr in mfail.iterrows():
                if rr['start_date'].date() <= t_current.date() <= rr['end_date'].date():
                    fail_today[rr['factory']] = True

        for fac, q in list(factory_orders.items()):
            if not q: continue

            fac_ctry = site_info.loc[fac, 'country']
            reg_cap = week_reg_cap.get((fac, wk_monday), 0.0)
            ot_cap  = week_ot_cap.get((fac, wk_monday), 0.0)
            reg_used= week_reg_used[(fac, wk_monday)]
            ot_used = week_ot_used[(fac, wk_monday)]
            reg_left= max(0.0, reg_cap - reg_used)
            ot_left = max(0.0, ot_cap  - ot_used)

            # 휴일/주말이면 정규 0
            if HOL.get((fac_ctry, t_current.date()), 0) == 1 or t_current.weekday() >= 5:
                reg_left_today = 0.0
            else:
                reg_left_today = reg_left

            # 고장이면 오늘 생산 불가
            if fail_today.get(fac, False):
                reg_left_today = 0.0
                ot_left = 0.0

            new_q = deque()
            while q and (reg_left_today > 1e-9 or ot_left > 1e-9):
                order = q.popleft()
                to_wh = order['wh']
                sku   = order['sku']
                qty   = float(order['qty'])
                preferred = order['preferred_mode']
                hrs_per_unit = float(a_hours.get(sku, 0.0))

                if qty <= 1e-9 or hrs_per_unit <= 0.0:
                    continue

                total_hours_need = qty * hrs_per_unit

                # 정규→OT 순 배정
                reg_h_assign = min(reg_left_today, total_hours_need)
                rem_hours    = total_hours_need - reg_h_assign
                ot_h_assign  = min(ot_left, rem_hours)
                rem_hours   -= ot_h_assign

                reg_units = int(np.floor(reg_h_assign / hrs_per_unit)) if hrs_per_unit>0 else 0
                reg_h_used= reg_units * hrs_per_unit
                ot_units  = int(np.floor(ot_h_assign  / hrs_per_unit)) if hrs_per_unit>0 else 0
                ot_h_used = ot_units * hrs_per_unit

                produced_units = reg_units + ot_units
                leftover_qty   = max(0.0, qty - produced_units)

                # 임금(USD)
                pol_row = lab_pol[(lab_pol['country']==fac_ctry) & (lab_pol['year']==t_current.year)]
                if pol_row.empty:
                    reg_wage_local = 0.0
                    ot_mult = 1.5
                else:
                    reg_wage_local = float(pol_row['regular_wage_local'].iloc[0])
                    ot_mult        = float(pol_row['ot_mult'].iloc[0])
                usd_per_local = FX.get((fac_ctry, t_current.date()), 1.0)
                reg_wage_usd  = reg_wage_local * usd_per_local
                labor_cost = reg_h_used * reg_wage_usd + ot_h_used * reg_wage_usd * ot_mult
                day_prod_labor += labor_cost

                # 비노동 생산원가
                base_c = float(base_cost.get((sku, fac), 0.0))
                base_cost_total = base_c * produced_units
                day_prod_base  += base_cost_total

                # 리드타임/운송모드/거리
                default_mode = preferred
                lead_days, mode_used, dist_km, c_f, c_w = compute_lead_days(site_info, mode_meta, fac, to_wh, default_mode=default_mode)
                arr_day = (t_current + timedelta(days=lead_days)).date()

                # 운송비(할증 포함)
                beta = float(mode_meta.loc[mode_used, 'cost_per_km_factor'])
                per_km = TRUCK_BASE_COST_PER_KM * beta
                base_trans_cost = per_km * dist_km * produced_units
                border = BORDER_FEE_PER_UNIT * produced_units if is_cross_border(c_f, c_w) else 0.0
                wx_mult  = WEATHER_MULT_BAD if is_bad_weather(weather_by_cc, c_f, t_current) else 1.0
                oil_mult = OIL_MULT_JUMP if oil_jump_week(oil_jump_weeks, t_current) else 1.0
                trans_cost = (base_trans_cost + border) * wx_mult * oil_mult
                day_trans_cost += trans_cost

                # 선적(오늘 출발, 도착일 입고)
                if produced_units > 0:
                    in_transit[arr_day].append((to_wh, sku, produced_units, fac, mode_used, dist_km, c_f))
                    # (A) open_pipeline: 선적 시 +
                    open_pipeline[(to_wh, sku)] += produced_units

                    # DB: 생산(정규/OT 분리) + 공장→창고 선적
                    if reg_units > 0:
                        db_rows.append({
                            "date": t_current.strftime("%Y-%m-%d"),
                            "factory/warehouse": fac,
                            "sku": sku,
                            "production_qty": int(reg_units),
                            "ot_qty": 0,
                            "ship_qty": 0,
                            "from": fac,
                            "to": None,
                            "mode": None
                        })
                    if ot_units > 0:
                        db_rows.append({
                            "date": t_current.strftime("%Y-%m-%d"),
                            "factory/warehouse": fac,
                            "sku": sku,
                            "production_qty": 0,
                            "ot_qty": int(ot_units),
                            "ship_qty": 0,
                            "from": fac,
                            "to": None,
                            "mode": None
                        })
                    db_rows.append({
                        "date": t_current.strftime("%Y-%m-%d"),
                        "factory/warehouse": fac,
                        "sku": sku,
                        "production_qty": 0,
                        "ot_qty": 0,
                        "ship_qty": int(produced_units),
                        "from": fac,
                        "to": to_wh,
                        "mode": mode_used
                    })

                # 시간 소모 업데이트
                reg_left_today -= reg_h_used
                reg_left_today  = max(0.0, reg_left_today)
                reg_used       += reg_h_used
                ot_left        -= ot_h_used
                ot_left         = max(0.0, ot_left)
                ot_used        += ot_h_used

                if leftover_qty > 1e-9:
                    order['qty'] = leftover_qty
                    new_q.append(order)

            while q:
                new_q.append(q.popleft())
            factory_orders[fac] = new_q

            week_reg_used[(fac, wk_monday)] = reg_used
            week_ot_used[(fac, wk_monday)]  = ot_used

        # ---- E) 일말 보관비 ----
        for (wh, sku), qv in list(inventory.items()):
            if qv <= 0: continue
            unit_hold = float(HOLD.get(sku, 0.0))
            day_hold_cost += unit_hold * qv

        # ---- F) 일일 비용 합산/저장 ----
        cost_accum["transport"]        += day_trans_cost
        cost_accum["holding"]          += day_hold_cost
        cost_accum["shortage"]         += day_short_cost
        cost_accum["production_labor"] += day_prod_labor
        cost_accum["production_base"]  += day_prod_base

        daily_cost_rows.append({
            "date": t_current.strftime("%Y-%m-%d"),
            "transport_cost": round(day_trans_cost, 2),
            "holding_cost":   round(day_hold_cost, 2),
            "shortage_cost":  round(day_short_cost, 2),
            "production_labor_cost": round(day_prod_labor, 2),
            "production_base_cost":  round(day_prod_base, 2),
            "total_cost":     round(day_trans_cost + day_hold_cost + day_short_cost + day_prod_labor + day_prod_base, 2),
        })

    # ---------------- (5) DB 및 리포트 저장 ----------------
    print("\n[3-4] DB/리포트 저장 중...")

    # 제출 DB
    submission_df = pd.DataFrame(
        db_rows,
        columns=["date","factory/warehouse","sku","production_qty","ot_qty","ship_qty","from","to","mode"]
    )
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
    with sqlite3.connect(DB_NAME) as conn:
        submission_df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    # 비용 일일 리포트
    cost_daily = pd.DataFrame(daily_cost_rows)
    cost_daily.to_csv("cost_daily.csv", index=False)

    # Fill-Rate 리포트 (주간/전체)
    fr_rows = []
    for (city, sku, wk), d in dem_week.items():
        s = ship_week.get((city, sku, wk), 0.0)
        fr = 0.0 if d <= 0 else (s / d)
        fr_rows.append({
            "week_monday": wk.strftime("%Y-%m-%d"),
            "city": city,
            "sku": sku,
            "demand": round(d, 2),
            "shipped": round(s, 2),
            "fill_rate": round(fr, 4)
        })
    fr_weekly = pd.DataFrame(fr_rows).sort_values(["week_monday","city","sku"])
    fr_weekly.to_csv("fr_weekly.csv", index=False)

    fr_overall = pd.DataFrame([{
        "total_demand": round(total_dem_all, 2),
        "total_shipped": round(total_ship_all, 2),
        "fill_rate_overall": 0.0 if total_dem_all<=0 else round(total_ship_all/total_dem_all, 4),
        "transport_cost": round(cost_accum["transport"], 2),
        "holding_cost": round(cost_accum["holding"], 2),
        "shortage_cost": round(cost_accum["shortage"], 2),
        "production_labor_cost": round(cost_accum["production_labor"], 2),
        "production_base_cost": round(cost_accum["production_base"], 2),
        "total_cost": round(sum(cost_accum.values()), 2)
    }])
    fr_overall.to_csv("fr_overall.csv", index=False)

    # 품절 상세 로그
    pd.DataFrame(shortages_log).to_csv("shortages_log.csv", index=False)

    print(f"\n--- 3단계 시뮬레이션 완료 ---")
    print(f"- 제출 DB: {DB_NAME} (table: {TABLE_NAME})")
    print(f"- 비용 일일 리포트: cost_daily.csv")
    print(f"- 주간/전체 FR: fr_weekly.csv / fr_overall.csv")
    print(f"- 품절 로그: shortages_log.csv")

if __name__ == "__main__":
    main_stage3_simulator()