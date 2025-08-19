# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sqlite3, math, json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum

# ===================== CONFIG =====================
DATA_DIR   = Path("data")
DATE_START = datetime(2018,1,1)
DATE_END   = datetime(2024,12,31)

MAX_FAC = 5
MAX_WH  = 20

TRUCK_BASE_COST_PER_KM = 12.0
BORDER_COST = 4000.0                 # 해외(국경 통과) 벌점
MAX_CROSS_BORDER_KM = 1000         # 해외 링크 최대 허용 거리 (km)

# ==================================================

def norm_city(x: str) -> str:
    return str(x).replace(' ', '_')

def haversine_km(lat1, lon1, lat2, lon2, R=6371.0):
    p1, l1, p2, l2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((p2 - p1)/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin((l1 - l2)/2.0)**2
    return float(2*R*np.arcsin(np.sqrt(a)))

def week_of(dt: datetime) -> int:
    return (dt.date() - DATE_START.date()).days // 7

def main():
    # ---------- load
    sites     = pd.read_csv(DATA_DIR/'site_candidates.csv')
    sites['city'] = sites['city'].map(norm_city)
    init_cost = pd.read_csv(DATA_DIR/'site_init_cost.csv')
    mode_meta = pd.read_csv(DATA_DIR/'transport_mode_meta.csv')
    fac_cap   = pd.read_csv(DATA_DIR/'factory_capacity.csv')

    # 수요: 2018-2022 실측 + 2023-2024 예측 → 전체기간 총합
    # demand_train.db (있으면 사용)
    try:
        con = sqlite3.connect(DATA_DIR/'demand_train.db')
        dem_hist = pd.read_sql_query("SELECT date, sku, city, demand FROM demand_train", con)
        con.close()
        dem_hist['date'] = pd.to_datetime(dem_hist['date'])
        dem_hist['city'] = dem_hist['city'].map(norm_city)
    except Exception:
        dem_hist = pd.DataFrame(columns=['date','sku','city','demand'])

    # forecast_submission_template.csv
    dem_fc = pd.read_csv(DATA_DIR/'forecast_submission_template.csv')
    dem_fc['date'] = pd.to_datetime(dem_fc['date'])
    dem_fc['city'] = dem_fc['city'].map(norm_city)
    dem_fc = dem_fc.rename(columns={'mean':'demand'})

    # 기간 필터 & 결합 → 도시 총수요
    dem = pd.concat([dem_hist, dem_fc], ignore_index=True)
    dem = dem[(dem['date']>=DATE_START)&(dem['date']<=DATE_END)]
    D_j = dem.groupby('city', as_index=False)['demand'].sum().rename(columns={'demand':'Q_city'})
    cities = sorted(D_j['city'].unique().tolist())

    # ---------- sets
    fac_df = sites[sites['site_type']=='factory'].copy()
    wh_df  = sites[sites['site_type']=='warehouse'].copy()

    I = fac_df['site_id'].tolist()
    K = wh_df['site_id'].tolist()
    J = cities

    site_lat = dict(zip(sites['site_id'], sites['lat']))
    site_lon = dict(zip(sites['site_id'], sites['lon']))
    site_country = dict(zip(sites['site_id'], sites['country']))
    site_city    = dict(zip(sites['site_id'], sites['city']))

    # 도시 좌표(후보지 동일도시 평균으로 근사)
    city_ll = sites.groupby('city')[['lat','lon']].mean().reset_index().set_index('city')
    city_lat = city_ll['lat'].to_dict()
    city_lon = city_ll['lon'].to_dict()

    # 도시→국가(과제 제공 목록)
    city_country = {
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

    # ---------- distances
    dist_ik = {(i,k): haversine_km(site_lat[i], site_lon[i], site_lat[k], site_lon[k]) for i in I for k in K}
    dist_kj = {(k,j): haversine_km(site_lat[k], site_lon[k], city_lat[j], city_lon[j]) for k in K for j in J}

    # ---------- transport per km cost ----------
    # 국내: TRUCK / 해외: SHIP (AIR 사용 안함)
    beta = dict(zip(mode_meta['mode'], mode_meta['cost_per_km_factor']))
    costpkm_truck = TRUCK_BASE_COST_PER_KM * beta['TRUCK']
    costpkm_ship  = TRUCK_BASE_COST_PER_KM * beta['SHIP']

    def is_cross_border_u2v(cu: str, cv: str) -> bool:
        return cu != cv

    def border_cost(cu: str, cv: str) -> float:
        return 0.0 if not is_cross_border_u2v(cu, cv) else BORDER_COST

    def per_km_cost_factory_to_wh(i, k):
        cu, cv = site_country[i], site_country[k]
        return costpkm_truck if not is_cross_border_u2v(cu, cv) else costpkm_ship

    def per_km_cost_wh_to_city(k, j):
        cu, cv = site_country[k], city_country[j]
        return costpkm_truck if not is_cross_border_u2v(cu, cv) else costpkm_ship

    # ---------- capacity proxy (주간 → 총합 근사)
    fac_cap['week_idx'] = fac_cap['week'].pipe(pd.to_datetime).apply(week_of)
    cap_week = (fac_cap.groupby(['factory','week_idx'], as_index=False)
                      .agg(reg=('reg_capacity','sum'), ot=('ot_capacity','sum')))
    cap_by_fac = cap_week.groupby('factory')[['reg','ot']].sum().sum(axis=1).to_dict()

    total_Q = float(D_j['Q_city'].sum())  # 전체 기간 총수요

    # ---------- model
    m = Model("SITE_LOCATION_ONLY")
    m.Params.OutputFlag = 0

    # vars
    x_fac = m.addVars(I, vtype=GRB.BINARY, name="x_fac")
    x_wh  = m.addVars(K, vtype=GRB.BINARY, name="x_wh")

    # 도시→창고 배정(연속 흐름)
    a_kj = m.addVars(K, J, lb=0.0, name="assign_kj")

    # 창고→공장 조달(연속 흐름)
    f_ik = m.addVars(I, K, lb=0.0, name="flow_ik")

    # 선택·개수 제약(최대 개수)
    m.addConstr(x_fac.sum() <= MAX_FAC, "fac_count")
    m.addConstr(x_wh.sum()  <= MAX_WH,  "wh_count")

    # 같은 도시에 공장/창고 1개 규칙
    for city, g in fac_df.groupby('city'):
        m.addConstr(quicksum(x_fac[i] for i in g['site_id']) <= 1, f"one_fac_per_city_{city}")
    for city, g in wh_df.groupby('city'):
        m.addConstr(quicksum(x_wh[k] for k in g['site_id']) <= 1, f"one_wh_per_city_{city}")

    # 도시 수요 보존
    Q_by_city = dict(zip(D_j['city'], D_j['Q_city']))
    for j in J:
        m.addConstr(quicksum(a_kj[k,j] for k in K) == Q_by_city[j], f"city_assign_{j}")

    # 창고가 열려야 배정 가능
    for k in K:
        for j in J:
            m.addConstr(a_kj[k,j] <= Q_by_city[j] * x_wh[k], f"use_open_wh_{k}_{j}")

    # 창고 수요 = 배정 합
    wh_demand = {k: quicksum(a_kj[k,j] for j in J) for k in K}

    # 공장 흐름 ≥ 해당 창고 수요 (공장들이 분담)
    for k in K:
        m.addConstr(quicksum(f_ik[i,k] for i in I) >= wh_demand[k], f"cover_wh_{k}")

    # 공장이 열려야 출하 가능
    BIGM = total_Q
    for i in I:
        for k in K:
            m.addConstr(f_ik[i,k] <= BIGM * x_fac[i], f"use_open_fac_{i}_{k}")

    # ---------- CROSS-BORDER DISTANCE CAP (핵심 규칙) ----------
    # 해외 링크이며 거리 > 1000km 인 경우, 흐름 0으로 강제
    '''for i in I:
        for k in K:
            if is_cross_border_u2v(site_country[i], site_country[k]) and dist_ik[(i,k)] > MAX_CROSS_BORDER_KM:
                m.addConstr(f_ik[i,k] == 0.0, name=f"cap_ik_{i}_{k}")
    '''
    for k in K:
        for j in J:
            if is_cross_border_u2v(site_country[k], city_country[j]) and dist_kj[(k,j)] > MAX_CROSS_BORDER_KM:
                m.addConstr(a_kj[k,j] == 0.0, name=f"cap_kj_{k}_{j}")

    # ---------- 비용: 건설 + 운송 ----------
    init_cost_map = dict(zip(init_cost['site_id'], init_cost['init_cost_usd']))

    # 공장→창고 운송비
    cost_ik_terms = []
    for i in I:
        for k in K:
            d = dist_ik[(i,k)]
            cu, cv = site_country[i], site_country[k]
            perkm = per_km_cost_factory_to_wh(i,k)
            bfee  = border_cost(cu, cv)   # 단위당 근사 벌점
            cost_ik_terms.append((perkm*d) * f_ik[i,k] + bfee * f_ik[i,k])

    # 창고→도시 운송비
    cost_kj_terms = []
    for k in K:
        for j in J:
            d = dist_kj[(k,j)]
            cu, cv = site_country[k], city_country[j]
            perkm = per_km_cost_wh_to_city(k,j)
            bfee  = border_cost(cu, cv)
            cost_kj_terms.append((perkm*d) * a_kj[k,j] + bfee * a_kj[k,j])

    build_cost = quicksum(init_cost_map.get(i,0.0)*x_fac[i] for i in I) + \
                 quicksum(init_cost_map.get(k,0.0)*x_wh[k]  for k in K)

    m.setObjective(build_cost + quicksum(cost_ik_terms) + quicksum(cost_kj_terms), GRB.MINIMIZE)

    # ---------- solve ----------
    m.optimize()
    if m.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT] or m.SolCount == 0:
        print("No solution.")
        return

    # ---------- EXPORT ROUTES ----------
    # (1) 공장→창고
    rows_ik = []
    for i in I:
        for k in K:
            q = float(f_ik[i,k].X)
            if q <= 1e-9: continue
            d = dist_ik[(i,k)]
            cu, cv = site_country[i], site_country[k]

            # 표기 모드: 국내=TRUCK, 해외=SHIP
            chosen_mode = 'TRUCK' if not is_cross_border_u2v(cu, cv) else 'SHIP'
            perkm_tbl = {'TRUCK': costpkm_truck, 'SHIP': costpkm_ship}
            perkm = perkm_tbl[chosen_mode]
            bfee  = border_cost(cu, cv)

            var_cost   = perkm * d * q
            total_cost = var_cost + bfee * q

            rows_ik.append({
                "factory": i,
                "warehouse": k,
                "flow_units": q,
                "distance_km": d,
                "cross_border": int(is_cross_border_u2v(cu, cv)),
                "mode": chosen_mode,
                "var_cost_usd": perkm * d,   # 단위당
                "border_fee_usd": bfee,      # 단위당
                "total_cost_usd": total_cost
            })

    df_routes_ik = pd.DataFrame(rows_ik)
    if not df_routes_ik.empty:
        df_routes_ik.sort_values(["factory","warehouse"], inplace=True)
        df_routes_ik.to_csv("routes_factory_to_warehouse.csv", index=False)

    # (2) 창고→도시
    rows_kj = []
    for k in K:
        for j in J:
            q = float(a_kj[k,j].X)
            if q <= 1e-9: continue
            d = dist_kj[(k,j)]
            cu, cv = site_country[k], city_country[j]

            chosen_mode = 'TRUCK' if not is_cross_border_u2v(cu, cv) else 'SHIP'
            perkm_tbl = {'TRUCK': costpkm_truck, 'SHIP': costpkm_ship}
            perkm = perkm_tbl[chosen_mode]
            bfee  = border_cost(cu, cv)

            var_cost   = perkm * d * q
            total_cost = var_cost + bfee * q

            rows_kj.append({
                "warehouse": k,
                "city": j,
                "flow_units": q,
                "distance_km": d,
                "cross_border": int(is_cross_border_u2v(cu, cv)),
                "mode": chosen_mode,
                "var_cost_usd": perkm * d,   # 단위당
                "border_fee_usd": bfee,      # 단위당
                "total_cost_usd": total_cost
            })

    df_routes_kj = pd.DataFrame(rows_kj)
    if not df_routes_kj.empty:
        df_routes_kj.sort_values(["warehouse","city"], inplace=True)
        df_routes_kj.to_csv("routes_warehouse_to_city.csv", index=False)

        # (3) 도시별 주 경로(가장 많이 배정된 창고)
        top_assign = (df_routes_kj.groupby("city", as_index=False)
                                 .apply(lambda g: g.loc[g["flow_units"].idxmax(),
                                                        ["warehouse","flow_units","total_cost_usd"]])
                                 .reset_index(drop=True))
        top_assign.rename(columns={"warehouse":"primary_warehouse",
                                   "flow_units":"primary_flow_units",
                                   "total_cost_usd":"primary_total_cost_usd"}, inplace=True)
        top_assign.to_csv("city_primary_warehouse.csv", index=False)

    print("→ routes_factory_to_warehouse.csv / routes_warehouse_to_city.csv 저장 완료")

    # ---------- OPENED SITES ----------
    opened_fac = [i for i in I if x_fac[i].X > 0.5]
    opened_wh  = [k for k in K if x_wh[k].X  > 0.5]
    print(f"Selected factories: {len(opened_fac)} / warehouses: {len(opened_wh)}")
    print("Factories:", opened_fac)
    print("Warehouses:", opened_wh)

    fixed_build = {
        "fac_on": {i: (1 if i in opened_fac else 0) for i in I},
        "wh_on" : {k: (1 if k in opened_wh  else 0) for k in K},
    }

    with open("fixed_build.json", "w") as f:
        json.dump(fixed_build, f, indent=2)

    pd.DataFrame({
        "factory": opened_fac,
        "country": [site_country[i] for i in opened_fac],
        "city":    [site_city[i] for i in opened_fac]
    }).to_csv("selected_factories.csv", index=False)

    pd.DataFrame({
        "warehouse": opened_wh,
        "country":   [site_country[k] for k in opened_wh],
        "city":      [site_city[k] for k in opened_wh]
    }).to_csv("selected_warehouses.csv", index=False)

    print("→ fixed_build.json / selected_*.csv 저장 완료")

if __name__ == "__main__":
    main()