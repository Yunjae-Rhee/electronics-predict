# -*- coding: utf-8 -*-
# ============================================================================
#  STAGE 2 TACTICAL PLANNER (Capacity-Aware Multi-Sourcing)
#  - Solver(Stage 3)와 데이터 로딩/전처리 로직을 공유하여 호환성 확보
#  - Stage 1 결과(selected_*.csv)를 입력받아 다음을 결정:
#    1. 생산 능력 균등화를 고려한 주/예비 공급처(Primary/Secondary Source)
#    2. 주 공급처 리드타임 기반의 안전 재고 및 재주문점(SS/ROP)
#  - 결과물: tactical_plan_advanced.json (Stage 3 시뮬레이션용)
# ============================================================================
from __future__ import annotations
import math, json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

# --- 2단계 계획용 상수 ---
DATA_DIR = Path("data")
Z_SCORE_95 = 1.645       # 95% 서비스 수준(Fill-Rate)에 해당하는 Z-score
CAPACITY_THRESHOLD = 1.0 # 부하율 임계값 (1.0 = 100%)

# --- 유틸 함수 (기존 솔버와 동일) ---
def haversine_km(lat1, lon1, lat2, lon2, R=6371.0):
    p1, l1, p2, l2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((p2 - p1)/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin((l1 - l2)/2.0)**2
    return float(2*R*np.arcsin(np.sqrt(a)))

def norm_city(name): return str(name).replace(' ', '_')
def ceil_pos(x): return 0 if x<=0 else int(math.ceil(x))
def truck_days_from_distance(dkm):
    d = max(0.0, float(dkm))
    if d <= 500.0:   return 2
    elif d <= 1000.: return 3
    elif d <= 2000.: return 5
    else:            return 8

def main_stage2_planner():
    """
    1단계 결과를 바탕으로 2단계 전술 계획(소싱/재고 정책)을 수립하는 메인 함수
    """
    print("--- 2단계 정교한 전술 계획 수립 시작 ---")

    # =================== LOAD DATA (기존 솔버와 동일한 로직) ========================
    print("Loading master tables…")
    sites     = pd.read_csv(f'{DATA_DIR}/site_candidates.csv'); sites['city']=sites['city'].map(norm_city)
    mode_meta = pd.read_csv(f'{DATA_DIR}/transport_mode_meta.csv')
    fac_cap   = pd.read_csv(f'{DATA_DIR}/factory_capacity.csv')
    
    # 수요 데이터 로드
    try:
        con = sqlite3.connect(f'{DATA_DIR}/demand_train.db')
        d_hist = pd.read_sql_query("SELECT date, sku, city, demand FROM demand_train", con, parse_dates=['date'])
        con.close()
        d_hist['city']=d_hist['city'].map(norm_city)
    except Exception:
        d_hist = pd.DataFrame()
    
    d_fut = pd.read_csv(f'{DATA_DIR}/forecast_submission_template.csv', parse_dates=['date'])
    d_fut['city']=d_fut['city'].map(norm_city)
    d_fut.rename(columns={'mean':'demand'}, inplace=True)
    all_dem = pd.concat([d_hist, d_fut], ignore_index=True).drop_duplicates(subset=['date','sku','city'], keep='last')

    # 집합(Sets) 정의
    fac_df = sites[sites['site_type']=='factory'].copy()
    wh_df  = sites[sites['site_type']=='warehouse'].copy()
    I = fac_df['site_id'].tolist()
    K = wh_df['site_id'].tolist()
    
    site_country = dict(zip(sites['site_id'], sites['country']))
    site_lat     = dict(zip(sites['site_id'], sites['lat']))
    site_lon     = dict(zip(sites['site_id'], sites['lon']))
    
    # 거리 및 리드타임 사전 계산
    modes = mode_meta['mode'].tolist()
    alpha_lead = dict(zip(mode_meta['mode'], mode_meta['leadtime_factor']))
    dist_ik = {(i,k): haversine_km(site_lat[i], site_lon[i], site_lat[k], site_lon[k]) for i in I for k in K}
    L_ik = {m:{} for m in modes}
    for (i,k), d in dist_ik.items():
        base_days = truck_days_from_distance(d)
        for m in modes: L_ik[m][(i,k)] = ceil_pos(alpha_lead[m] * base_days)
    
    # --- 1단계 결과물 로드 ---
    try:
        selected_factories = pd.read_csv("selected_factories.csv")['factory'].tolist()
        selected_warehouses = pd.read_csv("selected_warehouses.csv")['warehouse'].tolist()
    except FileNotFoundError:
        print("오류: 1단계 결과 파일('selected_factories.csv', 'selected_warehouses.csv')이 필요합니다.")
        return

    # --- 2. 다중 소싱 옵션 계산 (선택된 시설 대상) ---
    print("\n[2-1] 창고별 다중 소싱 옵션 계산 중...")
    wh_sourcing_options = defaultdict(list)
    for wh_id in selected_warehouses:
        options = []
        for fac_id in selected_factories:
            is_international = (site_country[wh_id] != site_country[fac_id])
            mode = 'SHIP' if is_international else 'TRUCK'
            lead_time = L_ik[mode][(fac_id, wh_id)]
            cost = dist_ik[(fac_id, wh_id)] # 비용은 거리로 근사
            options.append({'factory': fac_id, 'cost': cost, 'lead_time': int(lead_time)})
        wh_sourcing_options[wh_id] = sorted(options, key=lambda x: x['cost'])

   # --- 3. 생산 능력 균등화 ---
    print("\n[2-2] 생산 능력 균등화 작업 시작...")
    # 도시-창고 매핑 (수정된 코드)
    city_to_wh_map = {}
    # 도시의 대표 좌표를 미리 계산합니다 (동일 도시 내 후보지들의 평균 좌표).
    city_coords = sites.groupby('city')[['lat', 'lon']].mean()

    # 수요 데이터에 등장하는 모든 고유한 도시 목록을 가져옵니다.
    unique_cities_in_demand = all_dem['city'].unique()

    for city_name in unique_cities_in_demand:
        if city_name not in city_coords.index:
            continue # sites 테이블에 좌표 정보가 없는 도시는 건너뜁니다.
            
        city_lat, city_lon = city_coords.loc[city_name]
        
        # 각 도시에서 가장 가까운 창고를 찾습니다.
        distances = {
            wh_id: haversine_km(city_lat, city_lon, site_lat[wh_id], site_lon[wh_id])
            for wh_id in selected_warehouses
        }
        assigned_wh = min(distances, key=distances.get)
        city_to_wh_map[city_name] = assigned_wh # <-- 해결: 'Berlin' 같은 도시 이름을 키로 사용

    
    all_dem['warehouse'] = all_dem['city'].map(city_to_wh_map)
    wh_annual_demand = all_dem.groupby('warehouse')['demand'].sum().to_dict()
    factory_total_capacity = fac_cap.groupby('factory')[['reg_capacity', 'ot_capacity']].sum().sum(axis=1).to_dict()
    # 초기 할당 (1순위)
    current_assignment = {wh_id: options[0]['factory'] for wh_id, options in wh_sourcing_options.items()}

    for iteration in range(len(selected_warehouses) + 1):
        factory_load = defaultdict(float)
        for wh_id, fac_id in current_assignment.items():
            factory_load[fac_id] += wh_annual_demand.get(wh_id, 0)
        
        factory_utilization = {fac_id: factory_load[fac_id] / factory_total_capacity.get(fac_id, 1e9) for fac_id in selected_factories}
        overloaded = {f: u for f, u in factory_utilization.items() if u > CAPACITY_THRESHOLD}
        
        if not overloaded:
            print(f"Iteration {iteration}: 모든 공장 부하율 안정화. 균등화 완료.")
            break
        
        most_overloaded_fac = max(overloaded, key=overloaded.get)
        print(f"Iteration {iteration}: '{most_overloaded_fac}' 과부하 ({overloaded[most_overloaded_fac]:.1%}). 재조정 필요.")

        wh_to_reassign, min_penalty = None, float('inf')
        for wh_id, fac_id in current_assignment.items():
            if fac_id == most_overloaded_fac and len(wh_sourcing_options[wh_id]) > 1:
                penalty = wh_sourcing_options[wh_id][1]['cost'] - wh_sourcing_options[wh_id][0]['cost']
                if penalty < min_penalty:
                    min_penalty, wh_to_reassign = penalty, wh_id
        
        if wh_to_reassign:
            new_fac = wh_sourcing_options[wh_to_reassign][1]['factory']
            print(f"  -> '{wh_to_reassign}'의 공급처를 '{current_assignment[wh_to_reassign]}'에서 '{new_fac}'(으)로 변경.")
            current_assignment[wh_to_reassign] = new_fac
        else:
            print(f"  -> '{most_overloaded_fac}'의 부하를 줄일 다른 대안이 없습니다.")
            break
    
    # --- 4. 최종 정책 수립 및 저장 ---
    print("\n[2-3] 최종 소싱 및 재고 정책 수립 중...")
    final_sourcing_policy = {}
    for wh_id in selected_warehouses:
        primary_fac_id = current_assignment[wh_id]
        primary_option = next(item for item in wh_sourcing_options[wh_id] if item["factory"] == primary_fac_id)
        secondary_option = next((opt for opt in wh_sourcing_options[wh_id] if opt['factory'] != primary_fac_id), None)
        final_sourcing_policy[wh_id] = {"primary_source": primary_option, "secondary_source": secondary_option}

    inventory_policy = defaultdict(dict)
    demand_stats = all_dem.groupby(['warehouse', 'sku'])['demand'].agg(['mean', 'std']).fillna(0)
    for (wh_id, sku_id), stats in demand_stats.iterrows():
        if wh_id not in final_sourcing_policy: continue
        lead_time = final_sourcing_policy[wh_id]['primary_source']['lead_time']
        safety_stock = Z_SCORE_95 * np.sqrt(lead_time) * stats['std']
        reorder_point = (lead_time * stats['mean']) + safety_stock
        inventory_policy[wh_id][sku_id] = {
            "safety_stock": int(np.ceil(safety_stock)),
            "reorder_point": int(np.ceil(reorder_point))
        }

    final_plan = {"sourcing_policy": final_sourcing_policy, "inventory_policy": inventory_policy}
    output_path = "tactical_plan_advanced.json"
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(final_plan, f, indent=4)
        
    print(f"\n--- 2단계 정교한 전술 계획 수립 완료 ---")
    print(f"결과가 '{output_path}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    main_stage2_planner()