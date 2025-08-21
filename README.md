SmartPhone Global Supply-Chain Challenge 솔루션 실행 가이드
1. 프로젝트 개요
본 프로젝트는 스마트폰 공급망 최적화 문제를 해결하기 위해 3단계 계층적 계획(Hierarchical Planning) 접근법을 사용한다. 각 단계는 독립적인 Python 스크립트로 구성되어 있으며, 이전 단계의 결과물을 입력으로 사용하여 순차적으로 실행해야 한다.

1단계 (전략): 최적의 공장 및 창고 입지를 선정한다 (location.py).

2단계 (전술): 선정된 입지를 바탕으로 안정적인 운영을 위한 소싱 및 재고 규칙을 수립한다 (strategy.py).

3단계 (운영): 수립된 규칙에 따라 일일 시뮬레이션을 실행하여 최종 생산/물류 계획을 도출한다 (operation.py).

2. 사전 준비 사항
2.1 디렉토리 구조

코드를 실행하기 전, 프로젝트 디렉토리는 다음과 같은 구조를 가져야 한다.

```
.
├── data/                  # 모든 데이터 파일 (.csv, .db)이 위치하는 폴더
│   ├── site_candidates.csv
│   ├── demand_train.db
│   └── ... (기타 모든 데이터 파일)
├── main/                  # 모든 Python 실행 스크립트가 위치하는 폴더
│   ├── location.py        # 1단계: 입지 선정
│   ├── strategy.py        # 2단계: 전술 계획
│   └── operation.py       # 3단계: 운영 시뮬레이션
└── results/               # 각 단계의 결과물이 저장될 폴더 (자동 생성)
    ├── selected_factories.csv
    ├── selected_warehouses.csv
    ├── tactical_plan_advanced.json
    └── plan_submission_template.db
```
3. 실행 순서
3.1 1단계: 최적 입지 선정

전체 기간의 총수요와 운송비를 고려하여 최적의 공장 5곳과 창고 20곳을 선정한다.

입력: ```data/``` 폴더의 원본 데이터

출력 (```results/``` 폴더):

```selected_factories.csv```: 선정된 공장 목록

```selected_warehouses.csv```: 선정된 창고 목록

3.2 2단계: 전술 계획 수립

1단계에서 선정된 시설을 바탕으로, ABC-XYZ 분석과 생산 능력 균등화를 통해 주/예비 공급처 및 SKU별 재고 정책(SS, s, S)을 수립한다.

입력: ```results/selected_*.csv``` 파일 및 ```data/``` 폴더의 원본 데이터

출력 (```results/``` 폴더):

```tactical_plan_advanced.json```: 3단계 시뮬레이션이 사용할 소싱 및 재고 규칙

3.3 3단계: 운영 시뮬레이션

2단계에서 수립된 전술 계획에 따라, 2018년부터 2024년까지의 일일 운영을 시뮬레이션하여 최종 제출 파일을 생성한다.

입력: ```results/tactical_plan_advanced.json``` 파일 및 ```data/``` 폴더의 원본 데이터

출력 (```results/``` 폴더):

```plan_submission_template.db```: 최종 제출용 생산 및 물류 계획 DB 파일

전체 기간 및 주별 Fill-Rate

국가별, 제품군별 등 상세 분석 결과

