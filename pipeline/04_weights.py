"""
04_weights.py
=================================================================
job_profiles_parsed.json 에서 직접 3종 데이터를 로드해
537개 직업별 5축 가중치를 Z-score + Softmax 로 산출한다.

산출물:
  job_weights.json      직업별 가중치 (메인, 채점 시 사용)
  cluster_weights.json  군집별 평균 가중치 (기존 코드 호환용)
  clusters.csv          군집 + Sil 마킹 (갱신)

실행:
  python pipeline/04_weights.py
=================================================================
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'job_profiles_parsed.json')
OUT_DIR   = os.path.join(BASE_DIR, 'data', 'processed')

# ── 파라미터 ───────────────────────────────────────────────────
K           = 8
TEMPERATURE = 0.5   # tune_temperature.py 로 최적값 확인 후 변경
RANDOM_SEED = 42

SIM_CLUSTER_IDS = {'C2', 'C4', 'C5', 'C6', 'C8'}

AXIS_ITEMS = {
    'AX1': {
        'label': '정보분석·논리',
        'items': ['정보 수집', '정보, 자료 분석', '정보 처리',
                  '정보의 의미 해석', '컴퓨터 업무',
                  '기준에 따른 정보 평가', '정보 작성, 기록'],
    },
    'AX2': {
        'label': '관찰·탐색',
        'items': ['절차, 자료, 주변환경 관찰', '사물, 행동, 사건 파악',
                  '새로운 지식의 습득, 활용', '장비, 건축물, 자재 검사'],
    },
    'AX3': {
        'label': '전략·판단',
        'items': ['의사 결정, 문제점 해결', '목표, 전략 수립',
                  '업무 계획, 우선순위 결정', '창조적 생각'],
    },
    'AX4': {
        'label': '리더십·조직',
        'items': ['부하 직원들에게 업무 안내, 지시, 동기부여',
                  '팀 구성, 협업 촉진',
                  '사람들의 업무와 활동을 조직, 편성',
                  '인사 업무', '사람들의 능력 개발, 지도'],
    },
    'AX5': {
        'label': '대인서비스',
        'items': ['대인관계 유지', '업무상 사람들을 직접 응대',
                  '사람들을 배려, 돌봄', '사람들에게 영향력 행사'],
    },
}


def softmax(vals: list, temperature: float = 0.5) -> np.ndarray:
    v = np.array(vals, dtype=float)
    v = v - v.min() + 0.01
    e = np.exp(v / temperature)
    return e / e.sum()


def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    print('=' * 60)
    print('STEP 0. JSON 데이터 로드')
    print('=' * 60)

    raw = load_json(JSON_PATH)
    jobs = raw['jobs']
    print(f"직업 수: {len(jobs)}개  |  척도: {raw['scale']}")

    # ── 3종 데이터를 하나의 DataFrame 으로 결합 ────────────────
    rows = []
    for jcd, info in jobs.items():
        r = {'job_code': jcd, 'job_name': info['name']}
        r.update(info['activities'])   # 활동중요도 41종
        r.update(info['abilities'])    # 능력 44종
        r.update(info['character'])    # 성격 16종
        rows.append(r)

    df = (pd.DataFrame(rows)
            .set_index('job_code')
            .fillna(0))

    job_names = df['job_name']
    feat_cols = [c for c in df.columns if c != 'job_name']
    df_feat   = df[feat_cols]

    print(f"결합 행렬: {df_feat.shape[0]} 직업 × {df_feat.shape[1]} 항목")

    # ── Z-score 표준화 ─────────────────────────────────────────
    print('\n' + '=' * 60)
    print('STEP 1. Z-score 표준화')
    print('=' * 60)

    X_z = pd.DataFrame(
        StandardScaler().fit_transform(df_feat.values),
        index=df_feat.index,
        columns=df_feat.columns,
    )

    # 축 항목이 실제로 존재하는지 확인
    for ax, ax_info in AXIS_ITEMS.items():
        missing = [c for c in ax_info['items'] if c not in X_z.columns]
        found   = len(ax_info['items']) - len(missing)
        print(f"  {ax}: {found}/{len(ax_info['items'])}개 항목 매핑", end='')
        if missing:
            print(f"  (미발견: {missing})", end='')
        print()

    # ── 군집화 + Silhouette ────────────────────────────────────
    print('\n' + '=' * 60)
    print('STEP 2. 군집화 + Silhouette 마킹')
    print('=' * 60)

    # 군집화는 활동중요도 41종만 사용 (기존 방식 유지)
    act_keys = list(raw['fields']['activity_items'])
    X_act    = StandardScaler().fit_transform(df_feat[act_keys].values)

    km     = KMeans(n_clusters=K, init='k-means++',
                    random_state=RANDOM_SEED, n_init=25, max_iter=300)
    labels   = km.fit_predict(X_act)
    sil_samp = silhouette_samples(X_act, labels)
    sil_all  = silhouette_score(X_act, labels)

    print(f"전체 Silhouette (k={K}): {sil_all:.4f}")
    neg_count = (sil_samp < 0).sum()
    print(f"Sil < 0 직업: {neg_count}개 → sil_reliable=False 마킹")

    cluster_ids = [f'C{l+1}' for l in labels]

    # ── 직업별 가중치 계산 ──────────────────────────────────────
    print('\n' + '=' * 60)
    print('STEP 3. 직업별 5축 가중치 (Z-score + Softmax)')
    print('=' * 60)

    ax_keys = list(AXIS_ITEMS.keys())
    records = []

    for i, jcd in enumerate(df_feat.index):
        cid          = cluster_ids[i]
        sil_val      = float(sil_samp[i])
        sil_reliable = sil_val >= 0
        is_sim       = cid in SIM_CLUSTER_IDS

        # 각 축 Z-score 평균 (축 항목 중 실제 컬럼만)
        ax_z = {}
        for ax, ax_info in AXIS_ITEMS.items():
            valid = [c for c in ax_info['items'] if c in X_z.columns]
            ax_z[ax] = float(X_z.loc[jcd, valid].mean()) if valid else 0.0

        wts = softmax(list(ax_z.values()), TEMPERATURE)

        rec = {
            'job_code':             jcd,
            'job_name':             job_names.get(jcd, ''),
            'cluster_id':           cid,
            'silhouette':           round(sil_val, 4),
            'sil_reliable':         sil_reliable,
            'is_simulation_target': is_sim,
        }
        for j, ax in enumerate(ax_keys):
            rec[ax] = round(float(wts[j]), 4)
        records.append(rec)

    df_result = pd.DataFrame(records)

    # 가중치 합 검증
    w_sum = df_result[ax_keys].sum(axis=1)
    assert w_sum.between(0.999, 1.001).all(), '가중치 합 오류!'
    print(f"가중치 합 검증: min={w_sum.min():.6f}, max={w_sum.max():.6f}  OK")

    # ── 결과 요약 출력 ─────────────────────────────────────────
    print(f"\n시뮬레이션 대상 군집 직업 수:")
    for cid in sorted(SIM_CLUSTER_IDS):
        n = (df_result['cluster_id'] == cid).sum()
        print(f"  {cid}: {n}개 직업")

    print(f"\nSil<0 직업 예시 (상위 5개):")
    for _, r in df_result[~df_result['sil_reliable']].head(5).iterrows():
        print(f"  {r['job_name']:20s}  군집:{r['cluster_id']}  Sil:{r['silhouette']:.4f}")

    # ── job_weights.json 저장 (메인) ──────────────────────────
    print('\n' + '=' * 60)
    print('STEP 4. 저장')
    print('=' * 60)

    job_weights = {}
    for _, r in df_result.iterrows():
        job_weights[r['job_code']] = {
            'job_name':             r['job_name'],
            'cluster_id':           r['cluster_id'],
            'silhouette':           r['silhouette'],
            'sil_reliable':         bool(r['sil_reliable']),
            'is_simulation_target': bool(r['is_simulation_target']),
            'weights': {ax: r[ax] for ax in ax_keys},
        }

    out_jw = os.path.join(OUT_DIR, 'job_weights.json')
    with open(out_jw, 'w', encoding='utf-8') as f:
        json.dump(job_weights, f, ensure_ascii=False, indent=2)
    print(f"job_weights.json 저장: {out_jw}  ({len(job_weights)}개 직업)")

    # ── cluster_weights.json 저장 (호환용) ────────────────────
    cluster_weights = {}
    for cid in sorted(SIM_CLUSTER_IDS):
        sub   = df_result[df_result['cluster_id'] == cid]
        avg   = {ax: float(sub[ax].mean()) for ax in ax_keys}
        total = sum(avg.values())
        avg   = {ax: round(v / total, 4) for ax, v in avg.items()}
        cluster_weights[cid] = {'n_jobs': int(len(sub)), **avg}

    out_cw = os.path.join(OUT_DIR, 'cluster_weights.json')
    with open(out_cw, 'w', encoding='utf-8') as f:
        json.dump(cluster_weights, f, ensure_ascii=False, indent=2)
    print(f"cluster_weights.json 저장: {out_cw}")

    # ── clusters.csv 갱신 ─────────────────────────────────────
    out_cl = os.path.join(OUT_DIR, 'clusters.csv')
    df_result[['job_code', 'job_name', 'cluster_id',
               'silhouette', 'sil_reliable',
               'is_simulation_target']].to_csv(
        out_cl, index=False, encoding='utf-8-sig')
    print(f"clusters.csv 저장: {out_cl}")

    print('\n완료. 채점 공식:')
    print('  최종 점수 = sum(U_AXi × W_AXi) × 100')
    print('  U: LLM 출력 (0~1)  |  W: job_weights.json 의 직업별 weights')


if __name__ == '__main__':
    main()
