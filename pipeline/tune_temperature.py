"""
tune_temperature.py
=================================================================
Softmax temperature 최적값 탐색 스크립트

평가 기준:
  1. 변별력   - 군집 간 1위 축 가중치 차이 (클수록 좋음)
  2. 집중도   - 1위 축 가중치 평균 (클수록 선명)
  3. 분산 안정 - 직업별 가중치 표준편차 평균 (너무 크면 불안정)

권장 온도 선택 기준:
  - 변별력 + 집중도 가장 높은 구간
  - 단, 1위 축이 90% 이상이면 너무 쏠림 (하한선 고려)

실행:
  python pipeline/tune_temperature.py
=================================================================
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'job_profiles_parsed.json')

TEMPERATURES   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]
K              = 8
SIM_CLUSTERS   = {'C2', 'C4', 'C5', 'C6', 'C8'}
RANDOM_SEED    = 42

AXIS_ITEMS = {
    'AX1': ['정보 수집', '정보, 자료 분析', '정보 처리',
             '정보의 의미 해석', '컴퓨터 업무',
             '기준에 따른 정보 평가', '정보 작성, 기록'],
    'AX2': ['절차, 자료, 주변환경 관찰', '사물, 행동, 사건 파악',
             '새로운 지식의 습득, 활용', '장비, 건축물, 자재 검사'],
    'AX3': ['의사 결정, 문제점 해결', '목표, 전략 수립',
             '업무 계획, 우선순위 결정', '창조적 생각'],
    'AX4': ['부하 직원들에게 업무 안내, 지시, 동기부여',
             '팀 구성, 협업 촉진',
             '사람들의 업무와 활동을 조직, 편성',
             '인사 업무', '사람들의 능력 개발, 지도'],
    'AX5': ['대인관계 유지', '업무상 사람들을 직접 응대',
             '사람들을 배려, 돌봄', '사람들에게 영향력 행사'],
}


def softmax(vals, t):
    v = np.array(vals, dtype=float)
    v = v - v.min() + 0.01
    e = np.exp(v / t)
    return e / e.sum()


def main():
    # 데이터 로드
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    jobs = raw['jobs']

    rows = []
    for jcd, info in jobs.items():
        r = {'job_code': jcd}
        r.update(info['activities'])
        r.update(info['abilities'])
        r.update(info['character'])
        rows.append(r)

    df      = pd.DataFrame(rows).set_index('job_code').fillna(0)
    feat    = df
    X_z     = pd.DataFrame(
        StandardScaler().fit_transform(feat.values),
        index=feat.index, columns=feat.columns
    )

    # 군집화
    act_keys = list(raw['fields']['activity_items'])
    X_act    = StandardScaler().fit_transform(feat[act_keys].values)
    km       = KMeans(n_clusters=K, init='k-means++',
                      random_state=RANDOM_SEED, n_init=25, max_iter=300)
    labels   = km.fit_predict(X_act)
    cluster_ids = [f'C{l+1}' for l in labels]

    ax_keys = list(AXIS_ITEMS.keys())

    # 온도별 지표 계산
    stats = []
    print(f"{'Temp':>5}  {'변별력':>8}  {'집중도(1위)':>10}  {'1위>80%비율':>11}  권장")
    print('-' * 55)

    for t in TEMPERATURES:
        all_wts = []
        for jcd in feat.index:
            ax_z = {}
            for ax, items in AXIS_ITEMS.items():
                valid = [c for c in items if c in X_z.columns]
                ax_z[ax] = float(X_z.loc[jcd, valid].mean()) if valid else 0.0
            wts = softmax(list(ax_z.values()), t)
            all_wts.append(wts)

        wt_arr  = np.array(all_wts)           # (537, 5)
        top1    = wt_arr.max(axis=1)           # 1위 가중치
        top2    = np.sort(wt_arr, axis=1)[:, -2]  # 2위 가중치

        discrim = float((top1 - top2).mean())  # 1위-2위 차이 (변별력)
        concen  = float(top1.mean())           # 1위 평균 (집중도)
        over80  = float((top1 > 0.8).mean())   # 80% 초과 비율 (쏠림)

        ok = 'O' if (discrim > 0.15 and over80 < 0.3) else ' '
        stats.append({
            'temperature': t,
            'discriminability': round(discrim, 4),
            'top1_mean':        round(concen, 4),
            'top1_over80_rate': round(over80, 4),
        })
        print(f"{t:>5.1f}  {discrim:>8.4f}  {concen:>10.4f}  {over80:>11.4f}  {ok}")

    df_stats = pd.DataFrame(stats)

    # 최적 온도 자동 추천
    # 조건: 변별력 > 0.15, 쏠림(over80) < 0.3
    candidates = df_stats[
        (df_stats['discriminability'] > 0.15) &
        (df_stats['top1_over80_rate'] < 0.3)
    ]
    if len(candidates):
        best_t = candidates.loc[candidates['discriminability'].idxmax(), 'temperature']
        print(f"\n추천 temperature: {best_t}")
        print("04_weights.py 상단 TEMPERATURE 값을 위 값으로 변경하세요.")
    else:
        print("\n단일 추천 어려움. 그래프 확인 후 직접 결정하세요.")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='#F8F9FA')
    fig.suptitle('Softmax Temperature 튜닝 - 지표별 변화',
                 fontsize=12, fontweight='bold', color='#1F4E79')

    for ax, col, title, color in zip(
        axes,
        ['discriminability', 'top1_mean', 'top1_over80_rate'],
        ['변별력 (1위-2위 차이)', '집중도 (1위 평균)', '쏠림률 (1위>80%)'],
        ['#2E75B6', '#1D9E75', '#E24B4A']
    ):
        ax.set_facecolor('#F8F9FA')
        ax.plot(df_stats['temperature'], df_stats[col],
                'o-', color=color, lw=2, ms=6)
        if len(candidates):
            ax.axvline(best_t, color='#9B59B6', ls='--', lw=1.5,
                       label=f'추천 T={best_t}')
            ax.legend(fontsize=8)
        ax.set_xlabel('Temperature', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_fig = os.path.join(BASE_DIR, 'temperature_tuning.png')
    plt.savefig(out_fig, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print(f"\n그래프 저장: {out_fig}")

    # CSV 저장
    out_csv = os.path.join(BASE_DIR, 'temperature_tuning.csv')
    df_stats.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"수치 저장: {out_csv}")


if __name__ == '__main__':
    main()
