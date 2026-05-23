"""
=================================================================
KNOW 직업사전 K-means 군집 분석 코드
=================================================================
입력: job_profiles_parsed.json

출력:
  cluster_optimization.png  최적 k 탐색 그래프
  cluster_result.png        군집화 결과 시각화
  cluster_result.csv        직업별 군집 배정 결과

설계 근거:
  - 군집화 기준: 활동중요도 41종 단독
    (능력·성격은 가중치 산출 단계에서 활용)
  - 표준화: Z-score (StandardScaler)
  - 최적 k 탐색: Elbow Method + Silhouette Score + Hopkins 통계량
  - 최종 k: 그래프 확인 후 FINAL_K 값 변경

실행 방법:
  pip install scikit-learn pandas numpy matplotlib koreanize-matplotlib
  python clustering.py
=================================================================
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'job_profiles_parsed.json')

# ── 파라미터 ───────────────────────────────────────────────────
K_MIN   = 2
K_MAX   = 10
FINAL_K = 8   # 최적 k 그래프 확인 후 필요시 변경


# ── 0. 데이터 로드 ─────────────────────────────────────────────
print("=" * 60)
print("STEP 0. 데이터 로드")
print("=" * 60)

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    raw = json.load(f)

rows = []
for jcd, info in raw['jobs'].items():
    row = {'job_cd': jcd, 'job_name': info['name']}
    row.update(info['activities'])
    rows.append(row)

df       = pd.DataFrame(rows).set_index('job_cd')
act_cols = [c for c in df.columns if c != 'job_name']
df[act_cols] = df[act_cols].fillna(df[act_cols].mean())

print(f"직업 수:   {len(df)}개")
print(f"활동 항목: {len(act_cols)}개")
print(f"결측치:    {df[act_cols].isnull().sum().sum()}개")
print(f"점수 범위: {df[act_cols].min().min():.2f} ~ {df[act_cols].max().max():.2f}")

scaler = StandardScaler()
X_std  = scaler.fit_transform(df[act_cols].values)


# ── 1. Hopkins 통계량 ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1. 군집 타당성 검증 (Hopkins 통계량)")
print("=" * 60)

def hopkins_statistic(X, m=100, seed=42):
    np.random.seed(seed)
    n, d   = X.shape
    idx    = np.random.choice(n, m, replace=False)
    X_r    = np.random.uniform(X.min(0), X.max(0), (m, d))
    nbrs   = NearestNeighbors(n_neighbors=2).fit(X)
    u_dist = nbrs.kneighbors(X_r, n_neighbors=1)[0].flatten()
    w_dist = nbrs.kneighbors(X[idx], n_neighbors=2)[0][:, 1].flatten()
    return u_dist.sum() / (u_dist.sum() + w_dist.sum())

H = hopkins_statistic(X_std)
if   H > 0.7: verdict = "강한 군집 경향 (군집 분석 타당)"
elif H > 0.5: verdict = "약한 군집 경향 (해석 주의)"
else:         verdict = "군집 경향 미약"
print(f"Hopkins 통계량: {H:.4f}  ({verdict})")
print("기준: 0.5=랜덤, 1.0에 가까울수록 군집 명확")


# ── 2. 최적 k 탐색 ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2. 최적 k 탐색 (Elbow + Silhouette)")
print("=" * 60)

k_range    = range(K_MIN, K_MAX + 1)
inertias   = []
sil_scores = []

for k in k_range:
    km  = KMeans(n_clusters=k, init='k-means++',
                 random_state=42, n_init=25, max_iter=300)
    lbs = km.fit_predict(X_std)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_std, lbs))

best_k   = list(k_range)[sil_scores.index(max(sil_scores))]
sil2nd_k = sorted(k_range, key=lambda k: -sil_scores[k - K_MIN])[1]
delta2   = np.diff(-np.diff(inertias))
elbow_k  = list(k_range)[int(np.argmax(delta2)) + 1]

print(f"\n{'k':>4}  {'Inertia':>12}  {'Silhouette':>11}  비고")
print("-" * 48)
for k in k_range:
    i    = k - K_MIN
    note = ""
    if k == best_k:
        note = "Silhouette 1위"
    elif k == sil2nd_k:
        note = "Silhouette 2위"
    if k == elbow_k:
        note = (note + " / " if note else "") + "Elbow"
    print(f"{k:>4}  {inertias[i]:>12.1f}  {sil_scores[i]:>11.4f}  {note}")

print(f"\nSilhouette 최적: k={best_k}  ({max(sil_scores):.4f})")
print(f"Elbow 자동탐지:  k={elbow_k}")
print(f"최종 채택:       FINAL_K={FINAL_K}  (코드 상단에서 변경 가능)")


# ── 3. 최적 k 탐색 시각화 ──────────────────────────────────────
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5), facecolor='#F8F9FA')
fig1.suptitle('최적 군집 수(k) 탐색 결과\n'
              '(Hopkins 통계량 + Elbow Method + Silhouette Score)',
              fontsize=13, fontweight='bold', y=1.01, color='#1F4E79')

ax = axes1[0]; ax.set_facecolor('#F8F9FA')
ax.plot(list(k_range), inertias, 'o-', color='#2E75B6', lw=2, ms=7)
ax.axvline(FINAL_K, color='#E24B4A', ls='--', lw=2, label=f'채택 k={FINAL_K}')
ax.axvline(best_k,  color='#1D9E75', ls=':', lw=1.5, label=f'Sil 최적 k={best_k}')
ax.set_xlabel('k', fontsize=12); ax.set_ylabel('Inertia', fontsize=12)
ax.set_title('Elbow Method', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xticks(list(k_range))

ax = axes1[1]; ax.set_facecolor('#F8F9FA')
bar_clrs = ['#E24B4A' if k == FINAL_K
            else ('#1D9E75' if k == best_k else '#9EC8E8')
            for k in k_range]
ax.bar(list(k_range), sil_scores, color=bar_clrs,
       edgecolor='white', lw=0.5, width=0.7)
for k, s in zip(k_range, sil_scores):
    ax.text(k, s + 0.003, f'{s:.3f}', ha='center', va='bottom', fontsize=8)
ax.set_xlabel('k', fontsize=12); ax.set_ylabel('Silhouette Score', fontsize=12)
ax.set_title('Silhouette Score\n(초록=통계최적, 빨강=채택)',
             fontsize=12, fontweight='bold')
ax.set_xticks(list(k_range)); ax.grid(True, alpha=0.3, axis='y')

ax = axes1[2]; ax.set_facecolor('#F8F9FA')
ax.barh([''], [H], color='#1D9E75', height=0.4)
ax.barh([''], [1 - H], left=[H], color='#DDDDDD', height=0.4)
ax.axvline(0.5, color='#E24B4A', ls='--', lw=1.5, label='랜덤 기준 (0.5)')
ax.axvline(0.7, color='#1D9E75', ls=':',  lw=1.5, label='강한 군집 (0.7)')
ax.text(H + 0.01, 0, f'H={H:.4f}', va='center',
        fontsize=11, fontweight='bold', color='#1F4E79')
ax.set_xlim(0, 1.05); ax.set_yticks([])
ax.set_title('Hopkins 통계량\n(군집 분석 타당성)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

plt.tight_layout()
out1 = os.path.join(BASE_DIR, 'data', 'processed', 'cluster_optimization.png')
plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print(f"\n최적 k 탐색 그래프 저장: {out1}")


# ── 4. 최종 군집화 ──────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 3. k={FINAL_K} 군집화")
print("=" * 60)

km_final = KMeans(n_clusters=FINAL_K, init='k-means++',
                  random_state=42, n_init=25, max_iter=300)
labels   = km_final.fit_predict(X_std)
sil_samp = silhouette_samples(X_std, labels)
df['cluster']       = labels
df['cluster_label'] = [f'C{l + 1}' for l in labels]
df['silhouette']    = sil_samp

cl_size = df.groupby('cluster').size()
cl_sil  = df.groupby('cluster')['silhouette'].mean()

print(f"\n전체 Silhouette Score (k={FINAL_K}): "
      f"{silhouette_score(X_std, labels):.4f}")
print(f"\n{'군집':>5}  {'직업수':>6}  {'Sil평균':>8}  {'Sil<0비율':>9}  직업 예시")
print("-" * 72)
for ci in range(FINAL_K):
    mask    = labels == ci
    neg_pct = (sil_samp[mask] < 0).mean() * 100
    samples = df.loc[mask, 'job_name'].head(3).tolist()
    print(f"  C{ci+1:>2}  {cl_size[ci]:>6}개  {cl_sil[ci]:>+8.4f}  "
          f"{neg_pct:>7.1f}%  {', '.join(samples)}")


# ── 5. 군집화 결과 시각화 ──────────────────────────────────────
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_std)
ve    = pca.explained_variance_ratio_ * 100

palette = ['#2E75B6', '#E24B4A', '#1D9E75', '#9B59B6',
           '#E67E22', '#16A085', '#8E44AD', '#E74C3C']
colors  = [palette[ci % len(palette)] for ci in range(FINAL_K)]

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7), facecolor='#F8F9FA')
fig2.suptitle(f'k={FINAL_K} 군집화 결과 시각화\n'
              f'(KNOW 직업사전 활동중요도 41종, n={len(df)}개 직업)',
              fontsize=13, fontweight='bold', y=1.01, color='#1F4E79')

ax = axes2[0]; ax.set_facecolor('#F8F9FA')
for ci in range(FINAL_K):
    mask = labels == ci
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors[ci], alpha=0.55, s=25, edgecolors='none',
               label=f'C{ci+1} (n={cl_size[ci]})')
cp = pca.transform(km_final.cluster_centers_)
for ci in range(FINAL_K):
    ax.scatter(cp[ci, 0], cp[ci, 1], c=colors[ci],
               marker='*', s=300, edgecolors='white', lw=1.2, zorder=5)
    ax.annotate(f'C{ci+1}', cp[ci],
                xytext=(4, 5), textcoords='offset points',
                fontsize=9, fontweight='bold', color=colors[ci])
ax.set_xlabel(f'PC1 ({ve[0]:.1f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({ve[1]:.1f}%)', fontsize=11)
ax.set_title('PCA 2D 산점도', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.2)

ax2    = axes2[1]; ax2.set_facecolor('#F8F9FA')
x_pos  = np.arange(FINAL_K)
w      = 0.38
sil_v  = [cl_sil[ci] for ci in range(FINAL_K)]
cnt_v  = [cl_size[ci] for ci in range(FINAL_K)]
ax2_r  = ax2.twinx()
ax2.bar(x_pos - w/2, sil_v, w,
        color=colors, alpha=0.85, edgecolor='white', lw=0.5)
ax2_r.bar(x_pos + w/2, cnt_v, w,
          color=colors, alpha=0.40, edgecolor='white', lw=0.5)
ax2.axhline(0, color='#E24B4A', ls='--', lw=1, alpha=0.7)
ax2.set_xlabel('군집', fontsize=11)
ax2.set_ylabel('Silhouette 평균', fontsize=11)
ax2_r.set_ylabel('직업 수', fontsize=11)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'C{i+1}' for i in range(FINAL_K)], fontsize=10)
ax2.set_title('군집별 Silhouette & 직업 수', fontsize=12, fontweight='bold')
for i, (s, c) in enumerate(zip(sil_v, cnt_v)):
    ax2.text(i - w/2, s + 0.003, f'{s:.3f}', ha='center', fontsize=8)
    ax2_r.text(i + w/2, c + 0.5, str(c), ha='center', fontsize=8)
ax2.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
out2 = os.path.join(BASE_DIR, 'data', 'processed', 'cluster_result.png')
plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print(f"군집화 결과 시각화 저장: {out2}")


# ── 6. 결과 CSV 저장 ───────────────────────────────────────────
df_sorted = df.sort_values(by=['cluster', 'silhouette'],
                           ascending=[True, False])
out3 = os.path.join(BASE_DIR, 'data', 'processed', 'cluster_result.csv')
df_sorted[['job_name', 'cluster_label', 'silhouette']]\
    .to_csv(out3, encoding='utf-8-sig')
print(f"군집 결과 CSV 저장: {out3}")


# ── 7. 군집별 활동 특성 출력 ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4. 군집별 활동 특성 (상위 5개)")
print("=" * 60)

cl_mean = df.groupby('cluster')[act_cols].mean()

for ci in range(FINAL_K):
    cnt     = (labels == ci).sum()
    top5    = cl_mean.loc[ci].sort_values(ascending=False).head(5)
    samples = df[df['cluster'] == ci]['job_name'].head(4).tolist()
    act_str = ', '.join(f"{a}({v:.2f})" for a, v in top5.items())
    print(f"\nC{ci+1} (n={cnt}개)")
    print(f"  상위 활동: {act_str}")
    print(f"  직업 예시: {', '.join(samples)}")
