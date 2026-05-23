"""
=================================================================
KNOW 직업사전 계층적 요인분석 코드
=================================================================
입력: job_profiles_parsed.json

분석 구조:
  1차 FA: 활동중요도 41종 전체 -> Kaiser Rule -> 5개 요인
  2차 FA: F1 (항목 28개로 너무 큼) -> Kaiser Rule -> 4개 하위 요인
  최종 축: 의미 해석 + 제외 기준 적용 -> 사고방식 축 5개 결정

출력:
  factor_fig1_scree.png      스크리 플롯 (1차 + 2차)
  factor_fig2_loadings1.png  1차 FA 부하량 히트맵
  factor_fig3_loadings2.png  2차 FA 부하량 히트맵
  factor_fig4_axes.png       최종 축 요약
  factor_axes_table.csv      축별 소속 항목 테이블

실행 방법:
  pip install scikit-learn pandas numpy matplotlib koreanize-matplotlib
  python factor_analysis.py
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
from sklearn.decomposition import FactorAnalysis
import warnings
warnings.filterwarnings('ignore')

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR  = '/Users/imhyeongjun/Desktop/Coding/Project'
JSON_PATH = os.path.join(BASE_DIR,'data','processed','job_profiles_parsed.json')

# ── 파라미터 ───────────────────────────────────────────────────
THRESHOLD = 0.4   # 요인 부하량 유의 기준


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
X_std    = StandardScaler().fit_transform(df[act_cols].values)
n, p     = X_std.shape
print(f"직업 수: {n}개  |  활동 항목: {p}개")


# ── 1. 1차 FA ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1. 1차 요인분析 (활동중요도 41종 전체)")
print("=" * 60)

corr1 = np.corrcoef(X_std.T)
ev1   = np.linalg.eigvalsh(corr1)[::-1]
ev1   = ev1[ev1 > 0]
n_f1  = int((ev1 > 1).sum())
cv1   = np.cumsum(ev1 / p * 100)

print(f"\nKaiser Rule (고유값>1) 채택 요인 수: {n_f1}개")
print(f"\n{'요인':>6}  {'고유값':>10}  {'분산%':>8}  {'누적%':>8}")
print("-" * 40)
for i in range(min(10, len(ev1))):
    mark = "  <- 채택" if ev1[i] > 1 else ""
    print(f"  F{i+1:>2d}    {ev1[i]:>8.3f}  "
          f"{ev1[i]/p*100:>7.2f}%  {cv1[i]:>7.2f}%{mark}")

fa1 = FactorAnalysis(n_components=n_f1, rotation='varimax', random_state=42)
fa1.fit(X_std)
L1  = pd.DataFrame(fa1.components_.T, index=act_cols,
                   columns=[f'F{i+1}' for i in range(n_f1)])

print(f"\n1차 FA 결과 (|부하량| >= {THRESHOLD} 기준 항목 수):")
f_groups = {}
for col in L1.columns:
    items = L1[L1[col].abs() >= THRESHOLD].index.tolist()
    f_groups[col] = items
    sq   = (L1[col] ** 2).sum()
    print(f"  {col}: {len(items)}개 항목  (분산 설명 {sq/p*100:.1f}%)")

f1_items = f_groups['F1']
print(f"\nF1이 {len(f1_items)}개 항목 포함 -> 2차 FA 필요")


# ── 2. 2차 FA (F1 세분화) ──────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 2. 2차 요인분析 (F1 소속 {len(f1_items)}개 항목)")
print("=" * 60)

X_f1  = StandardScaler().fit_transform(df[f1_items].values)
p2    = len(f1_items)
corr2 = np.corrcoef(X_f1.T)
ev2   = np.linalg.eigvalsh(corr2)[::-1]
ev2   = ev2[ev2 > 0]
n_sf  = int((ev2 > 1).sum())
cv2   = np.cumsum(ev2 / p2 * 100)

print(f"\nKaiser Rule 하위 요인 수: {n_sf}개")
print(f"\n{'요인':>7}  {'고유값':>10}  {'분산%':>8}  {'누적%':>8}")
print("-" * 42)
for i in range(min(8, len(ev2))):
    mark = "  <- 채택" if ev2[i] > 1 else ""
    print(f"  SF{i+1:>2d}    {ev2[i]:>8.3f}  "
          f"{ev2[i]/p2*100:>7.2f}%  {cv2[i]:>7.2f}%{mark}")

fa2 = FactorAnalysis(n_components=n_sf, rotation='varimax', random_state=42)
fa2.fit(X_f1)
L2  = pd.DataFrame(fa2.components_.T, index=f1_items,
                   columns=[f'SF{i+1}' for i in range(n_sf)])

print(f"\n2차 FA 결과 (|부하량| >= {THRESHOLD} 기준):")
sf_groups = {}
for col in L2.columns:
    items = [(it, L2.loc[it, col])
             for it in L2[L2[col].abs() >= THRESHOLD].index]
    items.sort(key=lambda x: -abs(x[1]))
    sf_groups[col] = [it for it, _ in items]
    print(f"\n  {col} ({len(items)}개 항목):")
    for it, v in items:
        print(f"    {v:+.3f}  {it}")


# ── 3. 최종 사고방식 축 결정 ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3. 최종 사고방식 축 결정")
print("=" * 60)

# 제외 기준:
#   F2  기계·현장 조작 (9개)   -> 텍스트 미션 불가
#   F5  유의미 항목 없음        -> 정보 부족
#   SF3 신체+사무 혼합 (4개)   -> 이질적 구조
#   SF4 항목 2개뿐             -> 신뢰도 낮음

FINAL_AXES = {
    'AX1_정보분석·논리': {
        'source': 'SF1',
        'items':  ['정보 수집', '정보, 자료 분석', '정보 처리',
                   '정보의 의미 해석', '컴퓨터 업무',
                   '기준에 따른 정보 평가', '정보 작성, 기록'],
        'desc':   '데이터를 수집하고 논리적으로 분석하는 사고',
        'color':  '#2E75B6',
    },
    'AX2_관찰·탐색': {
        'source': 'SF1',
        'items':  ['절차, 자료, 주변환경 관찰', '사물, 행동, 사건 파악',
                   '새로운 지식의 습득, 활용'],
        'desc':   '현상을 주의깊게 관찰하고 탐색하는 사고',
        'color':  '#E67E22',
    },
    'AX3_전략·판단': {
        'source': 'SF1+SF2',
        'items':  ['의사 결정, 문제점 해결', '목표, 전략 수립',
                   '업무 계획, 우선순위 결정', '창조적 생각'],
        'desc':   '상황을 판단하고 전략적으로 결정하는 사고',
        'color':  '#9B59B6',
    },
    'AX4_리더십·조직': {
        'source': 'SF2+F4',
        'items':  ['부하 직원들에게 업무 안내, 지시, 동기부여',
                   '사람들의 업무와 활동을 조직, 편성',
                   '팀 구성, 협업 촉진', '인사 업무',
                   '사람들의 능력 개발, 지도'],
        'desc':   '사람과 조직을 이끄는 사고',
        'color':  '#16A085',
    },
    'AX5_대인서비스': {
        'source': 'F3',
        'items':  ['대인관계 유지', '업무상 사람들을 직접 응대',
                   '사람들을 배려, 돌봄', '사람들에게 영향력 행사'],
        'desc':   '상대를 공감하고 관계를 맺는 사고',
        'color':  '#1D9E75',
    },
}

print("\n최종 확정 사고방식 축:")
print(f"\n{'축':20s}  {'출처':10s}  {'항목 수':>6}  설명")
print("-" * 72)
for ax_key, ax_info in FINAL_AXES.items():
    nm = ax_key.split('_')[1]
    print(f"  {nm:18s}  {ax_info['source']:10s}  "
          f"{len(ax_info['items']):>6}개  {ax_info['desc']}")

print("\n제외 요인:")
print("  F2  (기계·현장 조작, 9개)  텍스트 미션 불가")
print("  F5  (유의미 항목 없음)     정보 부족")
print("  SF3 (신체+사무 혼합, 4개)  이질적 구조")
print("  SF4 (사물 관찰, 2개)       항목 수 부족")


# ── 4. 시각화 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4. 시각화")
print("=" * 60)

# 그림 1: 스크리 플롯
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#F8F9FA')
fig1.suptitle('계층적 요인분析 - 두 단계 스크리 플롯',
              fontsize=13, fontweight='bold', y=1.01, color='#1F4E79')

n1 = min(15, len(ev1))
ax1.set_facecolor('#F8F9FA')
ax1.plot(range(1, n1+1), ev1[:n1], 'o-', color='#2E75B6', lw=2, ms=7)
ax1.axhline(1, color='#E24B4A', ls='--', lw=1.5, label='고유값=1 (Kaiser)')
ax1.axvspan(0.5, n_f1+0.5, alpha=0.08, color='#2E75B6')
for i in range(n_f1):
    ax1.scatter(i+1, ev1[i], color='#E24B4A', s=90, zorder=4)
    pct = ev1[i]/p*100
    ax1.annotate(f'F{i+1}\n{pct:.1f}%', (i+1, ev1[i]),
                 xytext=(0, 12), textcoords='offset points',
                 ha='center', fontsize=8, color='#2E75B6', fontweight='bold')
ax1.set_xlabel('요인 번호', fontsize=11)
ax1.set_ylabel('고유값 (Eigenvalue)', fontsize=11)
ax1.set_title(f'1차 요인분析\n활동중요도 41종 전체 (n=537 직업)',
              fontsize=11, fontweight='bold')
ax1.legend(fontsize=9); ax1.set_xticks(range(1, n1+1)); ax1.grid(True, alpha=0.25)
ax1.text(0.97, 0.97, f'채택 요인: {n_f1}개\n누적 분산: {cv1[n_f1-1]:.1f}%',
         transform=ax1.transAxes, fontsize=9.5, va='top', ha='right',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#EEF4FF',
                   edgecolor='#2E75B6', lw=0.8))

n2 = min(10, len(ev2))
ax2.set_facecolor('#F8F9FA')
ax2.plot(range(1, n2+1), ev2[:n2], 'o-', color='#E24B4A', lw=2, ms=7)
ax2.axhline(1, color='#E24B4A', ls='--', lw=1.5, label='고유값=1 (Kaiser)')
ax2.axvspan(0.5, n_sf+0.5, alpha=0.08, color='#E24B4A')
for i in range(n_sf):
    ax2.scatter(i+1, ev2[i], color='#9B59B6', s=90, zorder=4)
    pct = ev2[i]/p2*100
    ax2.annotate(f'SF{i+1}\n{pct:.1f}%', (i+1, ev2[i]),
                 xytext=(0, 12), textcoords='offset points',
                 ha='center', fontsize=8, color='#9B59B6', fontweight='bold')
ax2.set_xlabel('하위 요인 번호', fontsize=11)
ax2.set_ylabel('고유값 (Eigenvalue)', fontsize=11)
ax2.set_title(f'2차 요인분析\nF1 소속 {p2}개 항목 세분화',
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=9); ax2.set_xticks(range(1, n2+1)); ax2.grid(True, alpha=0.25)
ax2.text(0.97, 0.97, f'채택 하위 요인: {n_sf}개\n누적 분산: {cv2[n_sf-1]:.1f}%',
         transform=ax2.transAxes, fontsize=9.5, va='top', ha='right',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEEEE',
                   edgecolor='#E24B4A', lw=0.8))

plt.tight_layout()
out1 = os.path.join(BASE_DIR,'data','processed','fa_fig1_scree.png'
plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print(f"그림1 저장: {out1}")

# 그림 2: 1차 FA 부하량 히트맵
fig2, ax = plt.subplots(figsize=(10, 14), facecolor='#F8F9FA')
ax.set_facecolor('#F8F9FA')
data2 = L1.values
im2   = ax.imshow(data2, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im2, ax=ax, shrink=0.4, label='요인 부하량')
ax.set_xticks(range(n_f1))
ax.set_xticklabels([f'F{i+1}' for i in range(n_f1)], fontsize=11)
ax.set_yticks(range(p))
ax.set_yticklabels(act_cols, fontsize=8.5)
for i in range(p):
    for j in range(n_f1):
        val = data2[i, j]
        if abs(val) >= THRESHOLD:
            tc = 'white' if abs(val) > 0.7 else '#222'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7.5, color=tc, fontweight='bold')
ax.set_title(f'1차 요인분析 - 부하량 행렬 (Varimax 회전)\n'
             f'강조 = |부하량| >= {THRESHOLD}',
             fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
out2 = os.path.join(BASE_DIR,'data','processed','fa_fig2_loadings1.png'
plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print(f"그림2 저장: {out2}")

# 그림 3: 2차 FA 부하량 히트맵
fig3, ax3 = plt.subplots(figsize=(9, 10), facecolor='#F8F9FA')
ax3.set_facecolor('#F8F9FA')
data3 = L2.values
im3   = ax3.imshow(data3, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im3, ax=ax3, shrink=0.5, label='요인 부하량')
sf_xlabels = {
    'SF1': 'SF1\n정보처리·분析',
    'SF2': 'SF2\n리더십·조직',
    'SF3': 'SF3\n신체+사무 (제외)',
    'SF4': 'SF4\n사물판단 (제외)',
}
ax3.set_xticks(range(n_sf))
ax3.set_xticklabels([sf_xlabels.get(f'SF{i+1}', f'SF{i+1}')
                     for i in range(n_sf)], fontsize=9.5)
ax3.set_yticks(range(len(f1_items)))
ax3.set_yticklabels([it[:22] for it in f1_items], fontsize=8.5)
for i in range(len(f1_items)):
    for j in range(n_sf):
        val = data3[i, j]
        if abs(val) >= THRESHOLD:
            tc = 'white' if abs(val) > 0.7 else '#222'
            ax3.text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=8, color=tc, fontweight='bold')
ax3.set_title(f'2차 요인분析 - F1 내부 세분화 부하량 행렬\n'
              f'(F1 소속 {len(f1_items)}개 항목 재분析)',
              fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
out3 = os.path.join(BASE_DIR,'data','processed','fa_fig3_loadings2.png'
plt.savefig(out3, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print(f"그림3 저장: {out3}")

# 그림 4: 최종 축 요약
fig4, ax4 = plt.subplots(figsize=(16, 7), facecolor='#F8F9FA')
ax4.axis('off'); ax4.set_facecolor('#F8F9FA')
ax_list = list(FINAL_AXES.items())
n_axes  = len(ax_list)
box_w   = 0.17
box_h   = 0.55
gap     = (1.0 - n_axes * box_w) / (n_axes + 1)
for idx, (ax_key, ax_info) in enumerate(ax_list):
    x  = gap * (idx + 1) + box_w * idx
    y  = 0.22
    nm = ax_key.split('_')[1]
    c  = ax_info['color']
    ax4.add_patch(plt.Rectangle((x, y), box_w, box_h,
                                transform=ax4.transAxes,
                                facecolor=c, edgecolor='white',
                                lw=1.5, alpha=0.88))
    ax4.text(x + box_w/2, y + box_h - 0.04,
             f"{ax_key.split('_')[0]}\n{nm}",
             transform=ax4.transAxes, ha='center', va='top',
             fontsize=10, color='white', fontweight='bold', linespacing=1.4)
    ax4.text(x + box_w/2, y + 0.04,
             '\n'.join(f"· {it[:14]}" for it in ax_info['items'][:5]),
             transform=ax4.transAxes, ha='center', va='bottom',
             fontsize=7.5, color='white', linespacing=1.4)
    ax4.text(x + box_w/2, y - 0.06,
             f"출처: {ax_info['source']}",
             transform=ax4.transAxes, ha='center', va='top',
             fontsize=8.5, color=c, fontweight='bold')
ax4.set_title('계층적 요인분析 결과 - 최종 사고방식 축 5개',
              fontsize=13, fontweight='bold', pad=15, color='#1F4E79')
ax4.text(0.5, 0.09,
         '1차 FA: F1(49.9%) 너무 큼 -> F2(제외), F3(AX5), F4(AX4), F5(제외)   |'
         '   2차 FA: SF1(AX1·AX2), SF2(AX3·AX4), SF3·SF4 제외',
         transform=ax4.transAxes, ha='center', va='center',
         fontsize=9, color='#555')
plt.tight_layout()
out4 = os.path.join(BASE_DIR,'data','processed','fa_fig4_axes.png'
plt.savefig(out4, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print(f"그림4 저장: {out4}")


# ── 5. 수치 테이블 저장 ────────────────────────────────────────
records = []
for ax_key, ax_info in FINAL_AXES.items():
    for item in ax_info['items']:
        records.append({
            '사고방식 축':    ax_key.split('_')[1],
            '출처 요인':      ax_info['source'],
            '소속 활동 항목': item,
            '설명':           ax_info['desc'],
        })
out5 = os.path.join(BASE_DIR,'data','processed','factor_axes_table.csv'
pd.DataFrame(records).to_csv(out5, encoding='utf-8-sig', index=False)
print(f"수치 테이블 저장: {out5}")


# ── 최종 요약 출력 ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("최종 결과 요약")
print("=" * 60)
print(f"\n1차 FA: 41종 -> {n_f1}개 요인  (누적 분산 {cv1[n_f1-1]:.1f}%)")
print(f"2차 FA: F1 {p2}개 항목 -> {n_sf}개 하위 요인  "
      f"(누적 분산 {cv2[n_sf-1]:.1f}%)")
print(f"\n최종 사고방식 축: {len(FINAL_AXES)}개")
for ax_key, ax_info in FINAL_AXES.items():
    nm = ax_key.split('_')[1]
    print(f"  {ax_key.split('_')[0]} {nm:14s}  "
          f"출처: {ax_info['source']:10s}  항목 {len(ax_info['items'])}개")
print(f"\n제외 요인: F2(기계현장), F5(항목없음), SF3(혼합), SF4(항목 2개)")
