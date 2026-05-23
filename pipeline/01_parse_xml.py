# 실행 전 BASE_DIR 을 job_data_raw 폴더 경로로 수정하세요
"""
=================================================================
KNOW 직업사전 XML → job_profiles_parsed.json 파싱 코드
=================================================================
입력: job_data_raw.zip (KNOW 직업사전 XML)

출력:
  job_profiles_parsed.json  - 직업별 점수 데이터 (JSON)

포함 데이터:
  dtlGb_2.xml  직업 기본 정보 (직업명, 직업 설명, 수행 직무)
  dtlGb_5.xml  능력 44종 (1~5점 Cmpr 척도)
  dtlGb_6.xml  성격 16종 (1~5점 Cmpr 척도)
  dtlGb_7.xml  활동중요도 41종 (1~5점 Cmpr 척도)

실행 방법:
  python job_data_parsing.py
  (표준 라이브러리만 사용, 추가 패키지 불필요)
=================================================================
"""

import zipfile
import xml.etree.ElementTree as ET
import json
from datetime import datetime

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR = '/Users/imhyeongjun/Desktop/Coding/Project'
ZIP_PATH = f'{BASE_DIR}/job_data_raw.zip'
OUT_PATH = f'{BASE_DIR}/job_profiles_parsed.json'


def parse_zip(zip_path):
    jobs           = {}
    ability_items  = {}
    character_items = {}
    activity_items = {}

    with zipfile.ZipFile(zip_path) as z:
        all_files = [n for n in z.namelist() if '__MACOSX' not in n]

        # dtlGb_2: 직업 기본 정보
        print("dtlGb_2 파싱 중 (직업 기본 정보)...")
        for path in [n for n in all_files if 'dtlGb_2.xml' in n]:
            try:
                root = ET.fromstring(
                    z.read(path).decode('utf-8', errors='replace'))
                jcd  = root.findtext('jobCd', '').strip()
                if not jcd:
                    continue
                name = (root.findtext('jobSmclNm', '')
                        or root.findtext('jobNm', '')).strip()
                exec_job = root.findtext('execJob', '').strip()
                exec_job_lines = [
                    line.lstrip('- ').strip()
                    for line in exec_job.replace('\r\n', '\n').split('\n')
                    if line.strip().lstrip('- ').strip()
                ]
                jobs[jcd] = {
                    'name':            name,
                    'category_large':  root.findtext('jobLrclNm', '').strip(),
                    'category_medium': root.findtext('jobMdclNm', '').strip(),
                    'summary':         root.findtext('jobSum', '').strip(),
                    'exec_job':        exec_job_lines,
                    'abilities':       {},
                    'character':       {},
                    'activities':      {},
                }
            except Exception:
                continue

        # dtlGb_5: 능력 44종
        print("dtlGb_5 파싱 중 (능력 44종)...")
        for path in [n for n in all_files if 'dtlGb_5.xml' in n]:
            try:
                root = ET.fromstring(
                    z.read(path).decode('utf-8', errors='replace'))
                jcd  = root.findtext('jobCd', '').strip()
                if not jcd:
                    continue
                d = {}
                for item in root.findall('jobAbilCmpr'):
                    nm = item.findtext('jobAblNmCmpr', '').strip()
                    sc = item.findtext('jobAblStatusCmpr', '').strip()
                    if nm and sc:
                        try:
                            d[nm] = float(sc)
                        except ValueError:
                            pass
                ability_items[jcd] = d
            except Exception:
                continue

        # dtlGb_6: 성격 16종
        print("dtlGb_6 파싱 중 (성격 16종)...")
        for path in [n for n in all_files if 'dtlGb_6.xml' in n]:
            try:
                root = ET.fromstring(
                    z.read(path).decode('utf-8', errors='replace'))
                jcd  = root.findtext('jobCd', '').strip()
                if not jcd:
                    continue
                d = {}
                for item in root.findall('jobChrCmpr'):
                    nm = item.findtext('jobChrNmCmpr', '').strip()
                    sc = item.findtext('jobChrStatusCmpr', '').strip()
                    if nm and sc:
                        try:
                            d[nm] = float(sc)
                        except ValueError:
                            pass
                character_items[jcd] = d
            except Exception:
                continue

        # dtlGb_7: 활동중요도 41종
        print("dtlGb_7 파싱 중 (활동중요도 41종)...")
        for path in [n for n in all_files if 'dtlGb_7.xml' in n]:
            try:
                root = ET.fromstring(
                    z.read(path).decode('utf-8', errors='replace'))
                jcd  = root.findtext('jobCd', '').strip()
                if not jcd:
                    continue
                d = {}
                for item in root.findall('jobActvImprtncCmpr'):
                    nm = item.findtext('jobActvImprtncNmCmpr', '').strip()
                    sc = item.findtext('jobActvImprtncStatusCmpr', '').strip()
                    if nm and sc:
                        try:
                            d[nm] = float(sc)
                        except ValueError:
                            pass
                activity_items[jcd] = d
            except Exception:
                continue

    for jcd in jobs:
        jobs[jcd]['abilities']  = ability_items.get(jcd, {})
        jobs[jcd]['character']  = character_items.get(jcd, {})
        jobs[jcd]['activities'] = activity_items.get(jcd, {})

    return jobs


def build_json(jobs):
    all_abl = sorted({k for j in jobs.values() for k in j['abilities']})
    all_chr = sorted({k for j in jobs.values() for k in j['character']})
    all_act = sorted({k for j in jobs.values() for k in j['activities']})

    result = {
        "schema_version": "2.0",
        "generated_at":   datetime.today().strftime('%Y-%m-%d'),
        "source":         "KNOW 직업사전 raw zip / dtlGb_2 + dtlGb_5 + dtlGb_6 + dtlGb_7",
        "total_jobs":     len(jobs),
        "scale":          "1~5점 (KNOW 직업사전 Cmpr 비교 점수, 소수점 1자리)",
        "fields": {
            "abilities_count":  len(all_abl),
            "character_count":  len(all_chr),
            "activities_count": len(all_act),
            "ability_items":    all_abl,
            "character_items":  all_chr,
            "activity_items":   all_act,
        },
        "jobs": {}
    }

    for jcd, info in sorted(jobs.items()):
        result["jobs"][jcd] = {
            "name":            info["name"],
            "category_large":  info["category_large"],
            "category_medium": info["category_medium"],
            "summary":         info["summary"],
            "exec_job":        info["exec_job"],
            "abilities":       info["abilities"],
            "character":       info["character"],
            "activities":      info["activities"],
        }

    return result


if __name__ == '__main__':
    print("=" * 55)
    print("KNOW 직업사전 XML → JSON 파싱 시작")
    print("=" * 55)

    jobs   = parse_zip(ZIP_PATH)
    result = build_json(jobs)

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n완료!")
    print(f"  직업 수:      {result['total_jobs']}개")
    print(f"  능력 항목:    {result['fields']['abilities_count']}종")
    print(f"  성격 항목:    {result['fields']['character_count']}종")
    print(f"  활동 항목:    {result['fields']['activities_count']}종")
    print(f"  저장 경로:    {OUT_PATH}")
