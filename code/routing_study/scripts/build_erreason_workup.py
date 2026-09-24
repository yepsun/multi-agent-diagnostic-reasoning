import os as _os
#!/usr/bin/env python3
"""构建 ER-Reason 子集的 workup-informed 输入 → data/er_reason_workup_subset.json。

信息密度对照实验：条件 A（现有）= 仅临床表现（ED 医师笔记截断于 MDM 之前）；
条件 B（本脚本）= 临床表现 + 本次就诊的客观检查结果（化验/生命体征、影像/
心电/超声报告及其 impression、ED 病程中记录的客观体检与监测数据）。

客观结果来源：ED_Provider_Notes_Text 中 "Final Disposition and ED Course"
之后的 ED Course 段。同一 CSV 的 Imaging_Text / ECG_Text / Echo_Text /
HP_Note_Text 列与本次 encounter **不对齐**（364 例实测：影像/心电文本与 ED
笔记的字符级重合仅 17/676，接近随机），故一律不用。

抽取与金标签完全无关：primaryeddiagnosisname 只用于事后统计标记
objective_contains_gold。

保留：
  T1 result — 化验/血气/尿检数值行、生命体征、影像/心电/超声报告区块
              （IMPRESSION / FINDINGS / ACTIONABLE FINDINGS / WET READ /
              "my interpretation ..." / TECHNIQUE 等）及其正文，以及带定量
              结果的客观发现句（"Low lung volumes with small right pleural
              effusion"）。
  T2 obs    — ED 病程中的客观体检与监测句（"Exam: ... no murmurs"、
              "O2 dipped to 80%"、"Tolerating PO"）。
  T3 note   — ED 病程中的非管理性临床叙述（症状/一般情况），仅作兜底。
  T2/T3 仅在 T1 不足 200 字符时依次补齐，保证各例新增信息量可比。
丢弃：
  - MDM 评估与处置、disposition/去向（admit/discharge/transfer/5150/…）
  - 处方与用药清单、随访与返院嘱托
  - 分诊/会诊/床位/转运等沟通句、交接班复述（sign out / handoff）
  - 模板噪音、文献引用、家属转述、诊断清单（"(CMS code)" 列表、R/O）
  - 非报告语境中的明确诊断断言（"likely X"、"concerning for X"、
    "diagnosis of X"、Ddx、assessment 等）

运行后在 stdout 打印可写入验证报告的指标：覆盖率（客观段 ≥200 字符）、
presentation/objective/text 的分位数、两类泄漏率、results/default/loose
三种策略的对比，以及低覆盖 case_id 清单。
实测（364 例子集）：覆盖率 65.9%（目标 >80%，未达标，原因见报告
routing_study/results/erreason_workup_input_report.md 第 3、6 节）；
泄漏 (a) 2.2%（非报告语境 1.9%）；泄漏 (b) 严判 1.4%、人工复核后 0%。

用法：
    ./.venv/bin/python routing_study/scripts/build_erreason_workup.py
    ./.venv/bin/python routing_study/scripts/build_erreason_workup.py --audit 20
    ./.venv/bin/python routing_study/scripts/build_erreason_workup.py --show <case_id>
"""
import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_erreason_subset import truncate_ed_note  # noqa: E402

CSV = ROOT / "data" / "er_reason" / "er_reason.csv"
SUBSET = ROOT / "data" / "er_reason_subset.json"
OUT = ROOT / "data" / "er_reason_workup_subset.json"

csv.field_size_limit(10**9)

MIN_OBJ_CHARS = 200
MAX_OBJ_CHARS = 6000
OBJECTIVE_HEADER = "Objective results from this ED visit:"

# ============================================================ 段落定位

RE_FINAL_DISP = re.compile(r"Final Disposition", re.I)
RE_ED_COURSE = re.compile(r"ED Course", re.I)
RE_COURSE_USER_INDEX = re.compile(r"ED Course User Index", re.I)
RE_ED_DISPO = re.compile(r"ED DISPOSITION\s*:", re.I)
RE_MEDS_PRESCRIBED = re.compile(r"Medications Prescribed this Visit", re.I)


RE_CUT_START = re.compile(
    r"(?i)(ED DISPOSITION\s*:|Medications Prescribed this Visit|"
    r"FOLLOW UP\s*:|ED Course User Index)")
RE_CUT_END = re.compile(
    r"(?i)(Course as of|ED Course|'s Documentation|Others' Documentation)")


def isolate_ed_course(note):
    """切出 ED Course 段（止于 User Index）。disposition/处方抬头在
    片段循环里用 cut 状态剔除，避免误伤紧随其后的病程内容。"""
    m = RE_FINAL_DISP.search(note)
    seg = note[m.start():] if m else note
    m = RE_ED_COURSE.search(seg)
    if m:
        seg = seg[m.end():]
    m = RE_COURSE_USER_INDEX.search(seg)
    if m:
        seg = seg[:m.start()]
    return seg.replace("\xa0", " ").replace("\t", " ")


RE_TIMESTAMP = re.compile(r"(?:(?<=\s)|(?<=\n)|^)(\d{3,4})\s(?=[A-Z0-9(*])")


def split_entries(seg):
    out, pos = [], 0
    for m in RE_TIMESTAMP.finditer(seg):
        body = seg[pos:m.start()].strip()
        if body:
            out.append((seg[max(0, pos - 6):pos].strip()[-4:] or None, body))
        pos = m.end()
    tail = seg[pos:].strip()
    if tail:
        out.append((None, tail))
    return out


RE_FRAG_SPLIT = re.compile(r"(?<=[.!?])\s+|\s{2,}|\n+")


def split_fragments(entry):
    raw = [p.strip() for p in RE_FRAG_SPLIT.split(entry) if p and p.strip()]
    out, buf = [], ""
    for p in raw:
        buf = (buf + " " + p).strip() if buf else p
        if len(buf) >= 14:
            out.append(buf)
            buf = ""
    if buf:
        if out:
            out[-1] = out[-1] + " " + buf
        else:
            out.append(buf)
    return out


# ============================================================ 噪音

RE_SECTION_HDR = re.compile(
    r"(?i)^\W*(?:\d{3,4}\s+)?(?:ed course(?: as of[\d/: ]*)?|others?'? documentation|"
    r"user index|[\w* .,'-]{0,45}?'?s documentation)(?:[\d/: ]*)\W*$")
RE_BARE_DATE = re.compile(r"^[\W]*[\d/: ]{6,22}[\W]*$")
RE_SIGNATURE = re.compile(
    r"(?:^|\s)(?:MD|DO|NP|PA-C|RN|Resident|Attending|PharmD|CRNP|PAC)\b"
    r"[^A-Za-z0-9]{0,8}(?:[\d/]{5,10}\s*)?\d{3,4}\s*[\W]*$|"
    r"^\W*(?:[\d/]{5,10}\s*)?\d{3,4}\s*[\w* .,'-]{1,45}?"
    r"(?:MD|DO|NP|PA-C|RN|Resident|Attending|PharmD)\W*$", re.I)
RE_REDACT_ONLY = re.compile(r"^[\W_]+$")
RE_TEMPLATE = re.compile(
    r"(\{\s*\*+\s*\|)|My Note\s*/\s*Procedures|If a procedure is done this visit|"
    r"create a \*+ now|^\s*>>|\[\s*\]\s*$|^\s*\{")
RE_CITATION = re.compile(
    r"(?i)(\bdoi\b|et al\.|pmid|\bPMID\b|resistance (?:usually )?develops|"
    r"^\W*aureus|\d{1,2}:\s*\d{3,4}\s*[-–]\s*\d{1,2}:\s*\d{3,4}|\b\d{1,2}\.\d{1,2}%|"
    r"the most frequent of these mutations|\bNCBI\b|UptoDate|UpToDate)")

# ============================================================ 丢弃：ED 处置层

RE_ED_MGMT = re.compile(
    r"(?i)("
    r"\badmit(?:ted|ting|s)?\b|\badmission\b|\bre-?admit\b|"
    r"\bdischarg\w+|\bd/c\b|\bdc home\b|\bdc'd\b|"
    r"\btransfer(?:red|ring)?\b|\bplacement\b|\bplacing\b|\bplaced in\b|"
    r"\bsign\s?-?\s?out\b|\bsigned out\b|\bhand\s?-?off\b|\bs\/o from\b|"
    r"\bpaged?\b|\bpaging\b|\bon\s+5150|\b5150\b|\binvoluntary (?:hold|transfer)\b|"
    r"\bmedications? prescribed\b|\bprescri\w+|\bpharmacy\b|\brefill\b|\bsig\b|"
    r"\btake \d|\bcapsule\b|\btablet\b|\bby mouth\b|"
    r"\btransport\b|\bambulance\b|\bBLS\b|\bALS transport\b|\bambulat\w+|\bambulatory\b|"
    r"\bawait\w*\b|\bpending (?:placement|admission|bed|transport)\b|"
    r"\bmedically cleared\b|\bmed cleared\b|\bmedically stable for (?:transfer|discharge)\b|"
    r"\bstable for (?:transfer|discharge|placement)\b|\bclearance\b|"
    r"\bdisposition\b|\bsocial work\b|\bcase management\b|"
    r"\bno (?:admit|bed) orders\b|\bplease place\b|"
    r"\bplaced\b\s*$"
    r")")

# 报告正文里仍需剔除的：明确 ED 去向/交接/处方/转运/评估复述
RE_ED_MGMT_IN_REPORT = re.compile(
    r"(?i)("
    r"\badmitted to\b|\bfor admission\b|\bplan(?:ned)? for admission\b|"
    r"\bdischarg\w+|\bd\/c home\b|\btransfer(?:red)? to\b|"
    r"\bsign\s?-?\s?out\b|\bsigning out\b|\bsigned out\b|\bhand\s?-?off\b|"
    r"\boncoming provider\b|\boff-?going\b|\bhandoff\b|"
    r"\bpaged?\b|\bpaging\b|\b5150\b|\bplacement\b|"
    r"\bmedications? prescribed\b|\bby mouth\b|"
    r"\bmedically cleared\b|\bmedically stable for\b|\btransport back\b|"
    r"\bclinical impression\b|\bimpression:\s*\w+\s+\w+\s+\w+\b|"
    r"\bdiscussed with\b|\bspoke with\b|\bspeak with\b|\basked (?:nurse|rn|staff)\b|"
    r"\bconsul(?:t|ted|ting)\b|\bsocial work\b|"
    r"\bpresents? with\b|\bpresenting with\b|\bp\/w\b|\bh\/o\b|\bhx of\b|"
    r"\bhere (?:for|with)\b|\bhistory of present\b|\bse ?en (?:by|in)\b|"
    r"\bwill (?:admit|plan|follow|call|reassess|re-?eval|discharge|sign)\b|"
    r"\bplan (?:for|to|is)\b|\bfollow[- ]?up with\b|\bpending (?:placement|admission|bed)\b"
    r")")

# 片段是否"像报告正文"（独立于抽取状态，用于泄漏判定）
RE_REPORT_LIKE_NUM = re.compile(
    r"(?i)^\W*\d{1,2}[.)]\s+\S")


def is_report_like(f):
    if RE_REPORT_HEADER.search(f):
        return True
    if RE_ED_MGMT_IN_REPORT.search(f):
        return False
    if re.search(rf"(?i)^\W*(?:\d{{3,4}}\s+)?(?:{MODALITY})\b", f):
        return True
    return bool(RE_REPORT_LIKE_NUM.match(f) and re.search(rf"(?i)({FINDING})", f))


def is_report_continuation(f):
    """报告正文的续行判据：编号条目 / 报告关键词 / 定量描述 / 影像词汇。"""
    if RE_REPORT_HEADER.search(f):
        return True
    if RE_REPORT_LIKE_NUM.match(f):
        return True
    if re.search(r"(?i)\b\d+(?:\.\d+)?\s*(?:cm|mm|cc|ml|l)\b", f):
        return True
    if re.search(r"(?i)\b(?:series|image|im:)\b", f):
        return True
    return bool(re.search(rf"(?i)({MODALITY})", f)
                and not RE_PLAN_OR_RESTATE.search(f))

# ============================================================ 丢弃：计划/复述/断言

RE_PLAN_OR_RESTATE = re.compile(
    r"(?i)("
    r"\bconsul(?:t|ted|ts|tation)\b|\brecommend\w*\b|\bplan(?:ned|s)?\b|"
    r"\bto do\b|\bto-do\b|\borders?\b|\bordered\b|"
    r"\bwill\s+(?:admit|re-?dose|give|start|order|sign|call|follow|place|"
    r"arrange|reassess|re-?eval|dispo|d/c|obtain|get|repeat|wait|treat|"
    r"continue|hold|monitor|awake|expect)\b|"
    r"\bfollow[- ]?up\b|\bf\/u\b|\breturn (?:to|if)\b|\bcome back\b|"
    r"\barrang\w+\b|\bawait\w*\b|\bpending\b|"
    r"\bpresents? (?:with|to)\b|\bpresenting with\b|\bp/w\b|\bpresented\b|"
    r"\bh\/o\b|\bhx\b|\bhistory of present\b|\bsent (?:in|to) (?:for|by)\b|"
    r"\bs\/p\b|"
    r"\battending note\b|\battending\b|\bresident\b|\bthis patient was\b|"
    r"\bnot involved in the care\b|\bper chart review\b|"
    r"\breceived sign ?out\b|\bs/?o from\b|\bsigned out to\b|"
    r"\bfor (?:admission|transfer|discharge|placement|surgery|the OR)\b|"
    r"\badmit to\b|\bconsult(?:ed)? by\b|\brequests?\b|\brequesting\b|"
    r"\bfeels? (?:that|like)\b|\bwould like\b|\bthink(?:s)?\b|\bwant(?:s|ed)?\b|"
    r"\bno longer\b|\bwill discuss\b|\bdiscussed (?:with|the)\b|"
    r"\bper (?:surgery|ortho|neuro|cards?|gi|pulm|psyc|heme|onc|ir|radiology|"
    r"medicine|urology|the\s+\w+ (?:team|service))\b|"
    r"\bfrom (?:the )?(?:operative|ortho|surgery|medicine|neurology|cardiology) "
    r"(?:note|report)\b"
    r")")

# 非报告语境的诊断断言（决定该片段是否可能泄漏答案）
RE_DX_ASSERT = re.compile(
    r"(?i)("
    r"\bdiagnos\w+\b|\bdx\b|\bddx\b|\bclinical impression\b|\bimpression of\b|"
    r"\bfound to have\b|\brule[ds]? (?:in|out)\b|\bassessment\s*:|\ba\/p\s*:|"
    r"\bconsistent with\b|\bconsistent w\/|\bc\/w\b|\bcorrelates? with\b|"
    r"\b(?:likely|probably|possibly|concerning for|concern(?:ed|ing)? for|"
    r"suggestive of|suggests?|suspicious for|indicative of|worrisome for|"
    r"favou?rs?|favou?red to represent|represents?|appears to be|is a case of|"
    r"picture of|read as|reads as|impression that|in keeping with|"
    r"indeterminate for|not excluded|compatible with)\b|"
    r"\b(?:here|admitted|hospitalized|seen|treated|evaluated) (?:for|with)\b|"
    r"\bhistory of present\b|\bhx\b.{0,40}\b(?:with|for|p\/w|c\/b)\b|"
    r"\b(?:acute|chronic|new|known|suspected|possible) [a-z]+(?:itis|emia|osis|"
    r"pathy|itis|oma|itis)\b|"
    r"\bpresumed\b|\bverified\b|\bwork(?:-| )?up (?:revealed|showed|consistent)\b"
    r")")

# ============================================================ 保留规则

LAB_NAME = (
    r"hemoglobin|hematocrit|platelets?|\bwbc['s]?\b|\brbc['s]?\b|sodium|potassium|"
    r"chloride|bicarbonate|\bco2\b|\bbun\b|creatinine|glucose|\bcalcium\b|magnesium|"
    r"phosph\w*|\bast\b|\balt\b|alkaline phosphat\w*|bilirubin|albumin|"
    r"lipase|amylase|troponin|\bbnp\b|pro-?bnp|lactate|\bph\b|\bpco2\b|\bpo2\b|"
    r"\binr\b|\bptt\b|\bpt\b|d-?dimer|\bcrp\b|c-reactive|sedimentation|\besr\b|"
    r"\btsh\b|\bt4\b|hba1c|acetone|ketones?|ethanol|alcohol|salicylate|"
    r"acetaminophen|ammonia|specific gravity|urinalysis|\bua\b|"
    r"venous blood gas|blood gas|\bcbc\b|\bbmp\b|\bcmp\b|lactic acid|\bcrt\b|"
    r"\bhcg\b|pregnancy test|influenza|\bcovid\b|\brsv\b|blood culture|"
    r"urine culture|\butox\b|drug screen|creatine kinase|\bck\b|"
    r"erythrocyte|\bwbc[s]?\b|rbc[s]?\b|leukocyte|nitrite|esterase|"
    r"international normalized|\bpt\/inr\b|\bpoc\b|\bpoc\s?glucose"
)
MODALITY = (
    r"x-?ray|\bXR\b|\bcxr\b|\bct\b|cta\b|\bmri\b|\bmra\b|\bpet\b|ultrasound|"
    r"sonograph\w*|\bus\b|pocus|echocardiogra\w*|\becho\b|\btt[ee]\b|\bekg\b|\becg\b|"
    r"electrocardiogram|12[- ]?lead|lead 12|radiograph\w*|doppler|mammogra\w*|"
    r"angiogram|nuclear medicine|fluoroscop\w*|\bviews?\b|\bscan\b|\bkxr\b|"
    r"venogram|mri/mra|\btee\b|nuclear stress|ctap|\bcta\b"
)
FINDING = (
    r"no evidence of|no acute |no fluid|no pneumothorax|no fracture|no pleural|"
    r"no free air|no focal |no mass|no lesion|no hemorrhage|no consolidation|"
    r"no abnormality|no abnormal|no stenosis|no occlusion|no thrombus|"
    r"no calculus|no stone|no obstruction|no sign of|no suspicion|"
    r"unremarkable|non-?contributory|non-?concerning|nonconcerning|unrevealing|"
    r"within normal|normal limits|\bWNL\b|\bNSR\b|sinus rhythm|"
    r"negative for|positive for|grossly normal|grossly unchanged|"
    r"reassuring|grossly intact|intact|"
    r"unchanged (?:from|compared)|compared (?:to|with) \d|"
    r"measuring up to|measures up to|up to \d+(?:\.\d+)? ?(?:mm|cm)\b|"
    r"effusion|opacit(?:y|ies)|consolidation|atelectasis|pneumothorax|"
    r"cardiomegaly|nodul\w+|mass\b|lesion|fractur\w+|calcul(?:us|i)|"
    r"gallstone|cholelithiasis|diverticul\w+|thrombus|stenosis|occlusion|"
    r"hemorrhage|infarct\w*|aneurysm|lymphadenopathy|herniation|hydrocephalus|"
    r"free fluid|collection|dilat\w+|bowel|air-?fluid|osteomyelit\w+|"
    r"calcification|emphysema|infiltrate|mucus plugging|soft tissue swell\w+|"
    r"fat stranding|edema|hypodensit\w+|hyperdensit\w+|hypodense|hyperdense|"
    r"airspace disease|interstitial|bibasilar|basilar|apical|bibasal|"
    r"pleural|pericardial|abdominal aortic|extravasation|pneumonia\b|"
    r"tenderness|tender|distention|distended|erythema|erythematous|"
    r"crepitus|cyanosis|jaundice|scleral icterus|swelling|swollen"
)
RE_RESULT = re.compile(
    r"(?i)("
    rf"\b(?:{LAB_NAME})\b[^.\n]{{0,22}}[:=]?\s*[<>]?\s*\d|"
    r"\d+(?:\.\d+)?\s*(?:mg/dl|mmol/l|meq/l|g/dl|mg\b|ml\b|mcg|ng/ml|pg/ml|"
    r"u/l|iu/l|mm/hr|k/ul|/ul|/mm3|/hpf|10\*3|%|mmhg|celsius|fahrenheit|"
    r"sec\b|ms\b|fl\b|ng\b|pg\b|units?/l|miu/ml|ng/dl|meq\b)\b|"
    r"\bBP\b\s*[\d(]|\bHR\b\s*[:=]?\s*\d|\bSpO2\b|O2 sat|"
    r"\b\d{2,3}/\d{2,3}\b|\b\d+\s*(?:beats|bpm)\b|"
    r"\bIMPRESSION\b|\bFINDINGS\b|ACTIONABLE FINDINGS|WET READ|"
    r"PRELIMINARY (?:READ|INTERPRETATION)|RADIOLOGY\b|\bPRELIM\b|"
    r"\bTECHNIQUE\b|\bInterpretation Summary\b|\bSTUDY\b\s*:|COMPARISON\b|"
    r"\baddendum\b|\bwet read\b|"
    rf"\b(?:{MODALITY})\b|"
    rf"(?:{FINDING})|"
    r"sinus tachycardia|sinus bradycardia|afib|atrial fibrillation|"
    r"ST elevation|ST depression|T-wave|QTc|poor R wave|"
    r"within normal limits|mildly|moderately|severely|markedly|"
    r"stable\b|improved\b|resolved\b|interval (?:worsening|improvement|change)|"
    r"grossly|significant(?:ly)?\b|minimal\b|trace\b|small\b|large\b"
    r")")

EXAM = (
    r"tachycardic|tachycardia|bradycardic|bradycardia|murmur\w*|rub\b|gallop|"
    r"lungs?\s+(?:are\s+)?(?:clear|CTAB)|CTAB|breath sounds|wheez\w+|crackles|"
    r"rales|rhonchi|stridor|accessory muscle|retractions|"
    r"abdomen|abdominal exam|soft,|nontender|non-tender|tender|guarding|"
    r"rebound|distend\w+|bowel sounds|"
    r"JVD|jugular venous|\bperfusion\b|capillary refill|pulses?|"
    r"extremit\w+ (?:warm|cool|cold)|\bno edema\b|\bedema\b|"
    r"rash|erythema|cellulitis|ulcer|wound|drainage|"
    r"neurolog\w+|focal deficit|CN\s*(?:II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|\d)|"
    r"pupils?|PERRL|gait|strength|sensation|reflex\w*|"
    r"mental status|\bAOx?\b|alert and oriented|\bawake\b|\bconfused\b|"
    r"well[- ]appearing|\bnontoxic\b|ill[- ]appearing|diaphoretic|pale\b|"
    r"mucous membranes|conjunctiv\w+|sclera|"
    r"\bexam\b|\bexamination\b|\bPE\b\s*:|\bROS\b"
)
RE_OBS = re.compile(
    r"(?i)("
    r"\bBP\b|\bHR\b|heart rate|\bRR\b|respiratory rate|\bspo2\b|O2 sat|"
    r"temperature|\btemp\b|\bvitals?\b|\bafebrile\b|\bhypotensive\b|\bhypertensive\b|"
    r"\bhypoxic\b|\bdesat\w+|\bO2\b|oxygen|"
    r"\btolerat\w+|\bPO intake\b|"
    r"\bdenies\b|\breports?\b|\bendorses\b|\bcomplains?\b|\bstates?\b|"
    r"\bstable\b|\bunchanged\b|\bimproved\b|\bworsened\b|\bdeteriorat\w+|"
    rf"(?:{EXAM})|"
    r"\bno (?:acute )?(?:distress|focal deficit|trauma|abnormalit\w+|signs)\b|"
    r"\d+\s*(?:beats|bpm)\b|\bnormal\b|\bdecreased\b|\bincreased\b|"
    r"\bnotable for\b|\breveal\w*|\bshow(?:s|ed|ing)\b|\bfound\b|"
    r"\bunable to\b|\bdeclin\w+|\bwithdrew\b|\bagitat\w+|\bcombative\b|"
    r"\bsleeping\b|\bsedated\b|\brestrain\w*|\bcooperat\w+|\bno distress\b"
    r")")

# T3 兜底：ED 病程中的临床叙述（症状/体征/一般情况），需含临床词汇
CLINICAL = (
    r"pain|painful|nausea|vomit\w*|emesis|fever|febrile|chills|rigors|"
    r"cough|dyspnea|shortness of breath|\bsob\b|hemoptysis|wheez\w*|"
    r"headache|dizz\w+|syncope|lightheaded\w*|weakness|numb\w+|tingling|"
    r"confusion|confused|somnolen\w+|letharg\w+|anxi\w+|agitat\w+|"
    r"abdominal|abdomen|\bchest\b|\bback\b|\bflank\b|\bpelvis\b|\bgroin\b|"
    r"\bhead\b|\bneck\b|\bthroat\b|\bleg\b|\barm\b|\bfoot\b|\bhand\b|\bhip\b|"
    r"breathing|breath sounds|airway|respiratory|cardiac|cardiopulmonary|"
    r"\bheart\b|\blung\w*|\bkidney\w*|renal|hepatic|\bliver\b|spleen|bladder|"
    r"\buro\b|urinary|urine|stool|bowel|gastric|epigastric|"
    r"\bwound\b|ulcer|rash|lesion|swelling|swollen|edema|erythema|"
    r"\bblood\b|hemoglobin|platelet|electrolyte|glucose|\bsugar\b|"
    r"oxygen|saturation|pressure|pulse|rhythm|rate\b|monitor\w*|"
    r"intake|output|fluid|hydration|dehydrat\w*|"
    r"symptom\w*|patient|pt\b|complaint\w*|history|medication\w*|dose|"
    r"catheter|foley|\bline\b|dressing|bandage|restraint\w*|"
    r"improve\w*|worsen\w*|stable|resolution|resolved|unchanged|"
    r"\bawake\b|\basleep\b|ambulat\w*|walk\w*|position\w*|"
    r"nursing|\brn\b|reassess\w*|assessment|examination|\bexam\b|"
    r"\b(?:Na|K|Cl|BUN|Cr|Hgb|Hct|Plt|INR|PTT|AST|ALT|WBC|RBC|TSH|CRP|ESR|"
    r"BNP|TSAT|HbA1c|LFTs?|CBC|BMP|CMP|VBG|ABG|UA|PT)\b"
)
RE_CLINICAL = re.compile(rf"(?i)({CLINICAL}|\d)")

RE_REPORT_HEADER = re.compile(
    r"(?i)("
    r"\bIMPRESSION\b|\bFINDINGS\b|ACTIONABLE FINDINGS|WET READ|"
    r"PRELIMINARY (?:READ|INTERPRETATION)|RADIOLOGY PRELIMINARY|"
    r"\bPRELIM\b\s*:?|\bTECHNIQUE\b|\bInterpretation Summary\b|"
    r"my (?:independent )?interpretation|"
    r"\bCOMPARISON\b|\bHISTORY\b\s*:|\bSTUDY\b\s*:|\bINDICATION\b\s*:|"
    r"the following read|with the following|"
    r"\b(?:XR|CT|CTA|MRI|MRA|CXR|US|EKG|ECG|TTE|TEE|ECHO)\b[^.\n]{0,40}?"
    r"(?:with|without|views?|read|report|study|exam)\b|"
    r"\[w?et read\]|wet read summary|interpretation\s*:"
    r")")

# 报告正文尾随的"非报告"信号（用于退出报告模式，防止把交接班叙述并入）
RE_LEAVE_REPORT = re.compile(
    r"(?i)("
    r"\bpresents? (?:with|to)\b|\bpresenting with\b|\bp/w\b|\bhx\b|\bh/o\b|"
    r"\bhere (?:for|with|after)\b|\bmedically cleared\b|\bsign ?out\b|"
    r"\bsigned out\b|\badmit(?:ted)? to\b|\bdischarg\w+ home\b|"
    r"\bdiscussed with\b|\bpaged\b|\bconsul(?:t|ted)\b|\bpending placement\b|"
    r"\b(?:placed|taken) (?:in|to) (?:a |the )?(?:bed|room|hallway)\b"
    r")")


RE_CREDENTIAL_ONLY = re.compile(
    r"(?i)^[\w* .,'-]{0,50}?\b(?:MD|DO|NP|PA-C|RN|CRNP|LPN|Resident|Attending|"
    r"PharmD|PAC|CNS)\b[\w* .,'-]{0,10}$")


# 若干"看似客观、实为诊断清单/医嘱/复述"的杂项（与金标签无关的格式特征）
RE_DROP_MISC = re.compile(
    r"(?i)("
    r"\(cms code\)|"                                   # 账单/诊断清单标记
    r"\br\s?/\s?o\b|\brule out\b|"                     # R/O 诊断
    r"\breturn precautions?\b|\breturn if\b|"
    r"\bdischarge instructions?\b|"
    r"^\W*s\s?/\s?o\b|\bs\/o from\b|"                  # sign-out 抬头
    r"\bhere w\/|\bpw\b|\bp\/w\b|"
    r"^\W*(?:[\w*]*\*[\w*]*)(?:-(?:wife|husband|family|daughter|son|mother|"
    r"father|partner|sister|brother))?\s*:|"           # 家属转述
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg)\b(?!\s*/)|"         # 给药剂量（非化验单位）
    r"\b(?:tablet|capsule|pill)\b|\btake one\b"
    r")")


def is_noise(f):
    if RE_TEMPLATE.search(f) or RE_CITATION.search(f):
        return True
    if RE_SECTION_HDR.match(f) or RE_BARE_DATE.match(f):
        return True
    if RE_SIGNATURE.search(f) or RE_REDACT_ONLY.match(f):
        return True
    if len(f) < 45 and RE_CREDENTIAL_ONLY.match(f):
        return True
    return len(f) < 5


def clean_fragment(frag):
    frag = re.sub(r"^\d{1,2}[.)]\s*", "", frag.strip())
    frag = re.sub(r"\s+", " ", frag).strip(" \";'")
    frag = re.sub(r"^[,;:.\-]+\s*", "", frag)
    return frag


def looks_like_modality_read(f):
    """检查名称/时间戳开头的报告引子（"CXR: Low lung volumes..."）。"""
    if len(f) < 12:
        return False
    if re.search(rf"(?i)^\W*(?:\d{{3,4}}\s+)?(?:{MODALITY})\b", f):
        return True
    return bool(re.search(r"(?i)(read|report|interpretation|summary|study)\s*:$", f))


POLICIES = ("results", "default", "loose")


def extract_objective(note, fallback=True, policy="default"):
    """返回 (objective_text, meta)。全程不接触金标签。

    policy:
      results — 只保留 T1（化验/生命体征/影像报告），最高精度、最低覆盖
      default — T1 不足 200 字符时依次补 T2（客观体检/观测）与 T3（病程临床叙述）
      loose   — 只剔除管理/去向/交接/处方句，其余全部保留（覆盖上限，含评估计划）

    meta["tags"] 与文本行一一对应：
      report — 影像/心电/超声报告区块（含表头、FINDINGS/IMPRESSION 正文）
      result — 化验/生命体征/定量客观结果
      obs    — 客观体检与病情观测
      note   — T3 兜底：ED 病程临床叙述（症状/一般情况，非管理性）
    """
    seg = isolate_ed_course(note)
    t1, t2, t3 = [], [], []
    in_report = False
    in_cut = False
    for ts, entry in split_entries(seg):
        for frag in split_fragments(entry):
            f = frag.strip()
            if not f:
                continue
            if RE_CUT_START.search(f):
                in_cut = True
                continue
            if in_cut:
                if RE_CUT_END.search(f):
                    in_cut = False
                else:
                    continue
            if is_noise(f):
                if RE_SECTION_HDR.match(f):
                    in_report = False
                continue
            if RE_ED_MGMT.search(f):
                in_report = False
                continue
            elif in_report and not is_report_continuation(f):
                in_report = False
            strong_header = bool(RE_REPORT_HEADER.search(f))
            if strong_header or (
                    not in_report and not RE_PLAN_OR_RESTATE.search(f)
                    and looks_like_modality_read(f)):
                in_report = True
            if in_report:
                c = clean_fragment(f)
                if c and not RE_ED_MGMT_IN_REPORT.search(c):
                    t1.append((c, "report"))
                elif c:
                    in_report = False
                continue
            if policy == "loose":
                t3.append((clean_fragment(f), "note"))
                continue
            if RE_PLAN_OR_RESTATE.search(f) or RE_DX_ASSERT.search(f) \
                    or RE_DROP_MISC.search(f):
                continue
            if RE_RESULT.search(f):
                c = clean_fragment(f)
                if c:
                    t1.append((c, "result"))
            elif RE_OBS.search(f):
                c = clean_fragment(f)
                if c:
                    t2.append((c, "obs"))
            elif RE_CLINICAL.search(f) and len(f) >= 12:
                c = clean_fragment(f)
                if c:
                    t3.append((c, "note"))
        for bucket in (t1, t2, t3):
            if ts and bucket and not re.match(r"^\W*\d{3,4}\b", bucket[-1][0]):
                bucket[-1] = (f"{ts} {bucket[-1][0]}", bucket[-1][1])
    kept = t1
    if policy == "loose":
        kept = t1 + t2 + t3
    elif policy == "default" and fallback:
        if sum(len(x) + 1 for x, _ in kept) < MIN_OBJ_CHARS:
            kept = kept + t2
        if sum(len(x) + 1 for x, _ in kept) < MIN_OBJ_CHARS:
            kept = kept + t3
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(x for x, _ in kept)).strip()
    tags = [t for _, t in kept]
    if len(text) > MAX_OBJ_CHARS:
        lines = text[:MAX_OBJ_CHARS].rsplit("\n", 1)
        text = lines[0]
        tags = tags[:len(text.split("\n"))]
    return text, {"tier2_used": len(kept) > len(t1), "tags": tags}


# ============================================================ 指标

def norm(s):
    s = s.replace("\xa0", " ")
    s = re.sub(r"\(cms code\)", " ", s, flags=re.I)
    s = re.sub(r"(?i),\s*(?:initial|subsequent)\s+encounter", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def contains_gold(objective, gold):
    g = norm(gold)
    if not g:
        return False
    o = norm(objective)
    if g in o:
        return True
    for part in [p.strip() for p in re.split(r"[,/]", gold) if len(p.strip()) > 4]:
        p = norm(part)
        if len(p.split()) >= 2 and p in o:
            return True
    return False


RE_DX_DEFINITIVE = re.compile(
    r"(?i)("
    r"\bdiagnos\w+\b|\bdx\s*[:.]|\bddx\b|\bclinical impression\b|\bimpression of\b|"
    r"\bfound to have\b|\brule[ds]? (?:in|out)\b|\bassessment\s*:|\ba\/p\s*:|"
    r"\bconfirmed\b|\bhas been diagnosed\b|\bknown (?:history of|dx)\b|"
    r"\bconsistent with (?:a |the )?(?:diagnosis|known|history)\b|"
    r"\bthis is (?:a |an )?case of\b|\bpatient (?:has|is a case of)\b|"
    r"\b(?:admitted|here|seen|treated|transferred) (?:for|with) [a-z]+\b"
    r")")

RE_DX_BROAD = re.compile(
    r"(?i)("
    r"\bdiagnos\w+\b|\bdx\b|\bddx\b|\bimpression\b|\bassessment\b|"
    r"\bconsistent with\b|\bconsistent w/|\bc\/w\b|\bcorrelates? with\b|"
    r"\b(?:likely|probably|possibly|concerning|suggest\w*|suspicious|"
    r"indicative|worrisome|favou?rs?|favou?red|represents?|appears to be|"
    r"in keeping with|compatible with|read as|reads as|concern for)\b"
    r")")


def split_tags(objective, tags):
    """把客观段按 provenance 分成 report / 非 report 两组片段。"""
    lines = [l.strip() for l in objective.split("\n")]
    lines = [l for l in lines if l]
    if len(tags) != len(lines):
        tags = ["?"] * len(lines)
    return list(zip(lines, tags))


def dx_hits(objective, tags, pattern):
    """非报告语境片段中命中 pattern 的句子（报告语境判据独立于抽取状态）。"""
    return [f for f, _ in split_tags(objective, tags)
            if not is_report_like(f)
            and pattern.search(re.sub(r"^\W*\d{3,4}\s+", "", f))]


def gold_hits(objective, tags, gold):
    """(a) 类：金标签字面出现；返回 (全部命中, 非报告语境命中)。"""
    all_hits, non_report = [], []
    for f, _ in split_tags(objective, tags):
        if contains_gold(f, gold):
            all_hits.append(f)
            if not is_report_like(f):
                non_report.append(f)
    return all_hits, non_report


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def metrics(cases, label):
    obj = [len(c["objective"]) for c in cases]
    return {
        "label": label, "n": len(cases), "obj": obj,
        "coverage": sum(1 for l in obj if l >= MIN_OBJ_CHARS) / len(cases),
        "pres": [len(c["presentation"]) for c in cases],
        "tot": [len(c["text"]) for c in cases],
        "leak_a": sum(1 for c in cases if c["_gold_all"]) / len(cases),
        "leak_a_nonreport": sum(1 for c in cases if c["_gold_nonreport"]) / len(cases),
        "tier2": sum(1 for c in cases if c["_tier2"]) / len(cases),
        "leak_b": sum(1 for c in cases if c["_dx"]) / len(cases),
    }


def build_case(row, presentation, policy="default"):
    obj, meta = extract_objective(row["ED_Provider_Notes_Text"] or "",
                                  policy=policy)
    gold = (row["primaryeddiagnosisname"] or "").strip()
    text = (f"{presentation}\n\n{OBJECTIVE_HEADER}\n{obj}" if obj else presentation)
    gold_all, gold_nonreport = gold_hits(obj, meta["tags"], gold)
    return {
        "case_id": row["encounterkey"],
        "gold": gold,
        "presentation": presentation,
        "objective": obj,
        "text": text,
        "objective_contains_gold": bool(gold_all),
        "_tags": meta["tags"],
        "_tier2": meta["tier2_used"],
        "_gold_all": gold_all,
        "_gold_nonreport": gold_nonreport,
        "_dx": dx_hits(obj, meta["tags"], RE_DX_DEFINITIVE),
        "_dx_broad": dx_hits(obj, meta["tags"], RE_DX_BROAD),
    }


def presentation_of(row):
    note = truncate_ed_note((row["ED_Provider_Notes_Text"] or "").strip())
    return (f"Age: {row['Age']}\nSex: {row['sex']}\n"
            f"Chief complaint: {row['primarychiefcomplaintname']}\n\n"
            f"ED note:\n{note}")


# ============================================================ 粗过滤对照

COARSE_DROP = re.compile(
    r"(?i)\b(likely|consistent with|suggests?|suggestive|diagnosis|diagnosed|"
    r"plan|recommend|admit|discharge|disposition|impression:)\b")
COARSE_KEEP = re.compile(
    r"(?i)(\d+(?:\.\d+)?\s*(?:mg/dl|mmol/l|meq/l|g/dl|%|mmhg)|:\s*[<>]?\d|"
    r"\bIMPRESSION\b|\bFINDINGS\b|\bXR\b|\bCT\b|\bMRI\b|\bEKG\b|\bECG\b|"
    r"ultrasound|radiograph|\bx-?ray\b)")


def coarse_objective(note):
    seg = isolate_ed_course(note)
    out = []
    for sent in re.split(r"(?<=[.!?])\s+|\s{2,}|\n+", seg):
        s = sent.strip()
        if not s or len(s) < 8 or is_noise(s):
            continue
        if COARSE_DROP.search(s) or not COARSE_KEEP.search(s):
            continue
        out.append(s)
    return "\n".join(out)


RE_AUDIT = re.compile(
    r"(?i)(diagnos\w*|\bdx\b|\bddx\b|\bimpression\b|\bassessment\b|\blikely\b|"
    r"\bconcerning\b|\bconsistent with\b|\bc\/w\b|\bsuggest\w*|\brepresent\w*|"
    r"\bfavou?r\w*|\bindicative\b|\bread as\b|\bin keeping with\b|"
    r"\bcompatible with\b|\bworrisome\b|\bprobably\b|\bknown\b)")


def audit_candidates(case, n=20, seed=20260918):
    """抽取人工抽查样本：客观段中含断言/推测语气的片段（不限 provenance）。"""
    import random
    rng = random.Random(seed)
    pool = []
    for f, tag in split_tags(case["objective"], case["_tags"]):
        if RE_AUDIT.search(re.sub(r"^\W*\d{3,4}\s+", "", f)):
            pool.append((f, tag))
    if not pool:
        return []
    pool.sort(key=lambda x: -len(x[0]))
    picked = pool[:n // 2] + rng.sample(pool, min(n - len(pool[:n // 2]), len(pool)))
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=int, default=0,
                    help="打印 N 例 (b) 类断言候选（含严判逐例）供人工抽查")
    ap.add_argument("--show", nargs="*", default=[],
                    help="打印指定 case_id 的客观段原文")
    args = ap.parse_args()

    subset = json.loads(SUBSET.read_text())
    sub_ids = [c["case_id"] for c in subset]
    sub_idset = set(sub_ids)

    rows_sub, rows_all = [], []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if not (r["ED_Provider_Notes_Text"] or "").strip():
                continue
            rows_all.append(r)
            if r["encounterkey"] in sub_idset:
                rows_sub.append(r)
    print(f"CSV 读取：子集 {len(rows_sub)}/{len(sub_ids)}，全体 {len(rows_all)}",
          file=sys.stderr)

    sub_cases, coarse = [], []
    for r in rows_sub:
        c = build_case(r, presentation_of(r))
        sub_cases.append(c)
        obj = coarse_objective(r["ED_Provider_Notes_Text"])
        gold = c["gold"]
        ga, gn = gold_hits(obj, ["?"] * len(obj.split("\n")), gold)
        coarse.append({
            "case_id": c["case_id"], "gold": gold,
            "presentation": c["presentation"], "objective": obj,
            "text": c["presentation"] + "\n" + obj,
            "objective_contains_gold": bool(ga),
            "_tags": ["?"] * len(obj.split("\n")),
            "_tier2": False, "_gold_all": ga, "_gold_nonreport": gn,
            "_dx": dx_hits(obj, ["?"] * len(obj.split("\n")), RE_DX_DEFINITIVE),
            "_dx_broad": dx_hits(obj, ["?"] * len(obj.split("\n")), RE_DX_BROAD)})
    OUT.write_text(json.dumps(
        [{k: v for k, v in c.items() if not k.startswith("_")} for c in sub_cases],
        ensure_ascii=False, indent=1))
    print(f"已写 {OUT}：{len(sub_cases)} 例", file=sys.stderr)

    all_cases = [build_case(r, presentation_of(r)) for r in rows_all]
    for mm in (metrics(sub_cases, "本抽取器(子集364)"),
               metrics(coarse, "粗过滤基线(子集364)"),
               metrics(all_cases, "本抽取器(全体3984)")):
        def line(name, xs):
            print(f"| {name} | {statistics.mean(xs):.0f} | {quantile(xs,.1)} | "
                  f"{quantile(xs,.25)} | {quantile(xs,.5)} | {quantile(xs,.75)} | "
                  f"{quantile(xs,.9)} | {max(xs)} |")
        print(f"\n## {mm['label']} (n={mm['n']})")
        print(f"- 覆盖率(客观段≥{MIN_OBJ_CHARS}字符): "
              f"{sum(1 for l in mm['obj'] if l>=MIN_OBJ_CHARS)}/{mm['n']} = "
              f"{mm['coverage']:.1%}")
        print(f"- 泄漏(a) 金标签字面出现: {mm['leak_a']:.1%}"
              f"（其中非报告语境: {mm['leak_a_nonreport']:.1%}）")
        print(f"- 泄漏(b) 明确最终诊断断言(严判): {mm['leak_b']:.1%}")
        print(f"- T2/T3 兜底启用比例: {mm['tier2']:.1%}")
        print("| 长度 | 均值 | p10 | p25 | 中位 | p75 | p90 | max |")
        print("|---|---|---|---|---|---|---|---|")
        line("presentation", mm["pres"])
        line("objective", mm["obj"])
        line("text(合计)", mm["tot"])

    print("\n## 抽取策略对比（子集364）")
    print("| 策略 | 覆盖率(≥200) | 泄漏(a) | 其中非报告语境 | 泄漏(b) | obj 均值 |")
    print("|---|---|---|---|---|---|")
    for pol in POLICIES:
        cs = [build_case(r, presentation_of(r), policy=pol) for r in rows_sub]
        mm = metrics(cs, pol)
        print(f"| {pol} | {mm['coverage']:.1%} | {mm['leak_a']:.1%} | "
              f"{mm['leak_a_nonreport']:.1%} | {mm['leak_b']:.1%} | "
              f"{statistics.mean(mm['obj']):.0f} |")

    thin = sorted([c for c in sub_cases if len(c["objective"]) < MIN_OBJ_CHARS],
                  key=lambda c: len(c["objective"]))
    print(f"\n- 子集客观段 <{MIN_OBJ_CHARS} 字符: {len(thin)} 例，"
          f"其中为空 {sum(1 for c in thin if not c['objective'])} 例")
    print(f"- 子集 (b) 严判命中: {len([c for c in sub_cases if c['_dx']])} 例；"
          f"宽网候选: {len([c for c in sub_cases if c['_dx_broad']])} 例")
    print("- 低覆盖 case_id（前 40）: "
          + ", ".join(f"{c['case_id']}({len(c['objective'])})" for c in thin[:40]))

    if args.audit:
        strict = [c for c in sub_cases if c["_dx"]]
        print("\n" + "=" * 78)
        print(f"(b) 严判命中 {len(strict)} 例；宽网候选 "
              f"{len([c for c in sub_cases if c['_dx_broad']])} 例")
        print("\n### (b-1) 严判命中逐例（非报告语境 + 明确诊断断言）")
        for i, c in enumerate(strict, 1):
            print("-" * 66)
            print(f"[{i}] case_id={c['case_id']}\nGOLD: {c['gold']}")
            for f in c["_dx"][:4]:
                print(f"   > {f[:230]}")
        print("\n### (b-2) 人工抽查样本（含断言/推测语气的片段，非报告语境优先）")
        picked = [c for c in sub_cases if audit_candidates(c)]
        picked.sort(key=lambda c: -sum(
            1 for f, tg in audit_candidates(c, 6) if tg != "report"))
        step = max(1, len(picked) // args.audit)
        for i, c in enumerate(picked[::step][:args.audit], 1):
            print("-" * 66)
            print(f"[{i}] case_id={c['case_id']}\nGOLD: {c['gold']}\n"
                  f"严格(b)命中: {c['_dx'] or '无'}")
            for f, tag in audit_candidates(c, 4):
                print(f"   <{tag}> {f[:230]}")

    if args.show:
        for cid in args.show:
            c = next(x for x in sub_cases if x["case_id"] == cid)
            print(f"\n===== {cid} | GOLD: {c['gold']} | obj {len(c['objective'])} "
                  f"=====\n{c['objective']}")


if __name__ == "__main__":
    sys.exit(main())
