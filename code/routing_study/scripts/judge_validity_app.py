#!/usr/bin/env python3
"""Judge-validity 双人盲评服务。

启动：
    ./.venv/bin/uvicorn judge_validity_app:app --port 8001
（在 routing_study/scripts/ 目录下，或加 sys.path）

- 评者 A: http://localhost:8001/?who=a   评者 B: /?who=b
- 界面只显示 (gold, candidate) 与判定规则，不显示 LLM judge 的结论（盲评）
- 保存 routing_study/results/judge_validity/annotations_{a,b}.json
- 双人完成后 GET /report 出 Cohen's κ（AB 之间 + 各自 vs LLM judge）
"""
import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_study import cohens_kappa, annotator_progress, STUDY_DIR  # noqa: E402

# 样本集注册表：set 名 → (样本文件, 标注前缀)
SETS = {
    "v2": (STUDY_DIR / "sample.json", "annotations"),
    "v4check": (STUDY_DIR / "sample_v4check.json", "v4check_annotations"),
    "er": (STUDY_DIR / "sample_er.json", "er_annotations"),
    "mcr": (STUDY_DIR / "sample_mcr.json", "mcr_annotations"),
    "erb2": (STUDY_DIR / "sample_er_b2.json", "erb2_annotations"),
    "erb3": (STUDY_DIR / "sample_er_b3.json", "erb3_annotations"),
}


def _paths(set_name: str):
    if set_name not in SETS:
        raise HTTPException(400, f"set 必须是 {'/'.join(SETS)}")
    sample_path, prefix = SETS[set_name]
    if not sample_path.exists():
        raise HTTPException(404, f"样本 {sample_path.name} 不存在")
    return sample_path, prefix


def _load(sample_path: Path):
    return json.loads(sample_path.read_text())


def _marks(prefix: str, who: str) -> dict:
    p = STUDY_DIR / f"{prefix}_{who}.json"
    return json.loads(p.read_text()) if p.exists() else {}

app = FastAPI(title="Judge Validity 盲评", docs_url=None, redoc_url=None)
WHOS = ("a", "b")

RUBRIC = """判定规则（按顺序执行）：
1. 两者命名了同一种疾病 → 正确（同义词、缩写、翻译均可）。
2. 一方额外提及并发症、合并症或受累器官 → 不影响。
3. 缺少分期/严重度/分型等限定词 → 不影响。
4. 仅当主要疾病本身不同，或只给了从未落到该疾病的泛称 → 错误。

⚠ 任务要点：本任务判定的是「候选是否把诊断落在与金标准相同的疾病上」，
不是「候选是否为该表现的合理鉴别诊断」。操作测试：看候选的主诊断
（通常是 "causing / secondary to / presenting with" 之前的主名词）——
· 主诊断 = gold 本身，病因/并发症只作附加成分 → 正确
  （gold「嗜睡」vs「嗜睡，考虑甲减所致」→ 正确）；
· 主诊断 = 别的病，gold 只是被提到的症状或诱因 → 错误
  （gold「嗜睡」vs「严重甲减导致嗜睡」→ 错误；
   gold「鼻出血」vs「继发于鼻出血的吸入性肺炎」→ 错误）；
· gold 是低特异性类别、候选是该类别的具体形式（子类而非病因）→ 正确
  （gold「头部外伤」vs「隐匿性颅内出血」→ 正确；
   但 gold「脓肿」vs「坏死性筋膜炎」→ 错误，后者不是脓肿的子类）。"""

PAGE = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>Judge Validity 盲评（评者 __WHO__）</title>
<style>
 body{font-family:-apple-system,"PingFang SC",sans-serif;max-width:860px;margin:24px auto;padding:0 16px;color:#17242F;line-height:1.6}
 .card{border:1px solid #DCE4EA;border-radius:6px;padding:18px 22px;margin:14px 0;background:#fff}
 .gold{font-size:17px;font-weight:600} .cand{font-size:17px;margin-top:8px}
 .label{font-size:12px;color:#5B6B77;letter-spacing:.1em}
 .nav{display:flex;gap:8px;align-items:center;margin:14px 0}
 button{border:1px solid #145C74;background:#fff;color:#145C74;border-radius:4px;padding:8px 20px;font-size:15px;cursor:pointer}
 button.primary{background:#145C74;color:#fff}
 .prog{font-size:13px;color:#5B6B77;margin-left:auto}
 .verdict{font-size:14px;padding:6px 12px;border-radius:4px;display:inline-block;margin-right:8px}
 .ok{background:#EAF4EE;color:#1B7A4B;border:1px solid #BAD5C6}
 .bad{background:#F9EDE9;color:#B3402A;border:1px solid #E4B8AC}
 .rubric{font-size:13px;color:#5B6B77;background:#EEF3F6;border-radius:4px;padding:10px 14px}
 h3{margin-bottom:4px} .done{color:#1B7A4B}
</style></head><body>
<h3>Judge Validity 盲评 — 评者 __WHOUpper__ <span class="prog" id="prog"></span></h3>
<div class="rubric">__RUBRIC__</div>
<div class="nav">
 <button onclick="load_item()">载入下一待评条目</button>
 <span id="status" style="font-size:13px;color:#5B6B77"></span>
</div>
<div class="card" id="card" style="display:none">
 <div class="label">参考（金标准）诊断</div><div class="gold" id="gold_zh"></div>
 <details style="margin-top:2px"><summary style="font-size:12px;color:#5B6B77;cursor:pointer">英文原文</summary><div class="gold" id="gold" style="font-size:14px;font-weight:400;color:#5B6B77"></div></details>
 <div class="label" style="margin-top:14px">模型给出的诊断</div><div class="cand" id="cand_zh"></div>
 <details style="margin-top:2px"><summary style="font-size:12px;color:#5B6B77;cursor:pointer">英文原文</summary><div class="cand" id="cand" style="font-size:14px;color:#5B6B77"></div></details>
 <div style="margin-top:16px">
  <button class="ok verdict" onclick="submit(true)">正确（同一疾病）</button>
  <button class="bad verdict" onclick="submit(false)">错误（不同疾病）</button>
 </div>
</div>
<p style="font-size:13px;color:#5B6B77">共 __TOTAL__ 条。判定结果仅保存判定本身；你的标注与 LLM judge 的比对在双人完成后由 /report 自动生成。</p>
<script>
const WHO = "__WHO__";
const SET = "__SET__";
const $ = id => document.getElementById(id);
let cur = null;
async function refresh() {
  const r = await fetch(`/progress?who=${WHO}&set=${SET}`).then(x => x.json());
  $("prog").textContent = `已完成 ${r.done} / ${r.total}`;
}
async function load_item() {
  const r = await fetch(`/next?who=${WHO}&set=${SET}`).then(x => x.json());
  if (r.error) { $("status").textContent = r.error; $("card").style.display = "none"; return; }
  cur = r.item_id;
  $("gold_zh").textContent = r.gold_zh || r.gold;
  $("gold").textContent = r.gold;
  $("cand_zh").textContent = r.candidate_zh || r.candidate;
  $("cand").textContent = r.candidate;
  $("card").style.display = "block";
  $("status").textContent = "";
  refresh();
}
async function submit(v) {
  if (cur === null) return;
  await fetch("/annotate", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({who: WHO, set: SET, item_id: cur, verdict: v})});
  $("card").style.display = "none"; cur = null;
  refresh(); load_item();
}
refresh();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index(who: str = "a", set_name: str = Query("v2", alias="set")):
    sample_path, _ = _paths(set_name)
    sample = _load(sample_path)
    html = (PAGE.replace("__WHO__", who)
            .replace("__SET__", set_name)
            .replace("__WHOUpper__", who.upper())
            .replace("__RUBRIC__", RUBRIC)
            .replace("__TOTAL__", str(len(sample))))
    return HTMLResponse(html)


@app.get("/next")
def next_item(who: str = "a", set_name: str = Query("v2", alias="set")):
    sample_path, prefix = _paths(set_name)
    sample = _load(sample_path)
    marks = _marks(prefix, who)
    todo = next((t for t in sample if str(t["item_id"]) not in marks), None)
    if todo is None:
        return JSONResponse({"error": "全部条目已完成，感谢！"})
    return {"item_id": todo["item_id"], "gold": todo["gold"],
            "gold_zh": todo.get("gold_zh", ""),
            "candidate": todo["candidate"],
            "candidate_zh": todo.get("candidate_zh", "")}


@app.post("/annotate")
def annotate(payload: dict):
    who = payload.get("who")
    if who not in WHOS:
        raise HTTPException(400, "who 必须是 a 或 b")
    sample_path, prefix = _paths(payload.get("set", "v2"))
    sample = _load(sample_path)
    valid = {t["item_id"] for t in sample}
    item_id = payload.get("item_id")
    if item_id not in valid:
        raise HTTPException(400, "item_id 无效")
    marks = _marks(prefix, who)
    marks[str(item_id)] = bool(payload.get("verdict"))
    (STUDY_DIR / f"{prefix}_{who}.json").write_text(
        json.dumps(marks, ensure_ascii=False, indent=1))
    return {"ok": True, "done": len(marks)}


@app.get("/progress")
def progress(who: str = "a", set_name: str = Query("v2", alias="set")):
    _, prefix = _paths(set_name)
    total = len(_load(_paths(set_name)[0]))
    marks = _marks(prefix, who)
    return {"done": len(marks), "total": total}


@app.get("/report")
def report(set_name: str = Query("v2", alias="set")):
    sample_path, prefix = _paths(set_name)
    sample = _load(sample_path)
    llm = {t["item_id"]: t.get("llm_verdict") for t in sample}
    a = _marks(prefix, "a")
    b = _marks(prefix, "b")
    shared = sorted(set(a) & set(b), key=int)
    if not shared:
        return {"error": "两位评者还没有共同完成的条目",
                "progress": {w: {"done": len(_marks(prefix, w)),
                                 "total": len(sample)} for w in WHOS}}
    va = [a[i] for i in shared]
    vb = [b[i] for i in shared]
    vllm = [llm[int(i)] for i in shared]
    out = {
        "n_shared": len(shared),
        "kappa_AB": cohens_kappa(va, vb),
        "kappa_A_vs_LLM": cohens_kappa(va, vllm),
        "kappa_B_vs_LLM": cohens_kappa(vb, vllm),
        "majority_vs_LLM": cohens_kappa(
            [x or y for x, y in zip(va, vb)], vllm),
    }
    out["set"] = set_name
    (STUDY_DIR / f"kappa_report_{set_name}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    return out



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
