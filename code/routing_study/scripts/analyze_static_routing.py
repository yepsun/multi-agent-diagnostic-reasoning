#!/usr/bin/env python3
"""方案一分析：四策略（准确率, 成本）表 + P 救回率 → markdown 报告。"""
import sys, os, json, statistics
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "results", "static_routing_raw.jsonl")
REPORT = os.path.join(HERE, "..", "results", "static_routing_report.md")

# USD per 1M tokens (in, out)
PRICES = {
    "qwen": (0.05, 0.40),        # 占位价，主实验前按官方牌价覆盖
    "deepseek-flash": (0.27, 1.10),    # deepseek-v4-pro 牌价按需更新
}
IN_OUT_RATIO = 0.8  # in 占 total 的比例（近似）


def cost_usd(tokens: int, provider: str) -> float:
    p_in, p_out = PRICES[provider]
    tin = tokens * IN_OUT_RATIO
    tout = tokens * (1 - IN_OUT_RATIO)
    return (tin * p_in + tout * p_out) / 1e6


def strategy_table(rows):
    n = len(rows)
    t = {}
    t["all_a"] = dict(
        accuracy=sum(r["a_correct"] for r in rows) / n,
        total_tokens=sum(r["a_tokens"] for r in rows),
    )
    t["all_p"] = dict(
        accuracy=sum(r["p_correct"] for r in rows) / n,
        total_tokens=sum(r["p_tokens"] for r in rows),
    )
    t["all_p_qwen"] = dict(
        accuracy=sum(r["p_qwen_correct"] for r in rows) / n,
        total_tokens=sum(r["p_qwen_tokens"] for r in rows),
    )
    # oracle：a 对走 a，a 错时若 p 对则走 p（成本仍计 p）；p 也错则钱照花、结果同 A
    tok = 0
    correct = 0
    for r in rows:
        if r["a_correct"]:
            tok += r["a_tokens"]; correct += 1
        elif r["p_correct"]:
            tok += r["p_tokens"]; correct += 1
        else:
            # a 错时即使 p 也错，仍按已调用 P 计费
            tok += r["p_tokens"]
    t["oracle"] = dict(accuracy=correct / n, total_tokens=tok)
    # 随机路由（50/50）：期望准确率 = 0.5*acc_A + 0.5*acc_P
    t["random_50"] = dict(
        accuracy=0.5 * t["all_a"]["accuracy"] + 0.5 * t["all_p"]["accuracy"],
        total_tokens=int(0.5 * t["all_a"]["total_tokens"] + 0.5 * t["all_p"]["total_tokens"]),
    )
    # 换算成本
    for name, prov in (("all_a", "qwen"), ("all_p", "deepseek-flash"),
                       ("all_p_qwen", "qwen"), ("random_50", "deepseek-flash")):
        t[name]["avg_cost_usd"] = cost_usd(t[name]["total_tokens"] / max(n, 1), prov)
    t["oracle"]["avg_cost_usd"] = statistics.mean([
        cost_usd(r["a_tokens"], "qwen") if r["a_correct"]
        else cost_usd(r["p_tokens"], "deepseek-flash") for r in rows
    ])
    return t


def rescue_rate(rows):
    wrong = [r for r in rows if not r["a_correct"]]
    rp = sum(1 for r in wrong if r["p_correct"])
    rq = sum(1 for r in wrong if r["p_qwen_correct"])
    return {"a_wrong_n": len(wrong), "rescued_p": rp, "rescued_p_qwen": rq,
            "rate_p": rp / len(wrong) if wrong else 0.0,
            "rate_p_qwen": rq / len(wrong) if wrong else 0.0}


def main():
    rows = [json.loads(l) for l in open(RAW) if l.strip()]
    t = strategy_table(rows)
    r = rescue_rate(rows)
    lines = [
        "# 方案一：静态路由验证报告",
        f"日期：{date.today().isoformat()}；N={len(rows)} 病例（CPC/MGH QA 集）",
        "",
        "| 策略 | 准确率 | 总 tokens | 平均成本/case (USD) |",
        "|---|---|---|---|",
    ]
    label = {"all_a": "全 A (qwen3.8-flash×5)", "all_p": "全 P (deepseek-v4-pro)",
             "all_p_qwen": "全 P' (qwen 消融)", "oracle": "Oracle 路由上限",
             "random_50": "随机路由 50/50"}
    for k in ("all_a", "all_p", "all_p_qwen", "random_50", "oracle"):
        v = t[k]
        lines.append(f"| {label[k]} | {v['accuracy']:.1%} | {v['total_tokens']:,} | ${v['avg_cost_usd']:.5f} |")
    lines += [
        "",
        "## P 救回率（go/no-go 决策点）",
        f"- A 错 {r['a_wrong_n']}/{len(rows)} 题",
        f"- deepseek-v4-pro 救回 {r['rescued_p']}（{r['rate_p']:.0%}）",
        f"- qwen 消融救回 {r['rescued_p_qwen']}（{r['rate_p_qwen']:.0%}）",
        "",
        "**决策规则**：rate_p ≥ 50% → go（进入方案二）；< 50% → 路由上限不足，早停。",
        "",
        "## 附注",
        "- 一致率分布与 a_correct 的交叉表见原始 JSONL（方案二信号分析的输入）",
        "- 成本为牌价估算，token 数是硬数据",
    ]
    open(REPORT, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
