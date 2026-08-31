# -*- coding: utf-8 -*-
"""
【流水线 13/16】里程碑六 · Arena 竞技场可视化看板
运行位置：本机（秒级，不调 API）
输入：output/final_eval_results.json
输出：output/final_arena_dashboard.html（自包含，浏览器直接打开）
前置步骤：12　｜　后续步骤：—

Arena 竞技场可视化看板：读取 output/final_eval_results.json，生成自包含 HTML
（output/final_arena_dashboard.html），包含：
  - 核心指标卡片（SFT/DPO 均分、Win Rate、Elo、Bad Case）
  - 8 类分组柱状图、7 维雷达图、战绩堆叠条（纯 SVG，无外部依赖）
  - 对战详情浏览：默认匿名显示"回答A/回答B"（还原竞技场盲测），可一键揭示身份；
    支持按判定结果 / 类别 / Bad Case 筛选

用法（评分完成后，任何一侧均可，秒级完成、不调 API）：
    python scripts/build_arena_dashboard.py                       # 默认读 output/final_eval_results.json
    python scripts/build_arena_dashboard.py --input <路径> --output <路径>
"""
import argparse
import html
import json
import os
import time

BAD_THRESH = 10
NON_UNDERSTAND_DIMS = ["required_facts覆盖", "required_questions追问", "required_actions正确",
                       "prohibited_actions违规", "完整性自然度", "是否编造"]
DIMS = ["理解正确性", "required_facts覆盖", "required_questions追问", "required_actions正确",
        "prohibited_actions违规", "完整性自然度", "是否编造"]
COLOR_DPO, COLOR_SFT, COLOR_TIE = "#2e9e5b", "#e07b39", "#9aa5b1"

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Microsoft YaHei", system-ui, sans-serif; background: #f4f6f9; color: #1f2933; padding: 24px; }
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 22px; margin-bottom: 4px; }
h2 { font-size: 17px; margin: 28px 0 12px; border-left: 4px solid #2e9e5b; padding-left: 10px; }
.sub { color: #616e7c; font-size: 13px; margin-bottom: 18px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
.card { background: #fff; border-radius: 10px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.card .k { font-size: 12px; color: #616e7c; margin-bottom: 6px; }
.card .v { font-size: 22px; font-weight: 700; }
.card .v.good { color: #2e9e5b; } .card .v.warn { color: #e07b39; }
.card .d { font-size: 11px; color: #7b8794; margin-top: 4px; }
.panel { background: #fff; border-radius: 10px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.legend { font-size: 12px; color: #616e7c; margin: 6px 0 0; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 12px; }
.bar-label { font-size: 11px; fill: #52606d; }
.bar-value { font-size: 11px; font-weight: 700; }
.stack { display: flex; height: 18px; border-radius: 4px; overflow: hidden; margin: 3px 0 8px; }
.stack div { height: 100%; }
.stack-row { display: grid; grid-template-columns: 220px 1fr 110px; align-items: center; gap: 8px; font-size: 12px; }
.filters { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 12px; }
.filters button, .filters select { border: 1px solid #cbd2d9; background: #fff; border-radius: 6px;
  padding: 6px 12px; font-size: 13px; cursor: pointer; }
.filters button.on { background: #1f2933; color: #fff; border-color: #1f2933; }
.filters label { font-size: 13px; color: #52606d; margin-left: 8px; }
.count { font-size: 12px; color: #616e7c; margin-left: auto; }
.battle { background: #fff; border-radius: 10px; padding: 14px 16px; margin-bottom: 14px;
  box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.bhead { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 13px; margin-bottom: 8px; }
.badge { padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: 600; }
.badge.cat { background: #e4e7eb; color: #3e4c59; }
.badge.dpo { background: #d9f0e2; color: #1d6f3e; } .badge.sft { background: #fbe6d7; color: #9a4d12; }
.badge.tie { background: #e4e7eb; color: #52606d; }
.badge.bad { background: #fadbd8; color: #a93226; }
.badge.diff-up { background: #d9f0e2; color: #1d6f3e; } .badge.diff-down { background: #fadbd8; color: #a93226; }
.q { font-size: 13px; color: #3e4c59; background: #f8f9fa; border-radius: 6px; padding: 8px 10px; margin-bottom: 10px; }
.ans { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.ans .col { border: 1px solid #e4e7eb; border-radius: 8px; padding: 10px; }
.ans h4 { font-size: 13px; margin-bottom: 6px; }
.ans h4 .ident { padding: 1px 7px; border-radius: 8px; font-size: 11px; margin-left: 6px; }
.ident.dpo { background: #2e9e5b; color: #fff; } .ident.sft { background: #e07b39; color: #fff; }
.ans .txt { font-size: 13px; line-height: 1.7; white-space: pre-wrap; word-break: break-word; }
.ans .sc { font-size: 12px; color: #616e7c; margin-top: 8px; }
.arena-note { font-size: 12px; color: #616e7c; margin-top: 10px; }
.tc { background: #eef4ff; border: 1px solid #c3d4f5; border-radius: 4px; padding: 0 4px;
  font-family: Consolas, monospace; font-size: 12px; }
footer { margin-top: 30px; font-size: 12px; color: #9aa5b1; }
@media (max-width: 760px) { .ans { grid-template-columns: 1fr; } }
"""


def is_bad(rec: dict, model: str) -> bool:
    s = rec[model]["scores"]
    return s["总分"] <= BAD_THRESH or any(s[k] == 0 for k in NON_UNDERSTAND_DIMS)


def fmt_answer(text: str) -> str:
    out = html.escape(text or "（空）")
    return out.replace("&lt;tool_call&gt;", '<span class="tc">&lt;tool_call&gt;') \
              .replace("&lt;/tool_call&gt;", "&lt;/tool_call&gt;</span>")


def metric_cards(items: list) -> str:
    n = len(items)
    sft = sum(r["sft"]["scores"]["总分"] for r in items) / n
    dpo = sum(r["dpo"]["scores"]["总分"] for r in items) / n
    w = sum(1 for r in items if r["arena"]["verdict"] == "dpo")
    l = sum(1 for r in items if r["arena"]["verdict"] == "sft")
    t = n - w - l
    wr = w / n * 100
    a, b = w + 0.5 * t, l + 0.5 * t
    import math
    elo = 400 * math.log10(a / b) if a > 0 and b > 0 else (800.0 if b <= 0 else -800.0)
    bad_sft = sum(1 for r in items if is_bad(r, "sft"))
    bad_dpo = sum(1 for r in items if is_bad(r, "dpo"))
    delta = dpo - sft
    cards = [
        ("SFT 总均分", f"{sft:.2f}<span style='font-size:13px;color:#616e7c'> /14</span>", f"Bad Case {bad_sft}", ""),
        ("DPO 总均分", f"{dpo:.2f}<span style='font-size:13px;color:#616e7c'> /14</span>", f"Bad Case {bad_dpo}", ""),
        ("均分差 Δ", f"{delta:+.2f}", "DPO − SFT", "good" if delta > 0 else ("warn" if delta < 0 else "")),
        ("竞技场 DPO 胜率", f"{wr:.1f}<span style='font-size:13px;color:#616e7c'>%</span>",
         f"胜 {w} / 负 {l} / 平 {t}", "good" if wr >= 50 else "warn"),
        ("Elo 分差", f"{elo:+.0f}", f"SFT 锚定 1200 → DPO ≈ {1200 + elo:.0f}", "good" if elo >= 0 else "warn"),
        ("逐条对比", "", "", ""),  # 占位，下面替换
    ]
    up = sum(1 for r in items if r["score_diff"] >= 2)
    down = sum(1 for r in items if r["score_diff"] <= -2)
    cards[5] = ("明显更好/更差", f"<span style='color:#2e9e5b'>{up}</span> / "
                                f"<span style='color:#a93226'>{down}</span>",
                "分差 ≥+2 / ≤−2（共 {} 条）".format(n), "")
    return "<div class='cards'>" + "".join(
        f"<div class='card'><div class='k'>{k}</div><div class='v {c}'>{v}</div><div class='d'>{d}</div></div>"
        for k, v, d, c in cards) + "</div>"


def cat_bars(items: list) -> str:
    by: dict = {}
    for r in items:
        by.setdefault(r["category_id"], []).append(r)
    cids = sorted(by)
    W, H, TOP, GAP = 900, 40 + 44 * len(cids), 26, 44
    rows = []
    for i, cid in enumerate(cids):
        rs = by[cid]
        sft = sum(r["sft"]["scores"]["总分"] for r in rs) / len(rs)
        dpo = sum(r["dpo"]["scores"]["总分"] for r in rs) / len(rs)
        y = TOP + i * GAP
        for j, (val, color, name) in enumerate([(sft, COLOR_SFT, "SFT"), (dpo, COLOR_DPO, "DPO")]):
            bw = max(val / 14 * 640, 2)
            rows.append(f"<rect x='170' y='{y + j * 15}' width='{bw:.1f}' height='12' rx='2' fill='{color}'/>")
            rows.append(f"<text x='{170 + bw + 6:.1f}' y='{y + j * 15 + 10}' class='bar-value' fill='{color}'>"
                        f"{val:.2f}</text>")
        rows.append(f"<text x='160' y='{y + 20}' text-anchor='end' class='bar-label'>cat{cid} "
                    f"{html.escape(rs[0]['category'][:10])}</text>")
        d = dpo - sft
        rows.append(f"<text x='830' y='{y + 20}' class='bar-label' fill='{'#1d6f3e' if d > 0 else ('#a93226' if d < 0 else '#7b8794')}'>"
                    f"Δ{d:+.2f}</text>")
    svg = (f"<svg viewBox='0 0 {W} {H}' width='100%' xmlns='http://www.w3.org/2000/svg'>"
           + "".join(rows) + "</svg>")
    legend = (f"<p class='legend'><i style='background:{COLOR_SFT}'></i>SFT 基线"
              f"<i style='background:{COLOR_DPO}'></i>DPO 模型（数字为均分/14，右列为 Δ）</p>")
    return f"<div class='panel'>{svg}{legend}</div>"


def radar(items: list) -> str:
    import math
    cx, cy, R = 210, 150, 105
    n_dim = len(DIMS)
    means = {}
    for m in ("sft", "dpo"):
        means[m] = [sum(r[m]["scores"][k] for r in items) / (2 * len(items)) for k in DIMS]
    pt = lambda i, v: (cx + R * v * math.sin(2 * math.pi * i / n_dim),
                       cy - R * v * math.cos(2 * math.pi * i / n_dim))
    rings, axes, polys = [], [], []
    for g in (0.5, 1.0, 1.5, 2.0):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, g) for i in range(n_dim)))
        rings.append(f"<polygon points='{pts}' fill='none' stroke='#e4e7eb' stroke-width='1'/>")
    for i, dim in enumerate(DIMS):
        x, y = pt(i, 1.0)
        axes.append(f"<line x1='{cx}' y1='{cy}' x2='{x:.1f}' y2='{y:.1f}' stroke='#e4e7eb'/>")
        lx, ly = pt(i, 1.28)
        anchor = "middle" if abs(lx - cx) < 10 else ("start" if lx > cx else "end")
        short = {"理解正确性": "理解", "required_facts覆盖": "facts", "required_questions追问": "追问",
                 "required_actions正确": "动作", "prohibited_actions违规": "无违规",
                 "完整性自然度": "完整", "是否编造": "无编造"}[dim]
        axes.append(f"<text x='{lx:.1f}' y='{ly:.1f}' text-anchor='{anchor}' class='bar-label'>{short}</text>")
    for m, color in (("sft", COLOR_SFT), ("dpo", COLOR_DPO)):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, means[m][i]) for i in range(n_dim)))
        polys.append(f"<polygon points='{pts}' fill='{color}' fill-opacity='0.25' stroke='{color}' stroke-width='2'/>")
    svg = (f"<svg viewBox='0 0 420 310' width='100%' style='max-width:460px' xmlns='http://www.w3.org/2000/svg'>"
           + "".join(rings + axes + polys) + "</svg>")
    legend = (f"<p class='legend'><i style='background:{COLOR_SFT}'></i>SFT 基线"
              f"<i style='background:{COLOR_DPO}'></i>DPO 模型（7 维得分率，满分 2/维）</p>")
    return f"<div class='panel'>{svg}{legend}</div>"


def battle_stacks(items: list) -> str:
    def row(label, rs):
        n = len(rs)
        w = sum(1 for r in rs if r["arena"]["verdict"] == "dpo")
        l = sum(1 for r in rs if r["arena"]["verdict"] == "sft")
        t = n - w - l
        seg = (f"<div style='width:{w / n * 100:.2f}%;background:{COLOR_DPO}'></div>"
               f"<div style='width:{l / n * 100:.2f}%;background:{COLOR_SFT}'></div>"
               f"<div style='width:{t / n * 100:.2f}%;background:{COLOR_TIE}'></div>")
        return f"<div class='stack-row'><span>{html.escape(label)}</span>" \
               f"<div class='stack'>{seg}</div><span>胜{w} 负{l} 平{t}</span></div>"
    by: dict = {}
    for r in items:
        by.setdefault(r["category_id"], []).append(r)
    rows = [row(f"总体（{len(items)} 条）", items)]
    for cid in sorted(by):
        rows.append(row(f"cat{cid} {by[cid][0]['category'][:10]}（{len(by[cid])} 条）", by[cid]))
    legend = (f"<p class='legend'><i style='background:{COLOR_DPO}'></i>DPO 胜"
              f"<i style='background:{COLOR_SFT}'></i>SFT 胜<i style='background:{COLOR_TIE}'></i>平局</p>")
    return f"<div class='panel'>{ ''.join(rows) }{legend}</div>"


def battle_cards(items: list, full: bool = True) -> str:
    cards = []
    for r in sorted(items, key=lambda x: (x["category_id"], x["seed_id"])):
        verdict = r["arena"]["verdict"]
        v_badge = {"dpo": "<span class='badge dpo'>竞技场：DPO 胜</span>",
                   "sft": "<span class='badge sft'>竞技场：SFT 胜</span>",
                   "tie": "<span class='badge tie'>竞技场：平局</span>"}[verdict]
        reason = "；".join(x for x in r["arena"].get("reasons", []) if x) or "—"
        if not full:  # 简化模式：无逐模型分数/回答，只显示判定与判词
            cards.append(
                f"<div class='battle' data-verdict='{verdict}' data-bad='0' data-cat='{r['category_id']}'>"
                f"<div class='bhead'><span class='badge cat'>cat{r['category_id']} {html.escape(r['category'])}"
                f"/{html.escape(r['subcategory'])}</span><span>{r['seed_id']}</span>{v_badge}</div>"
                f"<div class='q'>问题：{html.escape(r['question'])}</div>"
                f"<div class='arena-note'>判定：{r['arena'].get('method', '')}；判词：{html.escape(reason)}</div>"
                f"</div>")
            continue
        bad = is_bad(r, "dpo")
        diff = r["score_diff"]
        diff_badge = (f"<span class='badge diff-up'>Δ{diff:+d}</span>" if diff > 0 else
                      f"<span class='badge diff-down'>Δ{diff:+d}</span>" if diff < 0 else
                      "<span class='badge tie'>Δ0</span>")
        bad_badge = "<span class='badge bad'>DPO Bad Case</span>" if bad else ""
        dpo_is_a = r["arena"].get("dpo_is_a")
        id_a = {True: "DPO", False: "SFT", None: "?"}[dpo_is_a]
        id_b = {True: "SFT", False: "DPO", None: "?"}[dpo_is_a]
        reason = "；".join(x for x in r["arena"].get("reasons", []) if x) or "—"

        def col(tag, ident, model):
            s = r[model]["scores"]
            fl = r[model]["flags"]
            marks = []
            for name, label in [("安全劝阻缺失", "缺安全劝阻"), ("救援确认缺失", "缺救援确认"),
                                ("tool_call缺失", "缺tool_call"), ("半截话", "半截话")]:
                if fl.get(name):
                    marks.append(label)
            if s["是否编造"] < 2:
                marks.append("编造疑点")
            mark_html = f"<div class='sc'>缺陷标记：{'、'.join(marks)}</div>" if marks else ""
            judge_tag = "（Judge）" if fl.get("judge") else "（规则分）"
            return (f"<div class='col'><h4>{tag}<span class='ident {model.lower()} hidden-ident'>{ident}</span></h4>"
                    f"<div class='txt'>{fmt_answer(r[model]['answer'])}</div>"
                    f"<div class='sc'>得分 {s['总分']}/14{judge_tag}</div>{mark_html}</div>")

        cards.append(
            f"<div class='battle' data-verdict='{verdict}' data-bad='{1 if bad else 0}' data-cat='{r['category_id']}'>"
            f"<div class='bhead'><span class='badge cat'>cat{r['category_id']} {html.escape(r['category'])}"
            f"/{html.escape(r['subcategory'])}</span><span>{r['seed_id']}</span>"
            f"<span class='badge dpo'>DPO {r['dpo']['scores']['总分']}/14</span>"
            f"<span class='badge sft'>SFT {r['sft']['scores']['总分']}/14</span>{diff_badge}{v_badge}{bad_badge}</div>"
            f"<div class='q'>问题：{html.escape(r['question'])}</div>"
            f"<div class='ans'>{col('回答A', id_a, 'dpo' if dpo_is_a else 'sft')}"
            f"{col('回答B', id_b, 'sft' if dpo_is_a else 'dpo')}</div>"
            f"<div class='arena-note'>竞技场判定：{r['arena'].get('method', '')}；判词：{html.escape(reason)}</div>"
            f"</div>")
    return "\n".join(cards)


SCRIPT = """
(function(){
  var verdict='all', cat='all', bad=false;
  var count=document.getElementById('count');
  function apply(){
    var shown=0;
    document.querySelectorAll('.battle').forEach(function(el){
      var ok=(verdict==='all'||el.dataset.verdict===verdict)&&
             (cat==='all'||el.dataset.cat===cat)&&
             (!bad||el.dataset.bad==='1');
      el.style.display=ok?'':'none'; if(ok)shown++;
    });
    count.textContent='显示 '+shown+' / '+total+' 条对战';
  }
  var total=document.querySelectorAll('.battle').length;
  document.querySelectorAll('[data-f]').forEach(function(b){
    b.addEventListener('click',function(){
      document.querySelectorAll('[data-f]').forEach(function(x){x.classList.remove('on')});
      b.classList.add('on'); verdict=b.dataset.f; apply();
    });
  });
  var sel=document.getElementById('catSel');
  sel.addEventListener('change',function(){cat=sel.value;apply();});
  var bk=document.getElementById('badBtn');
  if(bk){bk.addEventListener('click',function(){bad=!bad;bk.classList.toggle('on',bad);apply();});}
  var rev=document.getElementById('reveal');
  function ident(){var r=document.getElementById('reveal');
    document.querySelectorAll('.hidden-ident').forEach(function(s){
      s.style.visibility=(r&&r.checked)?'visible':'hidden';});}
  if(rev){rev.addEventListener('change',ident);} ident(); apply();
})();
"""


def build(items: list, out_path: str) -> str:
    n = len(items)
    w = sum(1 for r in items if r["arena"]["verdict"] == "dpo")
    l = sum(1 for r in items if r["arena"]["verdict"] == "sft")
    t = n - w - l
    cats = sorted({r["category_id"] for r in items})
    full = "sft" in items[0] and "answer" in items[0]["sft"]  # 有逐模型分数/回答 = 完整模式
    cat_opts = "<option value='all'>全部类别</option>" + "".join(
        f"<option value='{cid}'>cat{cid} {html.escape(next(r['category'] for r in items if r['category_id'] == cid)[:12])}</option>"
        for cid in cats)
    title = "Arena 竞技场看板" if not full else "final_test 终局评估 · Arena 竞技场看板"
    sub = ("validation 64 条 · 约束版（DPO R2 + 约束层）vs 基线（DPO R2 裸）· 盲测双向对调复审"
           if not full else
           "SFT 基线（sft_model_merged）vs DPO（sft_model_merged + dpo_model_r2）　|　"
           "160 条 final_test · 盲测（评审只见\"回答A/B\"）· 随机站位 + 双向对调复审")
    mid = (f"<h2>竞技场战绩（Win/Tie/Loss）</h2>\n{battle_stacks(items)}"
           if not full else
           f"{metric_cards(items)}\n<h2>8 类分组得分对比</h2>\n{cat_bars(items)}\n"
           f"<h2>7 维得分雷达图</h2>\n{radar(items)}\n<h2>竞技场战绩（Win/Tie/Loss）</h2>\n{battle_stacks(items)}")
    doc = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>{title}</h1>
<p class="sub">{sub}　|　生成时间 {time.strftime('%Y-%m-%d %H:%M')}</p>
{mid}
<h2>对战详情（{n} 场，默认匿名；评审判词见卡片底部）</h2>
<div class="filters">
<button class="on" data-f="all">全部</button><button data-f="dpo">DPO 胜</button>
<button data-f="sft">SFT 胜</button><button data-f="tie">平局</button>
{('<button id="badBtn">仅 DPO Bad Case</button>' if full else '')}
<select id="catSel">{cat_opts}</select>
{('<label><input type="checkbox" id="reveal"> 揭示 A/B 身份</label>' if full else '')}
<span class="count" id="count"></span>
</div>
{battle_cards(items, full)}
<footer>数据来源：{os.path.basename(__file__)} 的输入文件（由 run_final_eval.py / run_arena_validation.py 产出）；
重新生成看板：python scripts/build_arena_dashboard.py --input 结果文件 --output html。纯静态页面，可离线打开。</footer>
</div><script>{SCRIPT}</script></body></html>"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output/final_eval_results.json")
    ap.add_argument("--output", default="output/final_arena_dashboard.html")
    args = ap.parse_args()
    src = args.input if os.path.exists(args.input) else None
    if src is None:
        # 兼容实例/本机的路径差异
        for cand in ("output/final_eval_results.json", "output/selftest/final_eval_results.json"):
            if os.path.exists(cand):
                src = cand
                break
    assert src, f"找不到评估明细文件: {args.input}（请先运行 scripts/run_final_eval.py --judge）"
    with open(src, encoding="utf-8") as f:
        items = json.load(f)
    out = build(items, args.output)
    w = sum(1 for r in items if r["arena"]["verdict"] == "dpo")
    l = sum(1 for r in items if r["arena"]["verdict"] == "sft")
    print(f"[看板] {len(items)} 场对战（DPO 胜 {w} / SFT 胜 {l} / 平 {len(items) - w - l}）")
    print(f"[看板] 已生成 {out}（浏览器直接打开；JupyterLab 里双击该文件也可预览）")


if __name__ == "__main__":
    main()
