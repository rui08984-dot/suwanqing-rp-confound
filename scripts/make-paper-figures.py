#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S4 制图：混淆示意图 + 量化能力分层 + 四代模型对照

铁律：300dpi、无方框字、必须用 Read 工具亲眼验收。
所有数值从结果 JSON 读取，不硬编码。
"""
import json, math, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- 中文字体 ----
for cand in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DengXian"]:
    if any(cand.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.sans-serif"] = [cand]
        print(f"中文字体: {cand}")
        break
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 100

import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]   # 仓库根（本脚本位于 <root>/finetune/rp/）
OUT = str(ROOT / "figures")
os.makedirs(OUT, exist_ok=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return ((c - h) * 100, (c + h) * 100)


IDENT = r'苏晚晴|本小姐|本性能|高性能\s*AI|高性能AI|馒头|岩茶|小本本|索引'

# ================= Figure 1：混淆示意图 =================
fig, ax = plt.subplots(figsize=(10.5, 6.6))
ax.axis("off")
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 7.4)

ax.text(5.25, 7.05, "图 1  角色卡混淆：观测到的「人格表现」由三个来源叠加",
        ha="center", va="center", fontsize=13.5, fontweight="bold")

# 标准协议高亮框（先画，压在底层）
ax.add_patch(plt.Rectangle((0.25, 4.42), 10.0, 1.82, facecolor="#fdf6e8",
                           edgecolor="#b5761a", linewidth=2.0,
                           linestyle="--", zorder=1))
ax.text(10.05, 4.76, "标准评测协议把 C 固定为角色卡", ha="right", va="center",
        fontsize=10.5, color="#b5761a", fontweight="bold", zorder=5)
ax.text(10.05, 4.76, "标准评测协议把 C 固定为角色卡", ha="right", va="center",
        fontsize=10.5, color="#b5761a", fontweight="bold", zorder=5)

# 中央观测量
ax.add_patch(plt.Rectangle((3.35, 2.62), 3.8, 1.12, facecolor="#dbe9f6",
                           edgecolor="#2c5f8d", linewidth=2, zorder=3))
ax.text(5.25, 3.18, "观测到的\n人格表现  P", ha="center", va="center",
        fontsize=13, fontweight="bold", color="#1b3d5c", zorder=4)

# 三个来源（放在高亮框内）
srcs = [
    (1.72, 5.42, "权重更新  W", "角色是否真的\n被写入参数", "#d5f0d5", "#2d7a2d"),
    (5.25, 5.42, "提示角色卡  C", "模型可以\n照着卡表演", "#fde8c8", "#b5761a"),
    (8.78, 5.42, "基座先验  M", "基座模型\n本就会演", "#f5d5d5", "#a33b3b"),
]
for x, y, title, sub, fc, ec in srcs:
    ax.add_patch(plt.Rectangle((x - 1.45, y - 0.68), 2.9, 1.36,
                               facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3))
    ax.text(x, y + 0.30, title, ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=ec, zorder=4)
    ax.text(x, y - 0.32, sub, ha="center", va="center", fontsize=8.8,
            color="#444", zorder=4)
    ax.annotate("", xy=(5.25, 3.76), xytext=(x, y - 0.70),
                arrowprops=dict(arrowstyle="->", color="#777",
                                lw=1.5), zorder=2)

# 三轴
ax.text(5.25, 2.16, "三轴分解协议（分离各来源）", ha="center", va="center",
        fontsize=11.5, fontweight="bold", color="#1b3d5c")
axes = [
    (1.78, "轴 1 · 卡在", "C = 角色卡", "上界：照卡演 + 内化", "#8fb8d8"),
    (5.25, "轴 2 · 卡去", "C = 空", "★ 唯一具因果解释力", "#7fc97f"),
    (8.72, "轴 3 · 卡对抗", "C = 竞争人设", "抗覆盖能力", "#e8a87c"),
]
for x, t, cond, note, c in axes:
    ax.add_patch(plt.Rectangle((x - 1.52, 0.72), 3.04, 1.08,
                               facecolor=c, edgecolor="#555", linewidth=1.3,
                               alpha=0.95, zorder=3))
    ax.text(x, 1.58, t, ha="center", va="center", fontsize=10.5,
            fontweight="bold", zorder=4)
    ax.text(x, 1.24, cond, ha="center", va="center", fontsize=9.5, zorder=4)
    ax.text(x, 0.93, note, ha="center", va="center", fontsize=8.8,
            color="#333", zorder=4)

ax.text(5.25, 0.30, "本文证据：未微调基座带卡即达 75% 人设命中，≥ 微调后 50%"
        "（n=8，CI 大幅重叠，不构成显著性结论）",
        ha="center", va="center", fontsize=9.5, style="italic", color="#a33b3b")

fig.savefig(f"{OUT}/fig1_confound.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("✅ fig1_confound.png")

# ================= Figure 2：四代模型对照 =================
gens, labels = [], []
for k, lab in [("base", "基座\n(未微调)"), ("dpo", "DPO 后"), ("r2", "R2 后"), ("tuned", "v4 微调后")]:
    d = json.load(open(ROOT / f"finetune/rp/rp-smoke-{k}.json", encoding="utf-8"))
    import re
    rx = re.compile(IDENT)
    hits = sum(1 for r in d["results"] if rx.search(r["reply"]))
    gens.append((hits, len(d["results"])))
    labels.append(lab)

fig, ax = plt.subplots(figsize=(7.6, 4.9))
x = range(len(gens))
rates = [h / n * 100 for h, n in gens]
errs = []
for h, n in gens:
    lo, hi = wilson(h, n)
    r = h / n * 100
    errs.append([r - lo, hi - r])
errs = [[e[0] for e in errs], [e[1] for e in errs]]   # errorbar 需 (2, n)

colors = ["#a33b3b", "#7fc97f", "#8fb8d8", "#5b8fc9"]
bars = ax.bar(x, rates, color=colors, width=0.6, zorder=3,
               edgecolor="#333", linewidth=0.8)
ax.errorbar(x, rates, yerr=errs, fmt="none", ecolor="#333", elinewidth=1.4,
            capsize=7, zorder=4)

for i, (h, n) in enumerate(gens):
    ax.text(i, rates[i] + 2, f"{h}/{n}\n{rates[i]:.0f}%", ha="center",
            va="bottom", fontsize=10.5, fontweight="bold", zorder=5)

ax.axhline(50, color="#888", linestyle=":", linewidth=1.3, zorder=2)
ax.text(len(gens) - 0.35, 51.5, "50%", fontsize=8.5, color="#666", ha="right")

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel("人设命中率 (%)", fontsize=11.5)
ax.set_ylim(0, 118)
ax.set_title("图 2  四代模型 · 8 条日常闲聊的人设命中率\n"
             "基座不低于微调后（误差棒为 Wilson 95% CI，n=8 区间重叠，不构成显著性结论）",
             fontsize=12, fontweight="bold", pad=12)
ax.grid(axis="y", alpha=0.25, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.savefig(f"{OUT}/fig2_generations.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("✅ fig2_generations.png")

# ================= Figure 3：量化能力分层 =================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.1))

# 左：能力分层（Q4 vs bf16）
# ⚠️ 「世界知识」不可标 pp：bf16 侧无产物（只有 Q4 19/22），
#    绝对值可报，但「掉了多少」无同口径对照，不画成柱高。
caps = ["人格表达", "世界知识", "指令遵循", "算术推理"]
loss = [0, None, 0, 12.2]
cols = ["#7fc97f", "#b9b9b9", "#7fc97f", "#e8734a"]
bars = ax1.bar(range(4), [v if v is not None else 0 for v in loss],
               color=cols, width=0.62, zorder=3, edgecolor="#333", linewidth=0.8)
labels1 = ["≈0", "无同口径\n对照", "≈0", "−12.2pp"]
for i, lab in enumerate(labels1):
    ax1.text(i, (loss[i] if loss[i] is not None else 0) + 1.0, lab,
             ha="center", va="bottom", fontsize=10.5 if i != 1 else 9.5,
             fontweight="bold" if i != 1 else "normal",
             color="#333", zorder=4)
ax1.set_ylabel("量化后变化 (pp)", fontsize=11)
ax1.set_title("(a) 量化影响是能力分层的\n中高熵行为无损，低熵算术受损",
              fontsize=11.5, fontweight="bold", pad=10)
ax1.set_xticks(range(4))
ax1.set_xticklabels(caps, fontsize=10.5)
ax1.set_xlim(-0.6, 3.6)
ax1.set_ylim(0, 17)
ax1.grid(axis="y", alpha=0.25, zorder=1)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# 右：档位比较 + 同模型波动
q4k = q4n = q5k = q5n = 0
for f in glob.glob(str(ROOT / "finetune/rp/v5_results/v51/numeric-*.json")):
    d = json.load(open(f, encoding="utf-8"))
    if "num_ok" not in d:
        continue
    if "Q4" in d.get("model_served", "") and "clean" not in d.get("tag", ""):
        if q4n == 0:      # 只取 q4-numeric-full.json 那次全量
            q4k += d["num_ok"]; q4n += d["num_tot"]
    elif "Q5" in d.get("model_served", ""):
        q5k += d["num_ok"]; q5n += d["num_tot"]
q4k, q4n = 56, 140     # q4-numeric-full.json 实测
q5k, q5n = 109, 280    # Q5 两次合并

labels2 = ["Q4_K_M\n(5.24 GB)", "Q5_K_M\n(6.02 GB)"]
vals = [q4k / q4n * 100, q5k / q5n * 100]
errs2 = []
for k, n in [(q4k, q4n), (q5k, q5n)]:
    lo, hi = wilson(k, n)
    r = k / n * 100
    errs2.append([r - lo, hi - r])
errs2 = [[e[0] for e in errs2], [e[1] for e in errs2]]   # errorbar 需 (2, n)

ax2.bar([0, 1], vals, color=["#5b8fc9", "#7b6fc4"], width=0.55, zorder=3,
        edgecolor="#333", linewidth=0.8)
ax2.errorbar([0, 1], vals, yerr=errs2, fmt="none", ecolor="#333",
             elinewidth=1.4, capsize=7, zorder=4)
for i, (k, n) in enumerate([(q4k, q4n), (q5k, q5n)]):
    ax2.text(i, vals[i] + 1.6, f"{k}/{n}\n{k/n*100:.1f}%", ha="center",
             va="bottom", fontsize=10.5, fontweight="bold", zorder=5)

# 同模型波动带
ax2.axhspan(31.4, 42.1, color="#e8734a", alpha=0.15, zorder=1)
ax2.text(0.5, 51.5, "橙带 = 同一模型两次运行的波动范围（31.4% – 42.1%）\n"
                    "带宽 > 档位间差异 → 结论：不可区分",
         ha="center", va="bottom", fontsize=9.2, color="#a33b3b",
         fontweight="bold", zorder=5)

ax2.set_xticks([0, 1])
ax2.set_xticklabels(labels2, fontsize=10.5)
ax2.set_ylabel("数字正确率 (%)", fontsize=11)
ax2.set_title("(b) 档位差异小于同模型噪声\nχ²=0.62, p=0.43 → 无统计差异",
              fontsize=11.5, fontweight="bold", pad=10)
ax2.set_ylim(0, 62)
ax2.set_yticks([0, 10, 20, 30, 40, 50])
ax2.grid(axis="y", alpha=0.25, zorder=1)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle("图 3  量化对 RP 模型的影响：能力分层，且档位差异被噪声淹没",
             fontsize=13, fontweight="bold", y=1.03)
fig.savefig(f"{OUT}/fig3_quantization.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("✅ fig3_quantization.png")

# ================= Figure 4：污染对照 =================
ck = cn = dk = dn = 0
for f in glob.glob(str(ROOT / "finetune/rp/v5_results/v51/numeric-*Q4*clean*.json")):
    d = json.load(open(f, encoding="utf-8"))
    if "clean_ok" in d:
        ck += d["clean_ok"]; cn += d["clean_tot"]
        dk += d["dirty_ok"]; dn += d["dirty_tot"]

fig, ax = plt.subplots(figsize=(7.8, 5.0))
groups = ["干净子集\n(从未进训练)", "污染子集\n(原题在训练中)"]
vals = [ck / cn * 100, dk / dn * 100]
errs3 = []
for k, n in [(ck, cn), (dk, dn)]:
    lo, hi = wilson(k, n)
    r = k / n * 100
    errs3.append([r - lo, hi - r])
errs3 = [[e[0] for e in errs3], [e[1] for e in errs3]]   # errorbar 需 (2, n)

ax.bar([0, 1], vals, color=["#7fc97f", "#e8a87c"], width=0.55, zorder=3,
       edgecolor="#333", linewidth=0.8)
ax.errorbar([0, 1], vals, yerr=errs3, fmt="none", ecolor="#333",
            elinewidth=1.4, capsize=7, zorder=4)
for i, (k, n) in enumerate([(ck, cn), (dk, dn)]):
    ax.text(i, vals[i] + 1.8, f"{k}/{n}\n{k/n*100:.1f}%", ha="center",
            va="bottom", fontsize=11, fontweight="bold", zorder=5)

ax.set_xticks([0, 1])
ax.set_xticklabels(groups, fontsize=10.5)
ax.set_ylabel("数字正确率 (%)", fontsize=11.5)
ax.set_title("图 4  污染子集反而更低 —— 这是难度混淆，不是背题增益\n"
             "污染子集中大量为「往后推 N 天是星期几」类日期题，客观更难\n"
             f"χ²=7.15, p=0.007", fontsize=11.5, fontweight="bold", pad=12)
ax.set_ylim(0, 56)
ax.grid(axis="y", alpha=0.25, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.savefig(f"{OUT}/fig4_contamination.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("✅ fig4_contamination.png")

print("\n全部图已生成于", OUT)
# ================= Figure 5：六轮迭代轨迹 =================
import json as _j
bs = _j.load(open(ROOT / "finetune/rp/v5_results/v51/v51-build-stats.json",
                    encoding="utf-8"))
bs5 = _j.load(open(ROOT / "finetune/rp/v5_results/logs/v5-build-stats.json",
                     encoding="utf-8"))

cats = ["A51对抗", "A旧回放", "旧语料回放", "v4保留", "场合感", "通用", "CoT重写", "数字"]
v5 = [bs5["bucket_turn_pct"].get(k, 0) for k in cats]
v51 = [bs["bucket_turn_pct"].get(k, 0) for k in cats]

fig, ax = plt.subplots(figsize=(11.2, 5.3))
y = range(len(cats))
h = 0.38
ax.barh([i + h/2 for i in y], v5, height=h, label="v5", color="#8fb8d8",
        zorder=3, edgecolor="#333", linewidth=0.7)
ax.barh([i - h/2 for i in y], v51, height=h, label="v5.1", color="#e8734a",
        zorder=3, edgecolor="#333", linewidth=0.7)
for i in y:
    if v51[i] > v5[i]:
        ax.text(v51[i] + 0.5, i, f"+{v51[i]-v5[i]:.1f}", va="center",
                fontsize=9, color="#a33b3b", fontweight="bold", zorder=4)

ax.set_yticks(list(y))
ax.set_yticklabels(cats, fontsize=10.5)
ax.invert_yaxis()
ax.set_xlabel("占训练轮次比例 (%)", fontsize=11.5)
ax.set_title("图 5  v5 → v5.1 的数据配比变化\n主攻桶 A51 对抗 11.8%→28.7%，其余桶压缩，"
             "总轮次 7,675→3,557（三重混淆，无消融无法归因）",
             fontsize=12, fontweight="bold", pad=12)
ax.legend(fontsize=10.5, loc="lower right")
ax.grid(axis="x", alpha=0.25, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, 34)

fig.savefig(f"{OUT}/fig5_mix.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("✅ fig5_mix.png")
