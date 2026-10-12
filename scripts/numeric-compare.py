#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数字能力对照测试（bf16 / Q4_K_M / Q5_K_M 同口径）

★ 为什么重写（原 v51-q4-numeric-full.py 的缺陷）：
  旧脚本硬编码 `model: "local"` + 输出里硬写 `"model": "Q4_K_M"`。
  端口上跑的是哪个 GGUF 它不管，标签却永远写 Q4 —— 于是
  「Q5 数字 40%」这条记录实际是同一次 Q4 重跑被贴了 Q5 的标签。
  本脚本改为**从 /v1/models 反查真实模型名**并据此命名产物，
  标签与实际服务对象绑定，杜绝再犯（铁律 19 的延伸：别信打印的标签）。

★ 同时记录每次采样：temperature>0 时同模型重跑也会有波动，
  所以单次结果只能做参考，不能直接当档位间结论。
"""
import json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
API = os.environ.get("SWQ_API", "http://127.0.0.1:8083")
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]   # 仓库根（本脚本位于 <root>/finetune/rp/）
CARD = json.load(open(ROOT / "finetune/rp/suwanqing_card.json",
                      encoding="utf-8"))["system_prompt"]
EVAL_NUM = str(ROOT / "finetune/rp/v5_data/eval_numeric.jsonl")
OUTDIR = str(ROOT / "finetune/rp/v5_results/v51")

# 铁律 25：知识/推理题必须给足 token，否则推理吃光、正文 0 字、测出假低分
MAX_TOKENS = int(os.environ.get("NUM_MAXTOK", "1200"))
CONCURRENCY = int(os.environ.get("NUM_CONC", "6"))
TEMPERATURE = float(os.environ.get("NUM_TEMP", "0.3"))
TAG = os.environ.get("NUM_TAG", "")          # 人工标注用；留空则自动命名
TIMEOUT = int(os.environ.get("NUM_TIMEOUT", "420"))


def served_model():
    """端口上真实在服务的模型 —— 标签的唯一可信来源。"""
    with urllib.request.urlopen(f"{API}/v1/models", timeout=15) as r:
        return json.loads(r.read())["data"][0]["id"]


def rows():
    rs = [json.loads(l) for l in open(EVAL_NUM, encoding="utf-8") if l.strip()]
    return [r for r in rs if r["type"] != "control"], [r for r in rs if r["type"] == "control"]


def ask(model, r, max_tokens=None):
    body = {"model": model,
            "messages": [{"role": "system", "content": CARD},
                         {"role": "user", "content": r["user"]}],
            "max_tokens": max_tokens or MAX_TOKENS, "temperature": TEMPERATURE}
    req = urllib.request.Request(f"{API}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    rr = json.loads(urllib.request.urlopen(req, timeout=TIMEOUT).read())
    ch = rr["choices"][0]
    c = ch["message"].get("content") or ""
    body_txt = c.split("</think>")[-1].strip() if "</think>" in c else c.strip()
    return body_txt, ch.get("finish_reason"), rr.get("usage", {})


def extract(t):
    tm = re.findall(r"(\d{1,2})\s*[点:：]\s*(\d{1,2})?", t)
    if tm:
        return f"{int(tm[0][0])}点{tm[0][1] or '0'}"
    n = re.findall(r"\d+", t)
    return n[0] if n else None


def judge(exp, got):
    if got is None:
        return False
    e, g = re.findall(r"\d+", exp), re.findall(r"\d+", got)
    if not e:
        return True
    if not g:
        return False
    return e[0] in g or g[0] in e


def main():
    model = served_model()
    auto_tag = re.sub(r"[^A-Za-z0-9]+", "-", os.path.basename(model).replace(".gguf", ""))
    tag = TAG or auto_tag
    out = os.path.join(OUTDIR, f"numeric-{tag}.json")

    num_rows, ctrl_rows = rows()
    print(f"实测服务模型 : {model}")
    print(f"评测集       : {len(num_rows)} 数字题 + {len(ctrl_rows)} 对照题")
    print(f"参数         : max_tokens={MAX_TOKENS} temp={TEMPERATURE} 并发={CONCURRENCY}")
    print(f"产物         : {out}\n", flush=True)

    stats = {"stop": 0, "length": 0, "empty": 0}

    def run_one(r):
        t0 = time.time()
        try:
            b, fr, usage = ask(model, r)
        except Exception as e:                      # 单题失败不该毁掉整轮
            return judge(r["answer"], None), {"q": r["user"][:44], "exp": r["answer"],
                                              "body": f"<请求失败 {e}>", "why": "error"}
        if fr == "length":
            stats["length"] += 1
        elif fr == "stop":
            stats["stop"] += 1
        if not b:
            stats["empty"] += 1
        good = judge(r["answer"], extract(b))
        return good, (None if good else {"q": r["user"][:44], "exp": r["answer"],
                                         "body": b[:110], "finish": fr,
                                         "tokens": usage.get("completion_tokens")})

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        results = list(ex.map(run_one, num_rows))
    elapsed = time.time() - t0

    ok = sum(g for g, _ in results)
    fails = [f for _, f in results if f]

    print(f"数字正确 {ok}/{len(num_rows)} = {ok/len(num_rows)*100:.1f}%   ({elapsed:.0f}s)")
    print(f"finish_reason 分布 : stop={stats['stop']} length={stats['length']}  空正文={stats['empty']}")
    if stats["length"]:
        print("⚠️ 有题被 token 上限截断（finish_reason=length），这些多半是假错，铁律 25")

    print("\n错题样例：")
    for f in fails[:8]:
        print(f"  ✗ {f['q']}")
        print(f"      期望[{f['exp']}] 答[{f.get('body','')[:80]}]")

    res = {"model_served": model, "tag": tag, "num_ok": ok, "num_tot": len(num_rows),
           "rate": round(ok / len(num_rows) * 100, 1),
           "finish_reason": stats, "params": {"max_tokens": MAX_TOKENS,
                                             "temperature": TEMPERATURE,
                                             "concurrency": CONCURRENCY},
           "elapsed_s": round(elapsed, 1),
           "bf16_reference": {"rate": 47.9, "source": "实例 bf16 矩阵 140题 67/140"},
           "fails": fails[:20]}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n明细 → {out}")
    print("=" * 62)
    d = res["rate"] - 47.9
    print(f"bf16 47.9% → {tag} {res['rate']}%  ({d:+.1f}pp)")


if __name__ == "__main__":
    main()