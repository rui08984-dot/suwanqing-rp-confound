#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数字能力 · 干净子集复测（剔除训练集污染题）

★ 为什么必须重测：
  `eval_numeric.jsonl` 的 140 道数字题里有 68 道（49%）在 `d_numeric.jsonl`
  训练数据里逐字出现过（其中 22 条连标准答案都相同）。
  那些题测的是「背题」，会把数字能力分数抬高。
  此脚本只跑**从未进训练**的 72 题，给出未被污染的能力下界。

★ 结论怎么用：
  干净 72 题的分数 = 真实泛化能力。
  与污染题的差值 = 背题贡献量。
  若两者差距不大，说明 v5.1 的数字提升不是背来的。
"""
import json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
API = os.environ.get("SWQ_API", "http://127.0.0.1:8083")
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]   # 仓库根（本脚本位于 <root>/finetune/rp/）
CARD = json.load(open(ROOT / "finetune/rp/suwanqing_card.json",
                      encoding="utf-8"))["system_prompt"]
BASE = str(ROOT / "finetune/rp/v5_data")
OUTDIR = str(ROOT / "finetune/rp/v5_results/v51")

MAX_TOKENS = int(os.environ.get("NUM_MAXTOK", "1200"))
CONCURRENCY = int(os.environ.get("NUM_CONC", "6"))
TEMPERATURE = float(os.environ.get("NUM_TEMP", "0.3"))
TAG = os.environ.get("NUM_TAG", "clean")


def served_model():
    with urllib.request.urlopen(f"{API}/v1/models", timeout=15) as r:
        return json.loads(r.read())["data"][0]["id"]


def split_rows():
    ev = [json.loads(l) for l in open(f"{BASE}/eval_numeric.jsonl", encoding="utf-8") if l.strip()]
    train_users = set()
    for l in open(f"{BASE}/d_numeric.jsonl", encoding="utf-8"):
        if not l.strip():
            continue
        for m in json.loads(l).get("messages", []):
            if m.get("role") == "user":
                train_users.add(m["content"].strip())
    num = [r for r in ev if r["type"] != "control"]
    clean = [r for r in num if r["user"].strip() not in train_users]
    dirty = [r for r in num if r["user"].strip() in train_users]
    return num, clean, dirty


def ask(model, r):
    body = {"model": model,
            "messages": [{"role": "system", "content": CARD},
                         {"role": "user", "content": r["user"]}],
            "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE}
    req = urllib.request.Request(f"{API}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    rr = json.loads(urllib.request.urlopen(req, timeout=420).read())
    ch = rr["choices"][0]
    c = ch["message"].get("content") or ""
    return (c.split("</think>")[-1].strip() if "</think>" in c else c.strip()), ch.get("finish_reason")


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
    auto = re.sub(r"[^A-Za-z0-9]+", "-", os.path.basename(model).replace(".gguf", ""))
    tag = f"{auto}-{TAG}"
    out = os.path.join(OUTDIR, f"numeric-{tag}.json")

    num, clean, dirty = split_rows()
    print(f"实测服务模型 : {model}")
    print(f"数字题 {len(num)} 道 = 干净 {len(clean)} + 污染 {len(dirty)}（污染题在训练集里逐字出现过）")
    print(f"产物         : {out}\n", flush=True)

    def run(rows):
        ok, fails, lens = 0, [], 0
        def one(r):
            try:
                b, fr = ask(model, r)
            except Exception as e:
                return False, {"q": r["user"][:40], "body": f"<失败 {e}>"}
            good = judge(r["answer"], extract(b))
            return good, (None if good else {"q": r["user"][:40], "exp": r["answer"],
                                             "body": b[:90], "finish": fr})
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            for good, f in ex.map(one, rows):
                ok += good
                if f:
                    fails.append(f)
                if f and f.get("finish") == "length":
                    lens += 1
        return ok, fails, lens

    t0 = time.time()
    c_ok, c_fails, c_len = run(clean)
    d_ok, d_fails, _ = run(dirty)
    el = time.time() - t0

    print(f"\n★ 干净子集（泛化能力真值） : {c_ok}/{len(clean)} = {c_ok/len(clean)*100:.1f}%"
          f"   截断 {c_len}")
    print(f"  污染子集（含训练题）      : {d_ok}/{len(dirty)} = {d_ok/len(dirty)*100:.1f}%")
    print(f"  背题增益                  : {(d_ok/len(dirty) - c_ok/len(clean))*100:+.1f}pp")
    print(f"  耗时 {el:.0f}s\n")

    print("干净子集错题样例：")
    for f in c_fails[:6]:
        print(f"  ✗ {f['q']}")
        print(f"      期望[{f.get('exp','')}] 答[{f.get('body','')[:70]}]")

    res = {"model_served": model, "tag": tag,
           "clean_ok": c_ok, "clean_tot": len(clean),
           "clean_rate": round(c_ok / len(clean) * 100, 1),
           "dirty_ok": d_ok, "dirty_tot": len(dirty),
           "dirty_rate": round(d_ok / len(dirty) * 100, 1),
           "memorization_gain_pp": round((d_ok / len(dirty) - c_ok / len(clean)) * 100, 1),
           "truncated": c_len, "params": {"max_tokens": MAX_TOKENS,
                                         "temperature": TEMPERATURE,
                                         "concurrency": CONCURRENCY},
           "elapsed_s": round(el, 1),
           "clean_fails": c_fails[:20]}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n明细 → {out}")


if __name__ == "__main__":
    main()