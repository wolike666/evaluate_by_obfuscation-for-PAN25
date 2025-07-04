#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict-fast-2-3gram.py — 分块＋pipeline 批量化推理
对每块文件（默认100）：
  1. TF-IDF 过滤 + Jaccard 简单过滤
  2. 收集 medium/hard 句对
  3. 一次性交给 pipeline
  4. 拆分结果、写 XML
用法：
    python scripts/predict-fast-2-3gram.py \
      --model_dir ./bert-base-train10000 \
      --data_dir 00_spot_check/00_spot_check \
      --truth_dir 00_spot_check/00_spot_check_truth \
      --output_dir ./chunked-v1-spot-output2 \
      --threshold_tfidf 0.7 \
      --threshold_jaccard_simple 0.1 \
      --threshold_jaccard_other 0.3 \
      --threshold_bert 0.8 \
      --threshold_bert_simple 0.75 \
      --batch_size 64 \
      --min_len 50 \
      --merge_gap 20 \
      --chunk_size 100


python scripts/predict-fast-2-3gram.py --model_dir ./bert-base-train10000  --data_dir 00_spot_check/00_spot_check --truth_dir 00_spot_check/00_spot_check_truth --output_dir ./bert_base_train10000-spot-fast2-3gram-output --threshold_tfidf 0.7 --threshold_jaccard_simple 0.1 --threshold_jaccard_other 0.3  --threshold_bert 0.8 --threshold_bert_simple 0.75 --batch_size 64 --min_len 50 --merge_gap 20  --chunk_size 100
"""
import os, gc, argparse
import numpy as np
import torch
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, pipeline
from xml.dom.minidom import parse, Document

def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir',      required=True)
    p.add_argument('--data_dir',       required=True)
    p.add_argument('--truth_dir',      required=True)
    p.add_argument('--output_dir',     required=True)
    p.add_argument('--threshold_tfidf',  type=float, default=0.7)
    p.add_argument('--threshold_jaccard_simple', type=float, default=0.4)
    p.add_argument('--threshold_jaccard_other', type=float, default=0.7)
    p.add_argument('--threshold_bert', type=float, default=0.8,
                                          help = "medium/hard 的 BERT 阈值")
    p.add_argument('--threshold_bert_simple', type=float, default=None,
                                           help = "simple 的 BERT 阈值，若留空则使用 --threshold_bert")
    p.add_argument('--batch_size',       type=int,   default=64)
    p.add_argument('--min_len',          type=int,   default=50)
    p.add_argument('--merge_gap',        type=int,   default=20)
    p.add_argument('--chunk_size',       type=int,   default=100)
    return p.parse_args()

def jaccard(a, b, n=3):
    """计算两个字符串的字符 n-gram Jaccard 相似度（默认为 3-gram）"""
    shingles_a = {a[i:i + n] for i in range(len(a) - n + 1)}
    shingles_b = {b[i:i + n] for i in range(len(b) - n + 1)}
    inter = len(shingles_a & shingles_b)
    union = len(shingles_a | shingles_b) + 1e-9
    return inter / union

def merge_feats(feats,gap):
    """
    将重叠或相邻（间隔 <= gap）的检测片段合并
    feats: [(start, end), ...] 列表
    返回合并后的 [ [s1, e1], [s2, e2], ... ]
    """
    out=[]
    for s,e in sorted(feats):
        if not out or s>out[-1][1]+gap:
            out.append([s,e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return out

def load_obf_levels(fn):
    """
    从 truth XML 中读取每个抄袭片段的 obfuscation 级别
    返回 [(offset_start, offset_end, level), ...]
    """
    dom=parse(fn)
    lv=[]
    for f in dom.getElementsByTagName("feature"):
        if f.getAttribute("name")!="plagiarism": continue
        o=int(f.getAttribute("this_offset"))
        l=int(f.getAttribute("this_length"))
        obf=f.getAttribute("obfuscation")
        lv.append((o,o+l,obf))
    return lv

def main():
    args = parse_args()

    # 如果没有单独指定 simple 的阈值，就用通用阈值
    if args.threshold_bert_simple is None:
        args.threshold_bert_simple = args.threshold_bert

    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================
    # 1. 全库一次性 fit TF-IDF
    # =========================================
    print("Fitting TF-IDF on all sentences...")
    all_sents=[]
    susp_dir = os.path.join(args.data_dir,"susp")
    src_dir  = os.path.join(args.data_dir,"src")
    # 收集所有 susp 文本的句子
    for fn in os.listdir(susp_dir):
        all_sents += sent_tokenize(open(os.path.join(susp_dir,fn),encoding="utf-8").read())
    # 收集所有 src 文本的句子
    for fn in os.listdir(src_dir):
        all_sents += sent_tokenize(open(os.path.join(src_dir,fn),encoding="utf-8").read())
    # 只 fit 一次，后续调用 transform 生成稀疏向量
    vec = TfidfVectorizer().fit(all_sents)
    del all_sents; gc.collect()

    # =========================================
    # 2. 初始化 Hugging Face pipeline
    # =========================================
    # 加载 tokenizer 和模型，并封装为 text-classification pipeline
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    clf = pipeline(
        "text-classification",
        model=args.model_dir,
        tokenizer=tokenizer,
        device=0,               # 强制使用 GPU 0；
        batch_size=args.batch_size,
        top_k=1,                # 只返回最可能的标签
        truncation=True,        # 自动截断超长输入
        padding="max_length",   # 填充到 max_length
        max_length=256
    )

    # 读取所有 pairs 列表
    pairs = [line.strip().split() for line in open(os.path.join(args.data_dir,"pairs"),encoding="utf-8")]

    # =========================================
    # 3. 分块处理：每 chunk_size 个文件对为一批
    # =========================================
    for chunk_start in range(0, len(pairs), args.chunk_size):
        # 取出当前块的文件对列表
        chunk = pairs[chunk_start: chunk_start+args.chunk_size]

        # feats 存储每个文件对的检测结果
        feats = {i: [] for i in range(len(chunk))}
        # bert_candidates 收集 medium/hard 句对，稍后一起送 pipeline
        bert_candidates = []  # 结构：(file_idx, susp_sent, src_sent, this_offset, this_length)
        pruned_by_jaccard = 0  # 统计被 Jaccard 拦下的句对数量

        # -------- 分块内：TF-IDF + Jaccard 预筛 --------
        for i,(susp,src) in enumerate(
                tqdm(chunk, desc=f"TFIDF/Jaccard {chunk_start}")):
            # 读入两个文本
            s_txt = open(os.path.join(susp_dir,susp),encoding="utf-8").read()
            t_txt = open(os.path.join(src_dir, src),encoding="utf-8").read()
            # 分句
            s_sents = sent_tokenize(s_txt)
            t_sents = sent_tokenize(t_txt)

            # 计算 TF-IDF 稀疏矩阵，然后求余弦相似度
            m1 = vec.transform(s_sents)                # susp 的句子向量
            m2 = vec.transform(t_sents)                # src 的句子向量
            sim = (m1 @ m2.T).toarray()                 # 稀疏矩阵乘积 → 稠密矩阵
            del m1, m2                                 # 及时释放
            # 加载对应 truth 的 obfuscation 级别
            obf_lv = load_obf_levels(os.path.join(args.truth_dir,f"{susp[:-4]}-{src[:-4]}.xml"))

            # 遍历所有相似度 >= threshold_tfidf 的句对
            for a,b in zip(*np.where(sim >= args.threshold_tfidf)):
                ss, ts = s_sents[a], t_sents[b]
                so, sl = s_txt.find(ss), len(ss)
                # 过滤过短句子
                if sl < args.min_len or len(ts) < args.min_len:
                    continue
                # 判断此句对的 obfuscation 级别
                obf = next((lbl for x,y,lbl in obf_lv if x<=so<y),None)
                # 先用 Jaccard 再决定
                j_thr = (args.threshold_jaccard_simple
                         if obf=='simple' else args.threshold_jaccard_other)
                score_j = jaccard(ss, ts)
                if score_j < j_thr:
                    pruned_by_jaccard += 1
                    continue
                bert_candidates.append((i, ss, ts, so, sl, obf))
            gc.collect()

        # 打印本 chunk Jaccard 拦下数量
        print(f"[chunk {chunk_start}] pruned_by_jaccard = {pruned_by_jaccard}")
        # -------- 分块内：BERT 批量推理 --------
        if bert_candidates:
            # 构造 pipeline 输入
            texts = [(ss, ts) for (_, ss, ts, _, _, _) in bert_candidates]
            preds = clf(texts)   # 批量前向

            # 解析结果，保留置信度 >= threshold_bert 的
            for (idx, ss, ts, so, sl, obf), res in zip(bert_candidates, preds):
                score = res["score"] if isinstance(res,dict) else res[0]["score"]
                # 不同 obf 用不同阈值
                thr = args.threshold_bert_simple if obf == 'simple' else args.threshold_bert

                if score >= thr:
                    # 再次定位 source 中文本偏移
                    to, tl = open(os.path.join(src_dir, chunk[idx][1]),encoding="utf-8").read().find(ts), len(ts)
                    feats[idx].append((so,so+sl, chunk[idx][1], to, tl))

            # 彻底释放 pipeline 中的临时缓存
            del bert_candidates, preds;
            torch.cuda.empty_cache();
            gc.collect()

        # -------- 分块内：写 XML --------
        for i,(susp,src) in enumerate(chunk):
            doc = Document()
            root=doc.createElement("document")
            root.setAttribute("reference", susp)
            doc.appendChild(root)
            raw = feats[i]  # 当前文件对的所有检测片段
            # 合并相邻/重叠的片段
            for so,eo in merge_feats([(r[0],r[1]) for r in raw], args.merge_gap):
                # 找到该起点对应的 source 及偏移信息
                _,_,src_fn,to,tl = next(r for r in raw if r[0]==so)
                feat=doc.createElement("feature")
                feat.setAttribute("name","detected-plagiarism")
                feat.setAttribute("this_offset",  str(so))
                feat.setAttribute("this_length",  str(eo-so))
                feat.setAttribute("source_reference", src_fn)
                feat.setAttribute("source_offset", str(to))
                feat.setAttribute("source_length", str(tl))
                root.appendChild(feat)

            # 写到磁盘
            outp = os.path.join(args.output_dir, f"{susp[:-4]}-{src[:-4]}.xml")
            with open(outp,"w",encoding="utf-8") as f:
                f.write(doc.toprettyxml(indent="  "))

        # 分块结束后释放 feats
        del feats;
        gc.collect()

    print("Done! Outputs in", args.output_dir)

if __name__=="__main__":
    main()
