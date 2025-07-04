#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_by_obfuscation.py - 基于官方 PAN 脚本，分开计算 simple/medium/hard 三种 obfuscation 类型的评测指标

用法：
    python evaluate_by_obfuscation.py \
        --truth_dir <truth_xml_dir> \
        --pred_dir  <pred_xml_dir>

示例：
    python evaluate_by_obfuscation.py \
        --truth_dir ./02_validation/02_validation_truth \
        --pred_dir  ./fast_chunked_output
"""
from __future__ import division

import os
import argparse
import math
import xml.etree.ElementTree as ET
from collections import namedtuple, defaultdict
from numpy.ma import zeros, sum as npsum

# 定义 Annotation 结构与官方保持一致
TREF, TOFF, TLEN, SREF, SOFF, SLEN, EXT = range(7)
Annotation = namedtuple('Annotation', [ 'this_reference',
                                        'this_offset',
                                        'this_length',
                                        'source_reference',
                                        'source_offset',
                                        'source_length',
                                        'is_external' ])

def load_truth_by_obf(truth_dir):
    """
    读取真值 XML，返回三种 obfuscation 类型下的 Annotation 列表：
      truth_map = {
          'simple': [Annotation, ...],
          'medium': [Annotation, ...],
          'hard':   [Annotation, ...]
      }
    其中对每条 truth 只记录 this_reference, this_offset, this_length；source_* 均置空/0，is_external=False。
    """
    truth_map = {'simple': [], 'medium': [], 'hard': []}
    for fn in os.listdir(truth_dir):
        if not fn.endswith('.xml'):
            continue
        path = os.path.join(truth_dir, fn)
        tree = ET.parse(path)
        root = tree.getroot()
        # 将文件名（去掉 .xml）作为 this_reference
        tref = os.path.splitext(fn)[0]
        for feat in root.findall('.//feature'):
            # 只关心 name="plagiarism" 的 tag
            if feat.get('name') != 'plagiarism':
                continue
            obf = feat.get('obfuscation')
            if obf not in truth_map:
                continue
            start  = int(feat.get('this_offset'))
            length = int(feat.get('this_length'))
            # 将 source 部分设为空，is_external=False
            ann = Annotation(tref, start, length, '', 0, 0, False)
            truth_map[obf].append(ann)
    return truth_map

def load_detections(pred_dir):
    """
    读取预测 XML，返回所有 detections 的 Annotation 列表。
    同样，只记录 this_reference, this_offset, this_length；source_* 均置空/0，is_external=False。
    """
    detections = []
    for fn in os.listdir(pred_dir):
        if not fn.endswith('.xml'):
            continue
        path = os.path.join(pred_dir, fn)
        tree = ET.parse(path)
        root = tree.getroot()
        tref = os.path.splitext(fn)[0]
        for feat in root.findall('.//feature'):
            # 只关心 name="detected-plagiarism" 的 tag
            if feat.get('name') != 'detected-plagiarism':
                continue
            start  = int(feat.get('this_offset'))
            length = int(feat.get('this_length'))
            ann = Annotation(tref, start, length, '', 0, 0, False)
            detections.append(ann)
    return detections

def index_annotations(annotations, xref=TREF):
    """
    构建一个倒排索引：map[str_reference] -> List[Annotation]
    """
    index = {}
    for ann in annotations:
        index.setdefault(ann[xref], []).append(ann)
    return index

def is_overlapping(ann1, ann2):
    """
    判断两个 Annotation 在 suspicious 文本（this_reference）上是否有重叠。
    由于 is_external 均为 False， source 部分不参与判断。
    """
    if ann1[TREF] != ann2[TREF]:
        return False
    # ann1 区间：[ann1[TOFF], ann1[TOFF]+ann1[TLEN])
    # ann2 区间：[ann2[TOFF], ann2[TOFF]+ann2[TLEN])
    if ann2[TOFF] + ann2[TLEN] <= ann1[TOFF]:
        return False
    if ann2[TOFF] >= ann1[TOFF] + ann1[TLEN]:
        return False
    return True

def overlap_chars(ann1, ann2):
    """
    计算两个 Annotation 在 suspicious 文本上的重叠字符数。
    """
    # 先判断有没有重叠
    if not is_overlapping(ann1, ann2):
        return 0
    start = max(ann1[TOFF], ann2[TOFF])
    end   = min(ann1[TOFF] + ann1[TLEN], ann2[TOFF] + ann2[TLEN])
    return end - start

def overlapping_chars(ann1, annotations):
    """
    计算 ann1 与 annotations 列表中所有 Annotation 在 suspicious 文本上的
    重叠字符总数（同一字符只算一次）。
    """
    # 获取所有与 ann1 在 suspicious 上有重叠的 ann2
    relevant = [ann2 for ann2 in annotations if is_overlapping(ann1, ann2)]
    if not relevant:
        return 0

    # 建立一个 boolean 数组，长度为 ann1[TLEN]，表示 ann1 中每个字符是否被覆盖
    mask = zeros(ann1[TLEN], dtype=bool)
    for ann2 in relevant:
        # 计算 ann2 与 ann1 在 suspicious 上的相对起点和长度
        offset_diff = ann2[TOFF] - ann1[TOFF]
        # 在 ann1 的坐标系下，被覆盖的区间：
        ov_start = max(0, offset_diff)
        ov_end = min(ann1[TLEN], offset_diff + ann2[TLEN])
        if ov_end > ov_start:
            mask[ov_start:ov_end] = True
    return npsum(mask)

def count_chars(annotations):
    """
    计算 annotations 列表中的 Annotation 在 suspicious 文本上覆盖的字符总数，
    同一 reference 内部的重叠字符只算一次。
    """
    # 先将同一 this_reference 下的 Annotation 分组，再对每组做并集计数
    index = index_annotations(annotations, xref=TREF)
    total = 0
    for tref in index:
        anns = index[tref]
        # 如果只有一个 Annotation，直接加上它的 this_length
        if len(anns) == 1:
            total += anns[0][TLEN]
            continue
        # 找到该 reference 上最大的 end
        max_end = max(ann[TOFF] + ann[TLEN] for ann in anns)
        mask = zeros(max_end, dtype=bool)
        for ann in anns:
            start = ann[TOFF]
            end   = ann[TOFF] + ann[TLEN]
            mask[start:end] = True
        total += npsum(mask)
    return total

def case_recall(case, detections):
    """
    计算一个真值 case 的 Recall：overlap_chars(case, detections) / case_length
    """
    num_detected = overlapping_chars(case, detections)
    denom = case[TLEN]
    if denom == 0:
        return 0.0
    return num_detected / denom

def macro_avg_recall(cases, detections):
    """
    宏平均 Recall：对每个 case 计算 Recall 后取平均，分母为 case 数量
    """
    if not cases and not detections:
        return 1.0
    if not cases or not detections:
        return 0.0
    recalls = []
    # 将 detections 按 reference 分组，便于快速查找
    det_index = index_annotations(detections, xref=TREF)
    for case in cases:
        tref = case[TREF]
        if tref not in det_index:
            recalls.append(0.0)
        else:
            recalls.append(case_recall(case, det_index[tref]))
    return sum(recalls) / len(cases)

def macro_avg_precision(cases, detections):
    """
    宏平均 Precision：对每个 detection 视为“待评价的 case”，调用 macro_avg_recall(detections, cases)
    """
    # 在官方脚本中，macro_avg_precision(cases, detections) = macro_avg_recall(detections, cases)
    return macro_avg_recall(detections, cases)

def true_detections(cases, detections):
    """
    按官方逻辑，得到仅与 cases 重叠的 detection 部分，返回一个 Annotation 列表，这里为了简化，
    直接返回所有能够与至少一个 case 重叠的原始 detection（不做截断）。
    """
    true_dets = []
    # 建立 case 按 reference 的索引
    case_index = index_annotations(cases, xref=TREF)
    det_index = index_annotations(detections, xref=TREF)
    for tref in case_index:
        if tref not in det_index:
            continue
        for det in det_index[tref]:
            # 只要 det 与任一 case 重叠，就视为“真实检测”
            for case in case_index[tref]:
                if is_overlapping(case, det):
                    true_dets.append(det)
                    break
    return true_dets

def micro_avg_recall_and_precision(cases, detections):
    """
    微平均 Recall 和 Precision：
      - num_plagiarized = 真值总字符数
      - num_detected   = 检测（预测）总字符数
      - num_plag_detected = 真值与预测的重叠字符数
    recall = num_plag_detected / num_plagiarized
    precision = num_plag_detected / num_detected
    """
    if not cases and not detections:
        return 1.0, 1.0
    if not cases or not detections:
        return 0.0, 0.0

    # 真值字符总数
    num_plag = count_chars(cases)
    # 检测字符总数
    num_det  = count_chars(detections)
    # 先筛选出“真实检测”的 detections
    true_dets = true_detections(cases, detections)
    # 重叠字符数 = count_chars(true_dets 与 cases 之间的交集)
    # 由于 true_dets 仅包含与 cases 重叠的那些 detection，直接用 count_chars(true_dets) 即为重叠字符数
    num_plag_det = count_chars(true_dets)

    rec = num_plag_det / num_plag if num_plag > 0 else 0.0
    prec = num_plag_det / num_det if num_det > 0 else 0.0
    return rec, prec

def granularity(cases, detections):
    """
    granularity = Σ_{每个被检测到的 case}(该 case 被多少个 detection 覆盖)  / 被检测到的 case 数
    若没有 detections，则返回 1
    """
    if not detections:
        return 1.0
    det_index = index_annotations(detections, xref=TREF)
    detections_per_case = []
    for case in cases:
        tref = case[TREF]
        if tref not in det_index:
            # 这个 case 没有任何 detection 覆盖 → 视为 0 次
            detections_per_case.append(0)
            continue
        # 统计有多少个 detection 与该 case 重叠
        count = 0
        for det in det_index[tref]:
            if is_overlapping(case, det):
                count += 1
        if count > 0:
            detections_per_case.append(count)
    if not detections_per_case:
        # 没有 case 被检测到
        return 1.0
    detected_cases = sum(1 for c in detections_per_case if c > 0)
    if detected_cases == 0:
        return 1.0
    return sum(detections_per_case) / detected_cases

def plagdet_score(rec, prec, gran):
    """
    按照官方定义合并得分：
      如果 rec=0 and prec=0 或者 gran < 1 → 返回 0
      否则 → ((2·rec·prec)/(rec+prec)) / log2(1 + gran)
    """
    if (rec == 0 and prec == 0) or prec < 0 or rec < 0 or gran < 1:
        return 0.0
    return ((2 * rec * prec) / (rec + prec)) / math.log(1 + gran, 2)

def calculate_f1(rec, prec):
    """计算F1分数"""
    if rec + prec > 0:
        return 2 * (rec * prec) / (rec + prec)
    return 0.0

def main():
    parser = argparse.ArgumentParser(
        description="基于官方 PAN 脚本，分开计算 simple/medium/hard 三种 obfuscation 类型的评测指标"
    )
    parser.add_argument('--truth_dir', required=True, help="真值 XML 目录")
    parser.add_argument('--pred_dir',  required=True, help="预测 XML 目录")
    args = parser.parse_args()

    # 1) 读取真值 & 按 obfuscation 类型分类
    truth_map = load_truth_by_obf(args.truth_dir)
    # 2) 读取所有检测结果
    all_detections = load_detections(args.pred_dir)

    # 逐个 obf 类型计算并输出
    print("Obfuscation  Micro_Recall  Micro_Precision  Micro_F1  Macro_Recall  Macro_Precision  Macro_F1  Granularity")

    for obf in ('simple', 'medium', 'hard'):
        cases = truth_map[obf]
        if not cases:
            print(f"{obf:10s}    {'0.000':>9s}    {'0.000':>14s}    {'0.000':>9s}    "
                  f"{'0.000':>9s}    {'0.000':>15s}    {'0.000':>9s}    "
                  f"{'1.000':>11s}    {'0.000':>7s}")
            continue

        # 计算基础指标
        micro_rec, micro_prec = micro_avg_recall_and_precision(cases, all_detections)
        macro_rec = macro_avg_recall(cases, all_detections)
        macro_prec = macro_avg_precision(cases, all_detections)
        gran = granularity(cases, all_detections)
        pd_score = plagdet_score(micro_rec, micro_prec, gran)

        # 新增F1计算
        micro_f1 = calculate_f1(micro_rec, micro_prec)
        macro_f1 = calculate_f1(macro_rec, macro_prec)

        print(f"{obf:10s}    "
              f"{micro_rec:9.4f}    {micro_prec:14.4f}    {micro_f1:9.4f}    "
              f"{macro_rec:9.4f}    {macro_prec:15.4f}    {macro_f1:9.4f}    "
              f"{gran:11.4f}    ")

if __name__ == '__main__':
    main()
