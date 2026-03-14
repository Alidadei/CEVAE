#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CEVAE 实验结果自动记录脚本

用法:
    python run_with_log.py -dataset ihdp
    python run_with_log.py -dataset ihdp1000 -separate_reps -n_reps 10
"""

from __future__ import print_function
import sys
import os
from datetime import datetime

def main():
    # 获取时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 解析命令行参数
    args = sys.argv[1:]

    if not args or "-dataset" not in " ".join(args):
        print("用法: python run_with_log.py [cevae_ihdp.py 的参数]")
        print("示例:")
        print("  python run_with_log.py -dataset ihdp")
        print("  python run_with_log.py -dataset ihdp1000 -separate_reps -n_reps 10")
        sys.exit(1)

    # 确定数据集和模式
    dataset = "unknown"
    mode = "standard"

    try:
        dataset_idx = args.index("-dataset")
        if dataset_idx + 1 < len(args):
            dataset = args[dataset_idx + 1]
    except ValueError:
        pass

    # 确定模式
    if dataset == "ihdp1000":
        if "-separate_reps" in args:
            mode = "separate"
        else:
            mode = "combined"
    elif dataset == "twins":
        mode = "standard"
    elif dataset == "ihdp":
        mode = "standard"

    # 生成文件名
    filename = "{}_{}_{}.txt".format(dataset, mode, timestamp)

    # 创建 record 目录
    record_dir = "record"
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    filepath = os.path.join(record_dir, filename)

    # 显示实验信息
    print("=" * 50)
    print("CEVAE 实验结果自动记录")
    print("=" * 50)
    print("数据集: {}".format(dataset))
    print("模式: {}".format(mode))
    print("时间戳: {}".format(timestamp))
    print("输出文件: {}".format(filepath))
    print("命令: python cevae_ihdp.py " + " ".join(args))
    print("=" * 50)
    print()
    print("开始训练...")

    # 运行训练并记录输出
    import subprocess

    process = subprocess.Popen(
        [sys.executable, "cevae_ihdp.py"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    # 用于收集重要输出
    important_output = []
    in_final_results = False
    final_results = []

    # 实时处理输出
    for line in process.stdout:
        line_lean = line.rstrip('\n')

        # 直接输出到终端（不过滤进度条，让它正常显示）
        if line_lean:
            print(line_lean, flush=True)

        # 只保存重要信息到文件（过滤掉进度条）
        # 进度条特征：包含回车符、进度条符号、百分比和ETA
        is_progress_bar = (
            '\r' in line or
            ('epoch #' in line and '|' in line and '%' in line and 'ETA:' in line)
        )

        if not is_progress_bar and line_lean:
            # 只保存非进度条的内容
            if line_lean.startswith("Replication"):
                important_output.append(line_lean)
            elif "Improved validation bound" in line_lean:
                important_output.append(line_lean)
            elif line_lean.startswith("Epoch:"):
                important_output.append(line_lean)
            elif "CEVAE model total scores" in line_lean:
                in_final_results = True
                important_output.append(line_lean)
            elif in_final_results:
                final_results.append(line_lean)
                important_output.append(line_lean)


    process.wait()

    # 写入文件（最终结果在前，训练过程在后）
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("CEVAE 实验记录\n")
        f.write("=" * 60 + "\n\n")

        # 实验配置
        f.write("【实验配置】\n")
        f.write("-" * 60 + "\n")
        f.write("数据集: {}\n".format(dataset))
        f.write("模式: {}\n".format(mode))
        f.write("时间: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("命令: python cevae_ihdp.py " + " ".join(args) + "\n")
        f.write("\n")

        # 最终结果（放在前面）
        if final_results:
            f.write("【最终结果】\n")
            f.write("-" * 60 + "\n")
            for result_line in final_results:
                f.write(result_line + "\n")
            f.write("\n")

        # 训练过程（放在后面）
        if important_output:
            f.write("【训练过程】\n")
            f.write("-" * 60 + "\n")
            for output_line in important_output:
                f.write(output_line + "\n")

        f.write("\n" + "=" * 60 + "\n")

    print()
    print("=" * 50)
    print("实验完成！")
    print("结果已保存到: {}".format(filepath))
    print("=" * 50)

if __name__ == "__main__":
    main()
