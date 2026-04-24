"""
文件说明：
- 根据 PDF 首页的 OCR 识别结果与 regex_pattern 规则，对输入目录中的 PDF 执行重命名处理的入口脚本。

主要职责：
- 解析配置和输入目录参数。
- 调用重命名工作流处理输入目录中的 PDF。
- 为“只重命名、不切分”的日常操作提供直接入口。

运行方式：
- 分类：独立运行
- 直接运行命令：python rename_pdfs_by_regex.py [--input-path ...]
- 直接运行用途：默认扫描 rename_input_path 下的 PDF，并直接按首页 OCR 结果原地重命名。
- 被谁调用：通常不作为其他脚本的依赖入口
- 作为依赖用途：无，公共能力已下沉到 workflows.rename_workflow 和 services.pdf_rename_service。

输入：
- 配置输入：config.yaml 中的 rename_input_path、regex_pattern
- 数据输入：输入目录下的 PDF、PDF 首页 OCR 文本
- 前置条件：输入目录中已有待重命名的 PDF；OCR 依赖环境可用

输出：
- 结果输出：原地重命名后的 PDF
- 日志输出：./log/rename_pdfs_by_regex.log
- 副作用：直接修改原文件名

核心入口：
- 主入口函数：main()
- 关键函数：parse_args()

依赖关系：
- 依赖的本项目模块：core.config、core.logging_utils、core.runtime、workflows.rename_workflow
- 依赖的第三方库：无

使用提醒：
- 默认从 rename_input_path 读取 PDF，并直接在原文件上重命名。
"""

import argparse
import os

from core.config import load_config
from core.logging_utils import setup_logger
from core.runtime import ensure_project_python
from workflows.rename_workflow import rename_pdfs


def parse_args():
    parser = argparse.ArgumentParser(
        description="按首页 OCR 结果重命名 PDF，默认直接处理 rename_input_path 中的 PDF。"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="规则配置文件路径，默认 config.yaml",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="输入目录，优先级高于 config.yaml/common.env 中的 rename_input_path",
    )
    parser.add_argument(
        "--filename-prefix",
        default="",
        help="重命名结果文件名前缀，例如传入 BG_ 后生成 BG_XXX.pdf",
    )
    return parser.parse_args()


def main():
    ensure_project_python()

    logger = setup_logger()
    args = parse_args()
    config_path = args.config

    if not os.path.exists(config_path):
        logger.error(f"配置文件 {config_path} 不存在。")
        return

    config = load_config(config_path)
    if args.input_path is None:
        config["rename_input_path"] = config.get(
            "rename_input_path", config.get("output_path", "./output/")
        )
    rename_pdfs(
        config,
        logger,
        input_path=args.input_path,
        filename_prefix=args.filename_prefix,
        in_place=True,
    )


if __name__ == "__main__":
    main()
