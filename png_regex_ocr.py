"""
文件说明：
- 对目标目录中的 PNG 图片执行 OCR 识别，并将 regex_pattern 的匹配统计写入 TXT 的入口脚本。

主要职责：
- 解析命令行参数。
- 确保使用项目内 .conda 解释器运行。
- 加载 PNG OCR 工作流所需配置。
- 调用 PNG OCR 正则统计工作流。

运行方式：
- 分类：独立运行
- 直接运行命令：python png_regex_ocr.py [--input-path ...] [--output-txt ...]
- 直接运行用途：批量识别指定目录中的 PNG，并导出 regex_pattern 统计结果。
- 被谁调用：通常不作为其他脚本依赖入口
- 作为依赖用途：无，公共能力已下沉到 workflows/ 和 services/。

输入：
- 配置输入：config.yaml、common.env、命令行参数 --config / --env / --input-path / --output-txt
- 数据输入：目标目录中的 PNG 图片
- 前置条件：目标目录存在；OCR 环境可用

输出：
- 结果输出：统计 TXT 文件
- 日志输出：./log/png_regex_ocr.log
- 副作用：写出统计 TXT 文件

核心入口：
- 主入口函数：main()
- 关键函数：parse_args()

依赖关系：
- 依赖的本项目模块：core.config、core.logging_utils、core.runtime、workflows.png_regex_workflow
- 依赖的第三方库：无
"""

import argparse

from core.config import load_runtime_config
from core.logging_utils import setup_logger
from core.runtime import ensure_project_python
from workflows.png_regex_workflow import run_png_regex_workflow


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量处理目标目录中的 PNG 图片，执行 OCR 并导出 regex_pattern 统计 TXT。"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="规则配置文件路径，默认 config.yaml",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="环境变量文件路径，默认自动读取 common.env",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="PNG 输入目录，优先级高于 config.yaml/common.env 中的 png_ocr_input_path",
    )
    parser.add_argument(
        "--output-txt",
        default=None,
        help="统计 TXT 输出路径，优先级高于 config.yaml/common.env 中的 png_ocr_output_txt_path",
    )
    return parser.parse_args()


def main():
    ensure_project_python()

    logger = setup_logger(log_file="./log/png_regex_ocr.log", filemode="w")
    args = parse_args()

    try:
        config = load_runtime_config(config_path=args.config, env_path=args.env)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return

    try:
        run_png_regex_workflow(
            config,
            logger,
            input_path=args.input_path,
            output_txt_path=args.output_txt,
        )
    except KeyboardInterrupt:
        logger.warning("收到 Ctrl+C，PNG OCR 工作流已被用户中断。")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
