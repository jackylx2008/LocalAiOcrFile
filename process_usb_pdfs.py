"""
文件说明：
- 自动扫描已插入 U 盘中的 PDF，复制到本地输入目录后串联执行切分和重命名的入口脚本。

主要职责：
- 解析命令行参数。
- 初始化运行环境与配置。
- 调用 USB 批处理工作流完成一条龙处理。

运行方式：
- 分类：独立运行
- 直接运行命令：python process_usb_pdfs.py
- 直接运行用途：作为整套业务流程的批处理入口，处理 U 盘中的 PDF 文件。
- 被谁调用：通常不作为其他脚本的依赖入口
- 作为依赖用途：无，公共能力已下沉到 workflows.usb_batch_workflow 和相关 services。

输入：
- 配置输入：config.yaml、common.env、命令行参数 --config / --env
- 数据输入：已插入 U 盘中的 PDF、本地 input_dir、output_path
- 前置条件：需在 Windows 下运行；U 盘已插入；项目解释器和依赖环境可用

输出：
- 结果输出：复制后的本地输入 PDF、切分后的 PDF、重命名后的 PDF
- 日志输出：./log/process_usb_pdfs.log
- 副作用：清空本地 input_dir 和 output_path，对文件进行复制、生成和重命名

核心入口：
- 主入口函数：main()
- 关键函数：parse_args()

依赖关系：
- 依赖的本项目模块：core.config、core.logging_utils、core.runtime、workflows.usb_batch_workflow
- 依赖的第三方库：无

使用提醒：
- 这是面向日常使用的主入口之一，适合“插入 U 盘后批量处理”场景。
- 默认只处理符合当前日期筛选规则的 PDF，不会无条件扫描全部文件。
"""

import argparse

from core.config import load_runtime_config
from core.logging_utils import setup_logger
from core.runtime import ensure_project_python
from workflows.usb_batch_workflow import run_usb_batch


DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_ENV_PATH = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="自动扫描所有已插入U盘，将 PDF 复制到本地 input 目录后执行切分和重命名。"
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="规则配置文件路径，默认 config.yaml",
    )
    parser.add_argument(
        "--env",
        default=DEFAULT_ENV_PATH,
        help="环境变量文件路径，默认自动读取 common.env",
    )
    return parser.parse_args()


def main():
    ensure_project_python()

    logger = setup_logger()
    args = parse_args()

    try:
        base_config = load_runtime_config(config_path=args.config, env_path=args.env)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return

    run_usb_batch(base_config, logger)


if __name__ == "__main__":
    main()
