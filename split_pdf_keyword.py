"""
文件说明：
- 项目中用于单个 PDF 按关键词执行 OCR 切分的入口脚本。

主要职责：
- 解析命令行参数。
- 确保使用项目内 .conda 解释器运行。
- 调用单文件切分工作流完成处理。

运行方式：
- 分类：独立运行
- 直接运行命令：python split_pdf_keyword.py --input-file <pdf路径>
- 直接运行用途：处理单个 PDF，执行 OCR 识别并按关键词切分。
- 被谁调用：通常不作为其他脚本的依赖入口
- 作为依赖用途：无，公共能力已下沉到 core/、services/、workflows/。

输入：
- 配置输入：config.yaml、common.env、命令行参数 --config / --env / --input-file / --output-path
- 数据输入：单个 PDF 文件
- 前置条件：需安装 OCR 和 PDF 相关依赖；输入 PDF 路径存在

输出：
- 结果输出：切分后的 PDF 文件
- 日志输出：./log/split_pdf_keyword.log
- 副作用：默认会清空 output_path 后再生成新的切分结果

核心入口：
- 主入口函数：main()
- 关键函数：parse_args()

依赖关系：
- 依赖的本项目模块：core.config、core.logging_utils、core.runtime、workflows.split_workflow
- 依赖的第三方库：无

使用提醒：
- 这是面向“手动处理单个 PDF”场景的主入口。
- 如需复用处理逻辑，请直接使用 workflows.split_workflow，而不是导入本入口脚本。
"""

import argparse

from core.config import load_runtime_config
from core.logging_utils import setup_logger
from core.runtime import ensure_project_python
from workflows.split_workflow import process_pdf_with_config


def parse_args():
    parser = argparse.ArgumentParser(description="按关键词切分单个 PDF。")
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
        "--input-file",
        default=None,
        help="待处理 PDF 路径，优先级高于 config.yaml/common.env",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="输出目录，优先级高于 config.yaml/common.env",
    )
    return parser.parse_args()


def main():
    ensure_project_python()

    logger = setup_logger()
    args = parse_args()

    try:
        config = load_runtime_config(
            config_path=args.config,
            env_path=args.env,
            input_file=args.input_file,
            output_path=args.output_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return

    process_pdf_with_config(config, logger=logger, clear_output=True)


if __name__ == "__main__":
    main()
