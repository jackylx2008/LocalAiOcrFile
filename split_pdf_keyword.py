"""
文件说明：
- 项目中用于单个 PDF 按关键词执行 OCR 切分的主入口脚本。

主要职责：
- 确保使用项目内 .conda 解释器运行。
- 加载运行时配置并在需要时清空输出目录。
- 调用 OCR 与切分模块完成单文件处理。

运行方式：
- 分类：两者都可以
- 直接运行命令：python split_pdf_keyword.py --input-file <pdf路径>
- 直接运行用途：处理单个 PDF，执行 OCR 识别并按关键词切分。
- 被谁调用：process_usb_pdfs.py、rename_pdfs_by_regex.py
- 作为依赖用途：向其他脚本提供解释器切换、运行时配置加载和单文件处理能力。

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
- 关键函数：ensure_project_python()、load_runtime_config()、process_pdf_with_config()

依赖关系：
- 依赖的本项目模块：config_loader.py、logging_config.py、ocr_engine.py、splitter.py
- 依赖的第三方库：无

使用提醒：
- 这是面向“手动处理单个 PDF”场景的主入口。
- process_pdf_with_config() 被复用时可通过 clear_output=False 避免清空已有输出。
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from config_loader import load_config
from logging_config import setup_logger


def ensure_project_python():
    project_python = Path(__file__).resolve().parent / ".conda" / "python.exe"
    if not project_python.exists():
        return

    current_python = Path(sys.executable).resolve()
    target_python = project_python.resolve()
    if os.path.normcase(str(current_python)) == os.path.normcase(str(target_python)):
        return

    print(f"检测到当前解释器为 {current_python}，切换到项目解释器 {target_python}")
    os.execv(str(target_python), [str(target_python), *sys.argv])


def clear_output_directory(output_path, logger):
    output_dir = Path(output_path)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录不存在，已创建: {output_dir}")
        return

    removed_count = 0
    for item in output_dir.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            else:
                shutil.rmtree(item)
            removed_count += 1
        except Exception as exc:
            logger.warning(f"清理输出目录项失败: {item}, error={exc}")

    logger.info(f"启动前已清空输出目录: {output_dir}，删除 {removed_count} 项")


def load_runtime_config(
    config_path="config.yaml", env_path=None, input_file=None, output_path=None
):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在。")

    config = load_config(config_path, env_path=env_path)

    if input_file is not None:
        config["input_file"] = str(Path(input_file))

    if output_path is not None:
        config["output_path"] = str(Path(output_path))

    return config


def process_pdf_with_config(config, logger=None, clear_output=True):
    app_logger = logger or setup_logger()
    input_file = config.get("input_file")
    output_path = config.get("output_path", "./output/")

    if not input_file:
        app_logger.error("配置中未指定 input_file。")
        return False

    if not os.path.exists(input_file):
        app_logger.error(f"输入文件 {input_file} 不存在。")
        return False

    from ocr_engine import run_startup_self_check
    from splitter import PDFSplitter

    if clear_output:
        clear_output_directory(output_path, app_logger)

    app_logger.info("执行启动前自检...")
    ocr_processor = run_startup_self_check(config, app_logger)

    app_logger.info(f"正在对文件进行 OCR 识别: {input_file}")
    ocr_results = ocr_processor.process_pdf(input_file)

    app_logger.info("初始化切分器...")
    splitter = PDFSplitter(config)

    app_logger.info("正在执行 PDF 切分处理...")
    splitter.split_by_ocr_results(input_file, ocr_results)

    app_logger.info("切分处理已完成。")
    return True


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
