"""
文件说明：
- 编排单个 PDF 的 OCR 识别与关键词切分流程。

主要职责：
- 在需要时清空输出目录。
- 执行 OCR 启动自检与整份 PDF 识别。
- 调用切分服务输出子 PDF。

运行方式：
- 分类：被依赖脚本
- 直接运行命令：不建议直接运行
- 直接运行用途：无独立业务入口，主要被单文件入口和 USB 批处理流程调用。
- 被谁调用：split_pdf_keyword.py、workflows.usb_batch_workflow
- 作为依赖用途：提供单文件切分工作流。

输入：
- 配置输入：input_file、output_path、OCR 配置、切分关键词
- 数据输入：单个 PDF 文件
- 前置条件：config 中必须包含可访问的 input_file

输出：
- 结果输出：切分后的 PDF 文件
- 日志输出：调用方 logger 或默认 logger
- 副作用：可选清空 output_path

核心入口：
- 关键函数：process_pdf_with_config()

依赖关系：
- 依赖的本项目模块：core.logging_utils、services.file_ops_service、services.ocr_service、services.pdf_split_service
- 依赖的第三方库：无

使用提醒：
- clear_output=True 时会先清空输出目录，再生成新文件。
"""

import os

from core.logging_utils import setup_logger
from services.file_ops_service import clear_output_directory
from services.ocr_service import run_startup_self_check
from services.pdf_split_service import PDFSplitter


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
