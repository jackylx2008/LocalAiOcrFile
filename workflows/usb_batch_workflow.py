"""
文件说明：
- 编排 U 盘 PDF 导入、本地切分和切分结果重命名的一条龙流程。

主要职责：
- 扫描可移动 U 盘并复制目标 PDF 到 input_dir。
- 对每个复制到本地的 PDF 执行切分。
- 对本轮新生成的切分结果执行重命名。

运行方式：
- 分类：被依赖脚本
- 直接运行命令：不建议直接运行
- 直接运行用途：无独立业务入口，主要被 process_usb_pdfs.py 调用。
- 被谁调用：process_usb_pdfs.py
- 作为依赖用途：提供完整的 USB 批处理工作流。

输入：
- 配置输入：input_dir、output_path、OCR 配置、切分关键词、重命名正则
- 数据输入：已插入 U 盘中的 PDF
- 前置条件：需在 Windows 环境运行并检测到可移动磁盘

输出：
- 结果输出：本地输入 PDF、切分后的 PDF、重命名后的 PDF
- 日志输出：调用方 logger
- 副作用：清空 input_dir 和 output_path，并生成/重命名文件

核心入口：
- 关键函数：run_usb_batch()、process_single_pdf()

依赖关系：
- 依赖的本项目模块：services.file_ops_service、services.pdf_rename_service、services.usb_scan_service、workflows.split_workflow
- 依赖的第三方库：无
"""

import copy
from pathlib import Path

from services.file_ops_service import clear_output_directory
from services.pdf_rename_service import rename_pdf_files
from services.usb_scan_service import (
    copy_pdfs_from_usb_drives,
    list_removable_drive_roots,
)
from workflows.split_workflow import process_pdf_with_config


def build_output_path(output_root, input_dir, pdf_path):
    return Path(output_root)


def process_single_pdf(pdf_path, base_config, input_dir, output_root, logger):
    output_path = build_output_path(output_root, input_dir, pdf_path)
    existing_pdfs = (
        {pdf.resolve() for pdf in output_path.glob("*.pdf")}
        if output_path.exists()
        else set()
    )

    config = copy.deepcopy(base_config)
    config["input_file"] = str(pdf_path)
    config["output_path"] = str(output_path)

    logger.info(f"开始处理本地 PDF: {pdf_path}")
    logger.info(f"输出目录: {output_path}")

    split_ok = process_pdf_with_config(config, logger=logger, clear_output=False)
    if not split_ok:
        return False

    generated_pdfs = []
    for generated_pdf in sorted(output_path.glob("*.pdf")):
        if generated_pdf.resolve() not in existing_pdfs:
            generated_pdfs.append(generated_pdf)

    if not generated_pdfs:
        logger.warning(f"当前输入文件未生成新的切分 PDF: {pdf_path}")
        return False

    rename_pdf_files(generated_pdfs, config, logger)
    logger.info(f"PDF 处理完成: {pdf_path}")
    return True


def run_usb_batch(base_config, logger):
    input_dir = Path(base_config.get("input_dir", "./input/"))
    output_root = Path(base_config.get("output_path", "./output/"))
    drive_roots = list_removable_drive_roots()

    if not drive_roots:
        logger.warning("未检测到已插入的可移动U盘。")
        return False

    logger.info("检测到以下U盘: " + ", ".join(str(path) for path in drive_roots))

    copied_pdfs = copy_pdfs_from_usb_drives(drive_roots, input_dir, logger)
    if not copied_pdfs:
        logger.warning(f"未从U盘复制到任何 PDF 文件，本地输入目录: {input_dir}")
        return False

    clear_output_directory(output_root, logger)
    logger.info(
        f"已复制 {len(copied_pdfs)} 个 PDF 到本地输入目录，开始执行切分和重命名。"
    )

    success_count = 0
    failed_files = []
    for pdf_path in copied_pdfs:
        try:
            if process_single_pdf(
                pdf_path, base_config, input_dir, output_root, logger
            ):
                success_count += 1
            else:
                failed_files.append(str(pdf_path))
        except Exception as exc:
            failed_files.append(str(pdf_path))
            logger.exception(f"处理失败: {pdf_path}, error={exc}")

    logger.info(f"批处理结束：成功 {success_count} 个，失败 {len(failed_files)} 个。")
    if failed_files:
        logger.warning("失败文件列表: " + " | ".join(failed_files))

    return success_count > 0
