"""
文件说明：
- 提供 U 盘扫描、PDF 筛选和本地复制能力。

主要职责：
- 枚举 Windows 可移动磁盘。
- 按日期规则筛选 PDF 文件。
- 将目标 PDF 复制到本地输入目录并处理重名。

运行方式：
- 分类：被依赖脚本
- 直接运行命令：不建议直接运行
- 直接运行用途：无独立业务入口，主要被 USB 批处理工作流调用。
- 被谁调用：workflows.usb_batch_workflow
- 作为依赖用途：提供 U 盘扫描和输入文件准备能力。

输入：
- 配置输入：目标日期匹配模式、是否递归扫描
- 数据输入：U 盘根目录、本地 input_dir
- 前置条件：运行环境为 Windows 时才能扫描可移动磁盘

输出：
- 结果输出：复制到本地的 PDF 路径列表
- 日志输出：调用方 logger
- 副作用：清空 input_dir，并复制文件到本地

核心入口：
- 关键函数：list_removable_drive_roots()、copy_pdfs_from_usb_drives()

依赖关系：
- 依赖的本项目模块：services.file_ops_service
- 依赖的第三方库：无

使用提醒：
- 默认按“文件修改日期等于今天”筛选 PDF。
"""

import ctypes
import os
import shutil
from datetime import date, datetime
from pathlib import Path

from services.file_ops_service import clear_directory


SEARCH_RECURSIVELY = True
DRIVE_REMOVABLE = 2
TODAY_MATCH_MODE = "modified"


def list_removable_drive_roots():
    if os.name != "nt":
        return []

    drive_roots = []
    kernel32 = ctypes.windll.kernel32
    drive_bitmask = kernel32.GetLogicalDrives()

    for index in range(26):
        if not (drive_bitmask & (1 << index)):
            continue

        drive_letter = chr(ord("A") + index)
        drive_root = f"{drive_letter}:\\"
        drive_type = kernel32.GetDriveTypeW(ctypes.c_wchar_p(drive_root))
        if drive_type == DRIVE_REMOVABLE:
            drive_roots.append(Path(drive_root))

    return sorted(drive_roots, key=lambda path: str(path).lower())


def find_pdf_files(root_dir):
    pdf_files = []

    if SEARCH_RECURSIVELY:
        for current_root, _, filenames in os.walk(root_dir):
            current_root = Path(current_root)
            for filename in filenames:
                if filename.lower().endswith(".pdf"):
                    pdf_files.append(current_root / filename)
    else:
        pdf_files.extend(root_dir.glob("*.pdf"))

    return sorted(pdf_files, key=lambda path: str(path).lower())


def file_matches_target_date(pdf_path, target_date):
    stat_result = pdf_path.stat()
    modified_date = datetime.fromtimestamp(stat_result.st_mtime).date()

    if TODAY_MATCH_MODE == "modified":
        return modified_date == target_date

    return False


def build_local_input_path(input_dir, source_pdf):
    modified_at = datetime.fromtimestamp(source_pdf.stat().st_mtime)
    timestamp_text = modified_at.strftime("%Y%m%d_%H%M%S")
    base_name = f"{source_pdf.stem}_{timestamp_text}"
    target_pdf = Path(input_dir) / f"{base_name}.pdf"
    counter = 2

    while target_pdf.exists():
        target_pdf = Path(input_dir) / f"{base_name}_{counter}.pdf"
        counter += 1

    return target_pdf


def copy_pdfs_from_usb_drives(drive_roots, input_dir, logger):
    clear_directory(input_dir, logger, label="输入目录")
    target_date = date.today()

    copied_files = []
    for drive_root in drive_roots:
        pdf_files = []
        for pdf_path in find_pdf_files(drive_root):
            try:
                if file_matches_target_date(pdf_path, target_date):
                    pdf_files.append(pdf_path)
            except OSError:
                logger.warning(f"读取 PDF 时间信息失败，已跳过: {pdf_path}")

        if not pdf_files:
            logger.info(
                f"U盘 {drive_root} 中未找到日期为 {target_date.isoformat()} 的 PDF 文件。"
            )
            continue

        logger.info(
            f"U盘 {drive_root} 中找到 {len(pdf_files)} 个修改日期为 {target_date.isoformat()} 的 PDF 文件。"
        )
        for source_pdf in pdf_files:
            try:
                target_pdf = build_local_input_path(input_dir, source_pdf)
                target_pdf.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_pdf, target_pdf)
                copied_files.append(target_pdf)
                logger.info(f"已复制 PDF: {source_pdf} -> {target_pdf}")
            except Exception as exc:
                logger.warning(f"复制 PDF 失败: {source_pdf}, error={exc}")

    return copied_files
