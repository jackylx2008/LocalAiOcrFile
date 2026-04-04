"""
文件说明：
- 自动扫描已插入 U 盘中的 PDF，复制到本地输入目录后串联执行切分和重命名。

主要职责：
- 枚举 Windows 可移动磁盘并筛选目标日期的 PDF。
- 将源 PDF 复制到本地 input_dir，并避免同名冲突。
- 依次调用切分流程和重命名流程完成批处理。

运行方式：
- 分类：独立运行
- 直接运行命令：python process_usb_pdfs.py
- 直接运行用途：作为整套业务流程的批处理入口，处理 U 盘中的 PDF 文件。
- 被谁调用：通常不作为其他脚本的依赖入口
- 作为依赖用途：无

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
- 关键函数：copy_pdfs_from_usb_drives()、process_single_pdf()、list_removable_drive_roots()

依赖关系：
- 依赖的本项目模块：logging_config.py、rename_pdfs_by_regex.py、split_pdf_keyword.py
- 依赖的第三方库：无

使用提醒：
- 这是面向日常使用的主入口之一，适合“插入 U 盘后批量处理”场景。
- 默认只处理符合当前日期筛选规则的 PDF，不会无条件扫描全部文件。
"""

import argparse
import copy
import ctypes
import os
import shutil
from datetime import date, datetime
from pathlib import Path

from logging_config import setup_logger
from rename_pdfs_by_regex import rename_pdf_files
from split_pdf_keyword import (
    clear_output_directory,
    ensure_project_python,
    load_runtime_config,
    process_pdf_with_config,
)


DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_ENV_PATH = None
SEARCH_RECURSIVELY = True
DRIVE_REMOVABLE = 2
TODAY_MATCH_MODE = "modified"


def clear_directory(directory, logger):
    target_dir = Path(directory)
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"目录不存在，已创建: {target_dir}")
        return

    removed_count = 0
    for item in target_dir.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            else:
                shutil.rmtree(item)
            removed_count += 1
        except Exception as exc:
            logger.warning(f"清理目录项失败: {item}, error={exc}")

    logger.info(f"启动前已清空目录: {target_dir}，删除 {removed_count} 项")


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
    clear_directory(input_dir, logger)
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


def build_output_path(output_root, input_dir, pdf_path):
    return Path(output_root)


def process_single_pdf(pdf_path, base_config, input_dir, output_root, logger):
    output_path = build_output_path(output_root, input_dir, pdf_path)
    existing_pdfs = {pdf.resolve() for pdf in output_path.glob("*.pdf")} if output_path.exists() else set()

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

    input_dir = Path(base_config.get("input_dir", "./input/"))
    output_root = Path(base_config.get("output_path", "./output/"))
    drive_roots = list_removable_drive_roots()

    if not drive_roots:
        logger.warning("未检测到已插入的可移动U盘。")
        return

    logger.info("检测到以下U盘: " + ", ".join(str(path) for path in drive_roots))

    copied_pdfs = copy_pdfs_from_usb_drives(drive_roots, input_dir, logger)
    if not copied_pdfs:
        logger.warning(f"未从U盘复制到任何 PDF 文件，本地输入目录: {input_dir}")
        return

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


if __name__ == "__main__":
    main()
