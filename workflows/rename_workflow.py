"""
文件说明：
- 编排输入目录扫描、复制输出和 PDF 重命名流程。

主要职责：
- 根据配置和命令行选择输入目录与输出目录。
- 收集待重命名 PDF 文件。
- 调用重命名服务完成首页 OCR 正则处理。

运行方式：
- 分类：被依赖脚本
- 直接运行命令：不建议直接运行
- 直接运行用途：无独立业务入口，主要被重命名入口脚本调用。
- 被谁调用：rename_pdfs_by_regex.py
- 作为依赖用途：提供默认“扫描 input_path 并输出到 output_path”的工作流。

输入：
- 配置输入：input_path、output_path、regex_pattern、in_place
- 数据输入：input_path 下的 PDF
- 前置条件：输入目录中已有待处理的 PDF

输出：
- 结果输出：重命名后的 PDF 文件
- 日志输出：调用方 logger
- 副作用：按模式复制文件到输出目录或修改原文件名

核心入口：
- 关键函数：rename_pdfs()

依赖关系：
- 依赖的本项目模块：services.pdf_rename_service
- 依赖的第三方库：无
"""

from pathlib import Path

from services.pdf_rename_service import rename_pdf_files


def _collect_pdf_files(input_path):
    source_dir = Path(input_path)
    return sorted(path for path in source_dir.glob("*.pdf") if path.is_file())


def rename_pdfs(
    config,
    logger,
    input_path=None,
    output_path=None,
    filename_prefix="",
    in_place=False,
):
    default_source = config.get(
        "rename_input_path", config.get("output_path", "./output/")
    )
    source_dir = Path(input_path or default_source).resolve()
    target_dir = Path(output_path or config.get("output_path", "./output/")).resolve()

    if not source_dir.exists():
        logger.error(f"输入目录不存在: {source_dir}")
        return False

    if not source_dir.is_dir():
        logger.error(f"输入路径不是目录: {source_dir}")
        return False

    if in_place and output_path is not None:
        logger.error("启用 --in-place 时不能同时指定 --output-path。")
        return False

    if not in_place and source_dir == target_dir:
        logger.error(
            "非原地重命名模式下，输入目录和输出目录不能相同，否则会与“保留原文件”逻辑冲突。"
        )
        return False

    pdf_files = _collect_pdf_files(source_dir)
    if not pdf_files:
        logger.warning(f"输入目录中没有待重命名的 PDF: {source_dir}")
        return False

    rename_pdf_files(
        pdf_files,
        config,
        logger,
        output_dir=target_dir,
        filename_prefix=filename_prefix,
        in_place=in_place,
    )
    return True
