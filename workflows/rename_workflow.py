"""
文件说明：
- 编排输出目录扫描与 PDF 重命名流程。

主要职责：
- 根据配置选择 output_path 目录。
- 收集待重命名 PDF 文件。
- 调用重命名服务完成首页 OCR 正则重命名。

运行方式：
- 分类：被依赖脚本
- 直接运行命令：不建议直接运行
- 直接运行用途：无独立业务入口，主要被重命名入口脚本调用。
- 被谁调用：rename_pdfs_by_regex.py
- 作为依赖用途：提供默认“扫描 output_path 并重命名”的工作流。

输入：
- 配置输入：output_path、regex_pattern
- 数据输入：output_path 下的 PDF
- 前置条件：输出目录中已有待重命名的 PDF

输出：
- 结果输出：重命名后的 PDF 文件
- 日志输出：调用方 logger
- 副作用：修改输出目录中文件名

核心入口：
- 关键函数：rename_pdfs_in_output()

依赖关系：
- 依赖的本项目模块：services.pdf_rename_service
- 依赖的第三方库：无
"""

from pathlib import Path

from services.pdf_rename_service import rename_pdf_files


def rename_pdfs_in_output(config, logger):
    output_dir = Path(config.get("output_path", "./output/"))
    pdf_files = sorted(output_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"输出目录中没有待重命名的 PDF: {output_dir}")
        return

    rename_pdf_files(pdf_files, config, logger)
