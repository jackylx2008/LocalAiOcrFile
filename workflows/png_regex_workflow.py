"""
文件说明：
- 编排目标目录下 PNG 图片的 OCR 识别与正则统计流程。

主要职责：
- 解析 PNG OCR 工作流目标目录与 TXT 输出路径配置。
- 收集目录中的 PNG 文件并执行 OCR 识别。
- 汇总 regex_pattern 命中结果并导出 TXT。

运行方式：
- 分类：被依赖脚本
- 直接运行命令：不建议直接运行
- 直接运行用途：无独立业务入口，主要被 png_regex_ocr.py 调用。
- 被谁调用：png_regex_ocr.py
- 作为依赖用途：提供 PNG OCR 正则统计工作流。

输入：
- 配置输入：png_ocr_input_path、png_ocr_output_txt_path、regex_pattern、ocr
- 数据输入：目标目录中的 PNG 文件
- 前置条件：目标目录存在；OCR 环境可用

输出：
- 结果输出：统计 TXT 文件
- 日志输出：调用方 logger
- 副作用：写出统计 TXT 文件

核心入口：
- 关键函数：run_png_regex_workflow()

依赖关系：
- 依赖的本项目模块：services.png_regex_ocr_service
- 依赖的第三方库：无
"""

from pathlib import Path

from services.png_regex_ocr_service import (
    extract_regex_matches_from_pngs,
    write_match_counts_to_txt,
)


def _collect_image_files(input_path):
    source_dir = Path(input_path)
    image_suffixes = {".png", ".jpg", ".jpeg"}
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_suffixes
    )


def run_png_regex_workflow(config, logger, input_path=None, output_txt_path=None):
    source_dir = Path(input_path or config.get("png_ocr_input_path", "./input/")).resolve()
    txt_path = Path(
        output_txt_path
        or config.get("png_ocr_output_txt_path")
        or source_dir / "png_ocr_regex_matches.txt"
    ).resolve()

    if not source_dir.exists():
        logger.error(f"PNG OCR 输入目录不存在: {source_dir}")
        return False

    if not source_dir.is_dir():
        logger.error(f"PNG OCR 输入路径不是目录: {source_dir}")
        return False

    if txt_path.exists():
        try:
            txt_path.unlink()
            logger.info(f"启动前已清空旧的 PNG OCR 统计 TXT: {txt_path}")
        except OSError as exc:
            logger.exception(f"清空旧的 PNG OCR 统计 TXT 失败: {txt_path}, error={exc}")
            return False

    png_files = _collect_image_files(source_dir)
    if not png_files:
        logger.warning(f"PNG OCR 输入目录中没有可识别图片文件(.png/.jpg/.jpeg): {source_dir}")
        return False

    logger.info(
        f"开始处理 PNG OCR 统计，目标目录: {source_dir}，图片数量: {len(png_files)}，输出 TXT: {txt_path}"
    )

    try:
        match_counter, unmatched_files = extract_regex_matches_from_pngs(
            png_files, config, logger
        )
    except Exception as exc:
        logger.exception(f"PNG OCR 识别与统计失败: {source_dir}, error={exc}")
        return False

    try:
        txt_result = write_match_counts_to_txt(match_counter, txt_path, logger)
    except Exception as exc:
        logger.exception(f"写出 PNG OCR 统计 TXT 失败: {txt_path}, error={exc}")
        return False

    logger.info(
        f"PNG OCR 统计结束：匹配字符串 {sum(match_counter.values())} 次，唯一项 {len(match_counter)} 个，"
        f"未匹配文件 {len(unmatched_files)} 个。"
    )
    return bool(txt_result)
