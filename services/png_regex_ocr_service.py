"""
文件说明：
- 提供 PNG 图片 OCR 识别与 regex_pattern 匹配统计能力。

主要职责：
- 读取 PNG 图片并执行 OCR。
- 基于 regex_pattern 提取所有命中字符串。
- 汇总命中次数并导出到 TXT 文件。

运行方式：
- 分类：被依赖脚本
- 直接运行命令：不建议直接运行
- 直接运行用途：无独立业务入口，主要被 workflow 调用。
- 被谁调用：workflows.png_regex_workflow
- 作为依赖用途：为 PNG OCR 统计流程提供核心实现。

输入：
- 配置输入：regex_pattern、ocr 配置
- 数据输入：PNG 文件路径列表
- 前置条件：OCR 环境可用；PNG 文件存在且可读取

输出：
- 结果输出：匹配统计结果、TXT 文件
- 日志输出：调用方 logger
- 副作用：写出统计 TXT 文件

核心入口：
- 关键函数：extract_regex_matches_from_pngs()、write_match_counts_to_txt()

依赖关系：
- 依赖的本项目模块：services.ocr_service、services.pdf_rename_service
- 依赖的第三方库：opencv-python
"""

from collections import Counter
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

from services.ocr_service import run_startup_self_check
from services.pdf_rename_service import compile_regex_patterns


def _build_match_candidates(page_text):
    lines = [line.strip() for line in str(page_text).splitlines() if line.strip()]
    compact_lines = [re.sub(r"\s+", "", line) for line in lines]
    full_text = "\n".join(lines)
    compact_text = re.sub(r"\s+", "", full_text)
    return lines + compact_lines + [full_text, compact_text]


def _extract_all_regex_matches(page_text, compiled_patterns):
    matches = []
    seen_items = set()
    seen_candidates = set()

    for candidate in _build_match_candidates(page_text):
        if not candidate or candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)

        for pattern_info in compiled_patterns:
            for matched_text in pattern_info["strict"].findall(candidate):
                normalized = _normalize_regex_findall_result(matched_text)
                if normalized and normalized not in seen_items:
                    matches.append(normalized)
                    seen_items.add(normalized)

            for matched_text in pattern_info["relaxed"].findall(candidate):
                normalized = _normalize_regex_findall_result(matched_text)
                if normalized and normalized not in seen_items:
                    matches.append(normalized)
                    seen_items.add(normalized)

    return matches


def _normalize_regex_findall_result(matched_text):
    if isinstance(matched_text, tuple):
        return "-".join(str(part) for part in matched_text if str(part))
    return str(matched_text).strip()


def _ocr_png_file(png_path, ocr_processor, logger):
    image = _read_image_with_unicode_path(png_path, logger)
    if image is None:
        logger.warning(f"读取图片失败，已跳过: {png_path}")
        return ""

    original_text = _run_ocr_on_image(image, ocr_processor)
    enhanced_image = _build_enhanced_ocr_image(image)
    enhanced_text = _run_ocr_on_image(enhanced_image, ocr_processor)
    page_text, selected_variant = _select_better_ocr_text(
        original_text, enhanced_text
    )

    text_lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    preview_lines = text_lines[:3]
    compact_text = re.sub(r"\s+", "", page_text)
    logger.info(
        f"文件 {Path(png_path).name} OCR 字符统计: 原始长度={len(page_text)}, "
        f"去空白长度={len(compact_text)}, 有效行数={len(text_lines)}"
    )
    logger.info(
        f"文件 {Path(png_path).name} OCR 结果选择: selected={selected_variant}, "
        f"original_len={len(original_text)}, enhanced_len={len(enhanced_text)}"
    )
    if preview_lines:
        logger.info(
            f"文件 {Path(png_path).name} 识别文本前 3 行: " + " | ".join(preview_lines)
        )
    else:
        logger.info(f"文件 {Path(png_path).name} 识别文本前 3 行: [无识别文本]")

    return page_text


def _run_ocr_on_image(image, ocr_processor):
    result, _ = ocr_processor.ocr(image)
    page_text = ""
    if result:
        for line in result:
            page_text += line[1] + "\n"
    return page_text


def _build_enhanced_ocr_image(image):
    enlarged = cv2.resize(
        image,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 5, 50, 50)
    sharpened = cv2.addWeighted(denoised, 1.6, cv2.GaussianBlur(denoised, (0, 0), 3), -0.6, 0)
    thresholded = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)


def _select_better_ocr_text(original_text, enhanced_text):
    original_score = _score_ocr_text(original_text)
    enhanced_score = _score_ocr_text(enhanced_text)
    if enhanced_score > original_score:
        return enhanced_text, "enhanced"
    return original_text, "original"


def _score_ocr_text(text):
    compact_text = re.sub(r"\s+", "", str(text))
    if not compact_text:
        return 0

    line_count = len([line for line in str(text).splitlines() if line.strip()])
    digit_count = sum(char.isdigit() for char in compact_text)
    dash_count = compact_text.count("-")
    alpha_count = sum(char.isalpha() for char in compact_text)
    return (
        len(compact_text) * 10
        + line_count * 5
        + digit_count * 3
        + dash_count * 6
        + alpha_count
    )


def _read_image_with_unicode_path(image_path, logger):
    path = Path(image_path)
    if not path.exists():
        logger.warning(f"图片路径不存在: {path}")
        return None

    try:
        file_size = path.stat().st_size
        logger.info(f"读取图片文件信息: path={path}, size={file_size} bytes")
    except OSError as exc:
        logger.warning(f"读取图片文件信息失败: {path}, error={exc}")
        file_size = None

    try:
        image_bytes = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        logger.warning(f"读取图片字节失败: {path}, error={exc}")
        image_bytes = np.array([], dtype=np.uint8)

    if image_bytes.size == 0:
        logger.warning(f"图片文件为空、未下载完成或不可读取: {path}")
    else:
        logger.info(f"读取图片字节成功: path={path}, byte_count={image_bytes.size}")

    image = (
        cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image_bytes.size > 0
        else None
    )
    if image is not None:
        return image

    logger.warning(f"OpenCV 解码图片失败，尝试使用 PIL 兜底读取: {path}")

    try:
        with Image.open(path) as pil_image:
            rgb_image = pil_image.convert("RGB")
            image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
            logger.info(
                f"PIL 兜底读取成功: path={path}, size={rgb_image.size}, mode={rgb_image.mode}"
            )
            return image
    except Exception as exc:
        logger.warning(
            f"PIL 兜底读取也失败，图片可能损坏、未本地下载或格式异常: {path}, error={exc}"
        )

    if file_size == 0:
        logger.warning(f"图片文件大小为 0，无法 OCR: {path}")
    return None


def extract_regex_matches_from_pngs(png_files, config, logger):
    png_files = sorted(Path(path) for path in png_files)
    compiled_patterns = compile_regex_patterns(config)
    configured_patterns = [pattern_info["raw"] for pattern_info in compiled_patterns]
    if not png_files:
        logger.warning("没有待识别的图片文件。")
        return Counter(), []

    if not compiled_patterns:
        logger.warning("未配置 regex_pattern，跳过 PNG OCR 统计。")
        return Counter(), []

    logger.info("执行启动前自检并初始化 OCR...")
    ocr_processor = run_startup_self_check(config, logger)
    logger.info(
        "PNG OCR 当前启用 regex_pattern: " + " | ".join(configured_patterns)
    )

    match_counter = Counter()
    unmatched_files = []
    for png_file in png_files:
        page_text = _ocr_png_file(png_file, ocr_processor, logger)
        matches = _extract_all_regex_matches(page_text, compiled_patterns)

        if not matches:
            unmatched_files.append(png_file.name)
            candidate_lengths = []
            for candidate in _build_match_candidates(page_text):
                if candidate:
                    candidate_lengths.append(str(len(candidate)))
            logger.warning(
                f"文件 {png_file.name} 未匹配到 regex_pattern，候选文本长度: "
                + (", ".join(candidate_lengths) if candidate_lengths else "[无候选文本]")
            )
            logger.debug(f"文件 {png_file.name} 完整 OCR 文本:\n{page_text}")
            continue

        for matched_text in matches:
            match_counter[matched_text] += 1

        logger.info(
            f"文件 {png_file.name} 命中 regex_pattern {len(matches)} 项: "
            + " | ".join(matches)
        )

    if unmatched_files:
        logger.warning(
            "以下图片未匹配到任何 regex_pattern: " + ", ".join(unmatched_files)
        )

    return match_counter, unmatched_files


def write_match_counts_to_txt(match_counter, output_txt_path, logger):
    output_path = Path(output_txt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for matched_text, count in sorted(
        match_counter.items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"{matched_text}\t{count}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"PNG OCR 统计 TXT 已生成: {output_path}")
    return output_path
