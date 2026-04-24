"""
文件说明：
- 提供 PDF 首页 OCR 正则匹配、复制输出和文件重命名能力。

主要职责：
- 识别 PDF 首页文本并提取候选命名片段。
- 基于 regex_pattern 规则匹配目标名称。
- 生成安全文件名并完成冲突规避后的复制/重命名。

运行方式：
- 分类：被依赖脚本
- 直接运行命令：不建议直接运行
- 直接运行用途：无独立业务入口，主要由 workflows 和入口脚本调用。
- 被谁调用：workflows.rename_workflow、workflows.usb_batch_workflow
- 作为依赖用途：为批量重命名场景提供核心实现。

输入：
- 配置输入：config 中的 regex_pattern
- 数据输入：PDF 路径列表、PDF 首页 OCR 文本
- 前置条件：OCR 依赖环境可用；传入文件应为有效 PDF

输出：
- 结果输出：重命名后的 PDF 文件
- 日志输出：调用方 logger
- 副作用：可按模式复制文件到输出目录，或直接修改原文件名

核心入口：
- 关键函数：rename_pdf_files()、find_first_regex_match()

依赖关系：
- 依赖的本项目模块：services.ocr_service
- 依赖的第三方库：无

使用提醒：
- 该模块不负责选择默认输入目录，目录扫描策略应交由 workflow 或入口脚本决定。
"""

import re
import shutil
from pathlib import Path

from services.ocr_service import run_startup_self_check


INVALID_FILENAME_CHARS = r'[<>:"/\\|?*]'
RENAME_CONTEXT_KEYWORD = "设计变更通知单"
RENAME_CONTEXT_WINDOW = 80
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def sanitize_filename(name):
    sanitized = re.sub(INVALID_FILENAME_CHARS, "_", str(name))
    sanitized = sanitized.strip().rstrip(".")
    sanitized = re.sub(r"\s+", " ", sanitized)
    sanitized = sanitized or "unnamed"
    if sanitized.upper() in WINDOWS_RESERVED_NAMES:
        sanitized = f"{sanitized}_file"
    return sanitized.upper()


def build_prefixed_filename(prefix, matched_text):
    normalized_prefix = re.sub(r"\s+", " ", str(prefix or "")).strip()
    normalized_matched_text = str(matched_text or "").strip()
    combined_name = (
        f"{normalized_prefix}{normalized_matched_text}"
        if normalized_prefix
        else normalized_matched_text
    )
    return sanitize_filename(combined_name)


def compile_regex_patterns(config):
    patterns = []
    for pattern in config.get("regex_pattern", []) or []:
        pattern = str(pattern).strip()
        if not pattern:
            continue

        relaxed_pattern = pattern[1:] if pattern.startswith("^") else pattern
        patterns.append(
            {
                "raw": pattern,
                "strict": re.compile(pattern),
                "relaxed": re.compile(relaxed_pattern),
            }
        )
    return patterns


def build_match_candidates(page_text):
    lines = [line.strip() for line in str(page_text).splitlines() if line.strip()]
    compact_lines = [re.sub(r"\s+", "", line) for line in lines]
    full_text = "\n".join(lines)
    compact_text = re.sub(r"\s+", "", full_text)
    return lines + compact_lines + [full_text, compact_text]


def get_required_context_distance(candidate, match):
    compact_candidate = re.sub(r"\s+", "", candidate)
    compact_match = re.sub(r"\s+", "", match.group(0))
    compact_keyword = re.sub(r"\s+", "", RENAME_CONTEXT_KEYWORD)

    if not compact_match:
        return None

    match_start = len(re.sub(r"\s+", "", candidate[: match.start()]))
    match_end = match_start + len(compact_match)
    context_start = max(0, match_start - RENAME_CONTEXT_WINDOW)
    context_end = min(len(compact_candidate), match_end + RENAME_CONTEXT_WINDOW)
    context = compact_candidate[context_start:context_end]

    keyword_distances = []
    search_from = 0
    while True:
        keyword_offset = context.find(compact_keyword, search_from)
        if keyword_offset < 0:
            break

        keyword_start = context_start + keyword_offset
        keyword_end = keyword_start + len(compact_keyword)
        if keyword_end <= match_start:
            keyword_distances.append(match_start - keyword_end)

        search_from = keyword_offset + 1

    if not keyword_distances:
        return None
    return min(keyword_distances)


def find_nearest_context_match(candidate, compiled_pattern):
    nearest_match = None
    nearest_distance = None

    for match in compiled_pattern.finditer(candidate):
        distance = get_required_context_distance(candidate, match)
        if distance is None:
            continue
        if nearest_distance is None or distance < nearest_distance:
            nearest_match = match
            nearest_distance = distance

    return nearest_match


def find_first_regex_match(page_text, compiled_patterns):
    seen_candidates = set()
    for candidate in build_match_candidates(page_text):
        if not candidate or candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)

        for pattern_info in compiled_patterns:
            strict_match = find_nearest_context_match(
                candidate, pattern_info["strict"]
            )
            if strict_match:
                return strict_match.group(0), pattern_info["raw"]

            relaxed_match = find_nearest_context_match(
                candidate, pattern_info["relaxed"]
            )
            if relaxed_match:
                return relaxed_match.group(0), pattern_info["raw"]

    return None, None


def ensure_unique_pdf_path(directory, base_name):
    candidate = directory / f"{base_name}.pdf"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{base_name}_{counter}.pdf"
        counter += 1
    return candidate


def ocr_first_page(pdf_path, ocr_processor, logger):
    for page_index, img in ocr_processor.pdf_to_images(pdf_path):
        result, _ = ocr_processor.ocr(img)

        page_text = ""
        if result:
            for line in result:
                page_text += line[1] + "\n"

        text_lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        preview_lines = text_lines[:3]
        if preview_lines:
            logger.info(
                f"文件 {Path(pdf_path).name} 第 {page_index + 1} 页识别文本前 3 行: "
                + " | ".join(preview_lines)
            )
        else:
            logger.info(
                f"文件 {Path(pdf_path).name} 第 {page_index + 1} 页识别文本前 3 行: [无识别文本]"
            )

        return {"page": page_index, "text": page_text}

    return {"page": 0, "text": ""}


def rename_pdf_files(
    pdf_files,
    config,
    logger,
    output_dir=None,
    filename_prefix="",
    in_place=True,
):
    pdf_files = sorted(Path(pdf_file) for pdf_file in pdf_files)
    compiled_patterns = compile_regex_patterns(config)
    unmatched_files = []

    if not pdf_files:
        logger.warning("没有待重命名的 PDF 文件。")
        return

    if not compiled_patterns:
        logger.warning("未配置 regex_pattern，跳过 PDF 首页识别重命名。")
        return

    logger.info("执行启动前自检并初始化 OCR...")
    ocr_processor = run_startup_self_check(config, logger)
    mode_text = "原地重命名" if in_place else "复制到输出目录并重命名"
    logger.info(
        f"开始按首页 OCR 结果处理 PDF，共 {len(pdf_files)} 个文件，模式: {mode_text}。"
    )

    resolved_output_dir = None
    if not in_place:
        resolved_output_dir = Path(output_dir or config.get("output_path", "./output/"))
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_file in pdf_files:
        target_dir = pdf_file.parent if in_place else resolved_output_dir
        first_page_result = ocr_first_page(str(pdf_file), ocr_processor, logger)
        matched_text, matched_pattern = find_first_regex_match(
            first_page_result.get("text", ""), compiled_patterns
        )

        if not matched_text:
            unmatched_files.append(pdf_file.name)
            continue

        safe_name = build_prefixed_filename(filename_prefix, matched_text)
        desired_path = target_dir / f"{safe_name}.pdf"
        if in_place and pdf_file.resolve() == desired_path.resolve():
            logger.info(f"文件名已符合匹配结果，无需重命名: {pdf_file.name}")
            continue

        target_path = ensure_unique_pdf_path(target_dir, safe_name)
        if in_place:
            pdf_file.rename(target_path)
            logger.info(
                f"首页匹配成功，已原地重命名 PDF: {pdf_file.name} -> {target_path.name}, "
                f"matched_text={matched_text}, pattern={matched_pattern}, "
                f"filename_prefix={filename_prefix}"
            )
        else:
            shutil.copy2(pdf_file, target_path)
            logger.info(
                f"首页匹配成功，已复制并重命名 PDF: {pdf_file.name} -> {target_path.name}, "
                f"matched_text={matched_text}, pattern={matched_pattern}, "
                f"filename_prefix={filename_prefix}"
            )

    if unmatched_files:
        logger.warning(
            "以下 PDF 首页未匹配到任何 regex_pattern，未执行重命名: "
            + ", ".join(unmatched_files)
        )
    else:
        logger.info("所有输出 PDF 均已在首页 OCR 阶段匹配到命名规则。")
