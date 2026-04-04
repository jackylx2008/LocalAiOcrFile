"""
文件说明：
- 提供 PDF 首页 OCR 正则匹配和文件重命名能力。

主要职责：
- 识别 PDF 首页文本并提取候选命名片段。
- 基于 regex_pattern 规则匹配目标名称。
- 生成安全文件名并完成冲突规避后的重命名。

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
- 副作用：直接修改文件名

核心入口：
- 关键函数：rename_pdf_files()、find_first_regex_match()

依赖关系：
- 依赖的本项目模块：services.ocr_service
- 依赖的第三方库：无

使用提醒：
- 该模块不负责选择默认输入目录，目录扫描策略应交由 workflow 或入口脚本决定。
"""

import re
from pathlib import Path

from services.ocr_service import run_startup_self_check


INVALID_FILENAME_CHARS = r'[<>:"/\\|?*]'
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
    return sanitized


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


def find_first_regex_match(page_text, compiled_patterns):
    seen_candidates = set()
    for candidate in build_match_candidates(page_text):
        if not candidate or candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)

        for pattern_info in compiled_patterns:
            strict_match = pattern_info["strict"].search(candidate)
            if strict_match:
                return strict_match.group(0), pattern_info["raw"]

            relaxed_match = pattern_info["relaxed"].search(candidate)
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


def rename_pdf_files(pdf_files, config, logger):
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
    logger.info(f"开始按首页 OCR 结果重命名 PDF，共 {len(pdf_files)} 个文件。")

    for pdf_file in pdf_files:
        output_dir = pdf_file.parent
        first_page_result = ocr_first_page(str(pdf_file), ocr_processor, logger)
        matched_text, matched_pattern = find_first_regex_match(
            first_page_result.get("text", ""), compiled_patterns
        )

        if not matched_text:
            unmatched_files.append(pdf_file.name)
            continue

        safe_name = sanitize_filename(matched_text)
        desired_path = output_dir / f"{safe_name}.pdf"
        if pdf_file.resolve() == desired_path.resolve():
            logger.info(f"文件名已符合匹配结果，无需重命名: {pdf_file.name}")
            continue

        target_path = ensure_unique_pdf_path(output_dir, safe_name)
        pdf_file.rename(target_path)
        logger.info(
            f"首页匹配成功，已重命名 PDF: {pdf_file.name} -> {target_path.name}, "
            f"matched_text={matched_text}, pattern={matched_pattern}"
        )

    if unmatched_files:
        logger.warning(
            "以下 PDF 首页未匹配到任何 regex_pattern，未执行重命名: "
            + ", ".join(unmatched_files)
        )
    else:
        logger.info("所有输出 PDF 均已在首页 OCR 阶段匹配到命名规则。")
