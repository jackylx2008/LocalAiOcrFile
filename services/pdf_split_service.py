"""
文件说明：
- 根据 OCR 结果和关键词规则，将单个 PDF 拆分为多个子 PDF。

主要职责：
- 判断每页是否同时命中全部切分关键词。
- 计算切分起始页并生成切分区间。
- 将原 PDF 导出为多个子 PDF 文件。

运行方式：
- 分类：被依赖脚本
- 直接运行命令：不建议直接运行
- 直接运行用途：无独立业务入口，主要作为切分模块被主流程脚本导入。
- 被谁调用：workflows.split_workflow
- 作为依赖用途：为单文件切分流程提供 PDF 拆分能力。

输入：
- 配置输入：config 中的 split_keywords / ocr.split_keywords、output_path
- 数据输入：输入 PDF 路径、OCR 结果列表
- 前置条件：输入 PDF 存在；ocr_results 已由上游流程生成

输出：
- 结果输出：多个切分后的 PDF 文件
- 日志输出：./log/splitter.log
- 副作用：在 output_path 中创建文件；必要时自动创建输出目录

核心入口：
- 主入口函数：无固定业务主入口
- 关键类：PDFSplitter
- 关键函数：split_by_ocr_results()

依赖关系：
- 依赖的本项目模块：core.logging_utils
- 依赖的第三方库：PyMuPDF

使用提醒：
- 该模块不负责 OCR，只消费上游已生成的 OCR 结果。
- 切分规则要求“同一页同时命中全部关键词”才会被视为新的起始页。
"""

import os
import fitz  # PyMuPDF
from core.logging_utils import setup_logger

logger = setup_logger(log_file="./log/splitter.log")


class PDFSplitter:
    def __init__(self, config):
        self.config = config
        self.keywords = config.get("split_keywords") or config.get("ocr", {}).get(
            "split_keywords", []
        )
        self.output_path = config.get("output_path", "./output/")

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            logger.info(f"创建输出目录: {self.output_path}")

    def split_by_ocr_results(self, input_pdf_path, ocr_results):
        """
        基于 OCR 结果进行切分。
        当某一页同时包含全部关键字时，该页作为新文档的起始。
        """
        if not ocr_results:
            logger.warning("没有 OCR 结果，无法切分。")
            return

        doc = fitz.open(input_pdf_path)
        total_pages = len(doc)

        # 记录切分点的起始页码
        split_points = []
        for res in ocr_results:
            page_index = res["page"]
            text = res["text"]

            normalized_text = "".join(str(text).split())

            if not self.keywords:
                continue

            valid_keywords = [kw for kw in self.keywords if str(kw).strip()]
            matched_keywords = []
            for kw in valid_keywords:
                normalized_kw = "".join(str(kw).split())
                if normalized_kw and normalized_kw in normalized_text:
                    matched_keywords.append(kw)

            if len(valid_keywords) > 0 and len(matched_keywords) == len(valid_keywords):
                logger.info(
                    f"第 {page_index + 1} 页同时命中全部切分关键字: {matched_keywords}"
                )
                split_points.append(page_index)

        # 如果第一页不是关键字页，默认加上第一页作为起始点
        if not split_points or split_points[0] != 0:
            split_points.insert(0, 0)

        # 根据切分点保存文件
        num_splits = len(split_points)
        for i in range(num_splits):
            start_page = split_points[i]
            # 结束页码是下一个切分点，或者是文档最后一页
            end_page = split_points[i + 1] if i + 1 < num_splits else total_pages

            new_doc = fitz.open()
            # insert_pdf 的页面范围是 [start, end)
            new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)

            input_filename = os.path.basename(input_pdf_path).replace(".pdf", "")
            output_filename = f"{input_filename}_part_{i + 1}_page_{start_page + 1}.pdf"
            save_path = os.path.join(self.output_path, output_filename)

            new_doc.save(save_path)
            new_doc.close()
            logger.info(
                f"已导出切分文件: {save_path} (包含页码 {start_page + 1} 到 {end_page})"
            )

        doc.close()
        logger.info(f"PDF 处理完成，共切分为 {num_splits} 个文件。")
