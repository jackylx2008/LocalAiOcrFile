"""
文件说明：
- 根据 PDF 首页的 OCR 识别结果与 regex_pattern 规则，对输出文件执行重命名的入口脚本。

主要职责：
- 解析配置并初始化运行环境。
- 调用默认输出目录重命名工作流。
- 为“只重命名、不切分”的日常操作提供直接入口。

运行方式：
- 分类：独立运行
- 直接运行命令：python rename_pdfs_by_regex.py
- 直接运行用途：独立扫描 output_path 下的 PDF，并按首页 OCR 结果执行重命名。
- 被谁调用：通常不作为其他脚本的依赖入口
- 作为依赖用途：无，公共能力已下沉到 workflows.rename_workflow 和 services.pdf_rename_service。

输入：
- 配置输入：config.yaml 中的 output_path、regex_pattern
- 数据输入：output_path 下的 PDF、PDF 首页 OCR 文本
- 前置条件：output_path 中已有待重命名的 PDF；OCR 依赖环境可用

输出：
- 结果输出：重命名后的 PDF 文件
- 日志输出：./log/rename_pdfs_by_regex.log
- 副作用：直接修改 PDF 文件名，不会复制文件

核心入口：
- 主入口函数：main()
- 关键函数：无

依赖关系：
- 依赖的本项目模块：core.config、core.logging_utils、core.runtime、workflows.rename_workflow
- 依赖的第三方库：无

使用提醒：
- 该脚本默认处理 output_path 目录中的 PDF，不读取 input_dir。
- 若首页 OCR 未匹配到任何 regex_pattern，文件会保留原名并在日志中告警。
"""

import os

from core.config import load_config
from core.logging_utils import setup_logger
from core.runtime import ensure_project_python
from workflows.rename_workflow import rename_pdfs_in_output


def main():
    ensure_project_python()

    logger = setup_logger()
    config_path = "config.yaml"

    if not os.path.exists(config_path):
        logger.error(f"配置文件 {config_path} 不存在。")
        return

    config = load_config(config_path)
    rename_pdfs_in_output(config, logger)


if __name__ == "__main__":
    main()
