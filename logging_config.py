"""
文件说明：
- 提供项目统一的日志初始化能力，按脚本名自动生成对应日志文件。

主要职责：
- 初始化控制台与文件双通道日志。
- 统一日志格式、日志级别和滚动文件策略。
- 为各业务脚本提供可复用的 setup_logger()。

运行方式：
- 分类：两者都可以
- 直接运行命令：python logging_config.py
- 直接运行用途：仅用于日志配置自测，会输出测试日志到 ./log/test_logger.log。
- 被谁调用：split_pdf_keyword.py、process_usb_pdfs.py、rename_pdfs_by_regex.py、ocr_engine.py、splitter.py
- 作为依赖用途：为其他脚本提供统一日志记录器。

输入：
- 配置输入：函数参数 log_level、log_file、filemode
- 数据输入：sys.argv[0] 用于推断默认日志文件名
- 前置条件：当前进程对 ./log 目录具有写权限

输出：
- 结果输出：logging.Logger 根记录器
- 日志输出：./log/*.log
- 副作用：清理并重建根日志处理器，影响当前进程全局日志配置

核心入口：
- 主入口函数：无固定业务主入口
- 关键函数：setup_logger()

依赖关系：
- 依赖的本项目模块：无
- 依赖的第三方库：无

使用提醒：
- 该模块会操作根日志记录器，重复初始化会覆盖已有 handler。
- 直接运行仅用于调试日志配置，不是项目主业务入口。
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

# 动态添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def setup_logger(
    log_level: int = logging.DEBUG,
    log_file: Optional[str] = None,
    filemode: str = "w",
):
    """
    设置日志记录器。

    :param log_level: 日志级别，默认为 DEBUG。
    :param log_file: 日志文件路径，如果为 None，则根据主模块名自动生成，如 ./log/main.log。
    :param filemode: 文件打开模式，默认为 'w' (覆盖)。
    :return: 配置好的日志记录器。
    """
    # 如果未显式指定日志文件，则按“一个脚本一个日志”规则自动生成
    if log_file is None:
        main_module = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "app"
        log_file = os.path.join(".", "log", f"{main_module}.log")

    # 创建日志文件夹
    log_dir = os.path.dirname(log_file) or "."
    os.makedirs(log_dir, exist_ok=True)

    # 配置日志格式
    log_format = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"

    # 设置日志级别
    app_logger = logging.getLogger()
    app_logger.setLevel(log_level)

    # 清除已有的处理器，避免重复添加
    if app_logger.handlers:
        app_logger.handlers.clear()

    # 控制台日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))

    # 文件日志处理器（使用滚动记录，防止单个文件过大）
    # maxBytes=10MB, backupCount=5
    file_handler = RotatingFileHandler(
        log_file,
        mode="a",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(log_format))

    # 添加处理器
    app_logger.addHandler(console_handler)
    app_logger.addHandler(file_handler)

    return app_logger


# 单独运行时的测试代码
if __name__ == "__main__":
    # 示例日志文件路径
    LOG_FILE_PATH = "./log/test_logger.log"

    # 初始化日志记录器
    logger = setup_logger(log_level=logging.INFO, log_file=LOG_FILE_PATH)

    # 测试日志输出
    logger.debug("This is a DEBUG message.")
    logger.info("This is an INFO message.")
    logger.warning("This is a WARNING message.")
    logger.error("This is an ERROR message.")
    logger.critical("This is a CRITICAL message.")
