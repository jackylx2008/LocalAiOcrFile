"""
文件说明：
- 提供目录清理等通用文件操作能力，供多个工作流复用。

主要职责：
- 按目录级别清理已有文件和子目录。
- 为输出目录清理提供统一包装。

运行方式：
- 分类：被依赖脚本
- 直接运行命令：不建议直接运行
- 直接运行用途：无独立业务入口，主要被 workflows 调用。
- 被谁调用：workflows.split_workflow、workflows.usb_batch_workflow、services.usb_scan_service
- 作为依赖用途：统一管理目录清理逻辑。

输入：
- 配置输入：无
- 数据输入：目标目录路径、logger、目录标签
- 前置条件：当前进程对目标目录具有读写权限

输出：
- 结果输出：无
- 日志输出：调用方 logger
- 副作用：删除目录下已有文件、符号链接和子目录

核心入口：
- 关键函数：clear_directory()、clear_output_directory()

依赖关系：
- 依赖的本项目模块：无
- 依赖的第三方库：无

使用提醒：
- 该模块包含删除操作，只应用于明确允许清理的工作目录。
"""

import shutil
from pathlib import Path


def clear_directory(directory, logger, label="目录"):
    target_dir = Path(directory)
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"{label}不存在，已创建: {target_dir}")
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
            logger.warning(f"清理{label}项失败: {item}, error={exc}")

    logger.info(f"启动前已清空{label}: {target_dir}，删除 {removed_count} 项")


def clear_output_directory(output_path, logger):
    clear_directory(output_path, logger, label="输出目录")
