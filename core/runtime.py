"""
文件说明：
- 提供项目运行时相关的公共工具，如自动切换到项目内 Python 解释器。

主要职责：
- 检测当前解释器是否为项目内 .conda/python.exe。
- 在需要时自动重启当前脚本到项目解释器。

运行方式：
- 分类：被依赖脚本
- 直接运行命令：不建议直接运行
- 直接运行用途：无独立业务入口，主要被各入口脚本导入。
- 被谁调用：split_pdf_keyword.py、rename_pdfs_by_regex.py、process_usb_pdfs.py
- 作为依赖用途：统一管理解释器切换逻辑。

输入：
- 配置输入：无
- 数据输入：当前进程解释器路径、命令行参数
- 前置条件：项目根目录下存在 .conda/python.exe 时才会触发切换

输出：
- 结果输出：无
- 日志输出：无
- 副作用：可能通过 os.execv 重启当前进程

核心入口：
- 关键函数：ensure_project_python()

依赖关系：
- 依赖的本项目模块：无
- 依赖的第三方库：无

使用提醒：
- 该函数应在入口脚本 main() 最开始调用。
"""

import os
import sys
from pathlib import Path


def ensure_project_python():
    project_root = Path(__file__).resolve().parents[1]
    project_python = project_root / ".conda" / "python.exe"
    if not project_python.exists():
        return

    current_python = Path(sys.executable).resolve()
    target_python = project_python.resolve()
    if os.path.normcase(str(current_python)) == os.path.normcase(str(target_python)):
        return

    print(f"检测到当前解释器为 {current_python}，切换到项目解释器 {target_python}")
    os.execv(str(target_python), [str(target_python), *sys.argv])
