"""
文件说明：
- 负责读取 config.yaml 和 common.env，并完成环境变量模板替换。

主要职责：
- 解析 .env 风格的环境变量文件。
- 将 config.yaml 中的 ${VAR} / ${VAR:-default} 渲染为实际值。
- 输出可供业务脚本直接使用的配置字典。

运行方式：
- 分类：被依赖脚本
- 直接运行命令：不建议直接运行
- 直接运行用途：无独立业务入口，主要作为配置加载模块被其他脚本导入。
- 被谁调用：split_pdf_keyword.py、rename_pdfs_by_regex.py
- 作为依赖用途：为主流程脚本提供统一配置加载能力。

输入：
- 配置输入：config.yaml、common.env、系统环境变量
- 数据输入：配置文件文本、环境变量文件文本
- 前置条件：config.yaml 需为合法 YAML；common.env 需为 KEY=VALUE 格式

输出：
- 结果输出：dict 配置对象
- 日志输出：无
- 副作用：无

核心入口：
- 主入口函数：load_config()
- 关键函数：parse_env_file()、render_config_template()

依赖关系：
- 依赖的本项目模块：无
- 依赖的第三方库：PyYAML

使用提醒：
- 该模块只负责加载与渲染配置，不负责校验业务字段是否完整。
- 当 config.yaml 中引用了未提供默认值的环境变量时，会抛出异常而不是静默跳过。
"""

import os
import re
from pathlib import Path

import yaml


ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def parse_env_file(env_path):
    env_values = {}
    if not env_path.exists():
        return env_values

    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            raise ValueError(
                f"环境变量文件格式错误: {env_path} 第 {line_number} 行缺少 '='"
            )

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            raise ValueError(
                f"环境变量文件格式错误: {env_path} 第 {line_number} 行变量名为空"
            )

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        env_values[key] = value

    return env_values


def render_config_template(config_text, env_values):
    missing_variables = set()

    def replace_env_var(match):
        var_name = match.group(1)
        default_value = match.group(2)

        if var_name in env_values:
            return env_values[var_name]
        if default_value is not None:
            return default_value

        missing_variables.add(var_name)
        return match.group(0)

    rendered = ENV_VAR_PATTERN.sub(replace_env_var, config_text)

    if missing_variables:
        missing_names = ", ".join(sorted(missing_variables))
        raise ValueError(f"config.yaml 缺少环境变量: {missing_names}")

    return rendered


def load_config(config_path, env_path=None):
    config_file = Path(config_path)
    if env_path is None:
        env_file = config_file.resolve().parent / "common.env"
    else:
        env_file = Path(env_path)

    file_env = parse_env_file(env_file)
    merged_env = {**file_env, **os.environ}

    rendered_config = render_config_template(
        config_file.read_text(encoding="utf-8"), merged_env
    )
    loaded = yaml.safe_load(rendered_config)
    return loaded or {}
