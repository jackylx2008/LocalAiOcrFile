# SplitPdfByKeyWord

基于 OCR 的 PDF 自动切分工具：先逐页识别文本，再按关键字规则切分并导出子 PDF。

## 文档导航

- CUDA/Python 环境配置手册：[`CUDA_PYTHON_SETUP.md`](CUDA_PYTHON_SETUP.md)
- Copilot 仓库指令：[`.github/copilot-instructions.md`](.github/copilot-instructions.md)
- 运行配置文件：[`config.yaml`](config.yaml)
- 本地环境变量：[`common.env`](common.env)

## 核心功能

- 使用 `rapidocr-onnxruntime` 对 PDF 每页 OCR 识别。
- 启动前自动执行环境自检，输出：
  - GPU 可用性
  - CUDA DLL 搜索路径
  - 关键 DLL 路径
  - `det/rec/cls` provider 状态
- 按 `split_keywords` 执行切分：
  - **同一页需要同时命中全部关键词** 才作为起始页。
  - 匹配时会忽略空白（空格、换行）差异，提高 OCR 噪声下命中率。
- 每页识别后，日志会打印该页识别文本前 3 行。
- 每次启动会先清空 `output_path`（默认 `./output/`）下已有文件。

## 目录说明

- `split_pdf_keyword.py`：单文件切分入口脚本
- `rename_pdfs_by_regex.py`：输出目录重命名入口脚本
- `process_usb_pdfs.py`：U 盘一条龙处理入口脚本
- `core/`：基础设施层，存放运行时、配置、日志
- `services/`：能力实现层，存放 OCR、切分、重命名、U 盘扫描、文件操作
- `workflows/`：流程编排层，串联切分、重命名、U 盘批处理流程
- `config.yaml`：规则型配置，如关键字、正则、OCR 参数映射
- `common.env`：环境隔离配置，如本地输入目录、GPU 开关
- `.vscode/settings.json`：VS Code 解释器与 Run Code 配置

## 快速开始

### 1) 安装依赖（必须使用项目解释器）

```powershell
.\.conda\python.exe -m pip install -U pip
.\.conda\python.exe -m pip install -r requirements.txt
.\.conda\python.exe -m pip install "onnxruntime-gpu[cuda,cudnn]==1.24.4"
```

### 2) 配置环境变量与关键词

先编辑 [`common.env`](common.env)：

```env
INPUT_DIR=./input/
OCR_USE_GPU=true
OCR_GPU_MEM=8000
OCR_USE_ANGLE_CLS=true
OCR_LANG=ch
OCR_DET_LIMIT_SIDE_LEN=2500
```

再编辑 [`config.yaml`](config.yaml) 中的规则部分：

```yaml
ocr:
  split_keywords:
    - "关键词1"
    - "关键词2"
```

### 3) 运行

```powershell
.\.conda\python.exe process_usb_pdfs.py
```

脚本默认会：

- 自动识别当前已插入的所有可移动 U 盘
- 递归扫描 U 盘中“当天修改”的 PDF，并先复制到本地 `input_dir`（默认 `./input/`）
- 复制到本地时会在文件名后追加修改时间，用于区分同名 PDF
- 所有切分结果直接输出到同一个 `output_path` 目录
- 先执行切分，再对切分结果按首页 OCR 正则重命名

如果你仍然需要手动处理单个文件，也可以直接运行：

```powershell
.\.conda\python.exe split_pdf_keyword.py --input-file .\input\example.pdf
```

如需按首页 OCR 正则重命名输出文件：

```powershell
.\.conda\python.exe rename_pdfs_by_regex.py
```

或直接在 VS Code 里点击 `Run Code`（已配置为 `.conda` 解释器）。

## 日志与输出

- 日志目录：`./log/`
- 日志文件：
  - `split_pdf_keyword.log`
  - `rename_pdfs_by_regex.log`
  - `process_usb_pdfs.log`
  - `ocr_engine.log`
  - `splitter.log`
- 输出目录：`output_path`（默认 `./output/`）
- 本地输入目录：`input_dir`（默认 `./input/`）

## VS Code 运行说明

本项目已在 `.vscode/settings.json` 固定：

- `python.defaultInterpreterPath = ${workspaceFolder}\.conda\python.exe`
- Code Runner Python 命令（PowerShell 兼容）：`& ".\.conda\python.exe" -u`

## 给 Copilot 的建议用法

当你在 VS Code 中让 Copilot 帮忙“配置环境 / CUDA / 运行项目”时，它会先读取 [`.github/copilot-instructions.md`](.github/copilot-instructions.md)，再按 [`CUDA_PYTHON_SETUP.md`](CUDA_PYTHON_SETUP.md) 执行。

你也可以直接提示：

```text
请先读取 CUDA_PYTHON_SETUP.md，然后按文档步骤在当前工作区完成 CUDA + Python 环境配置与验证。
```

## 常见问题

- `ImportError: cannot import name 'GraphOptimizationLevel'`：通常是误用 Anaconda `base`，请改用 `./.conda/python.exe`。
- PowerShell 报 `Unexpected token '-u'`：Code Runner 命令需使用 `&` 调用符，见 `.vscode/settings.json`。
- `CUDAExecutionProvider` 不可用：按 [`CUDA_PYTHON_SETUP.md`](CUDA_PYTHON_SETUP.md) 重新安装 `onnxruntime-gpu[cuda,cudnn]` 并执行启动前自检。
