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
- 若同一页命中 `not_split_keywords` 中任一关键词，即使同时命中全部 `split_keywords`，也**不会**作为切分起始页。
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
- `common.env`：环境隔离配置，如切分输入文件、切分输出目录、重命名输入目录、GPU 开关
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
INPUT_PATH=./input/
OUTPUT_PATH=./output/
SPLIT_INPUT_FILE=./input/example.pdf
SPLIT_OUTPUT_PATH=./output/
RENAME_INPUT_PATH=./output/
OCR_USE_GPU=true
OCR_GPU_MEM=8000
OCR_USE_ANGLE_CLS=true
OCR_LANG=ch
OCR_DET_LIMIT_SIDE_LEN=2500
```

再编辑 [`config.yaml`](config.yaml) 中的规则部分：

```yaml
split_input_file: ${SPLIT_INPUT_FILE:-}
split_output_path: ${SPLIT_OUTPUT_PATH:-./output/}
rename_input_path: ${RENAME_INPUT_PATH:-./output/}

ocr:
  split_keywords:
    - "关键词1"
    - "关键词2"
  not_split_keywords:
    - "排除切分页关键词"
```

脚本默认配置用途：
- `split_pdf_keyword.py`：使用 `split_input_file` 和 `split_output_path`
- `rename_pdfs_by_regex.py`：使用 `rename_input_path`
- `process_usb_pdfs.py`：仍使用 `input_path` 和 `output_path`

### 3) 运行

```powershell
.\.conda\python.exe process_usb_pdfs.py
```

脚本默认会：

- 自动识别当前已插入的所有可移动 U 盘
- 递归扫描 U 盘中“当天修改”的 PDF，并先复制到本地 `input_path`（默认 `./input/`）
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
.\.conda\python.exe rename_pdfs_by_regex.py --input-path .\output
```

其中默认命令会直接处理 `rename_input_path`（建议设为 `./output/`）中的 PDF，并原地重命名。

或直接在 VS Code 里点击 `Run Code`（已配置为 `.conda` 解释器）。

## PNG OCR 工作流

项目已增加一个面向 PNG 图片的 OCR 统计入口：

```powershell
.\.conda\python.exe png_regex_ocr.py
```

该流程会：

- 读取 `png_ocr_input_path` 下的 `*.png`
- 复用现有 OCR 引擎进行识别
- 按 `regex_pattern` 统计命中字符串
- 将结果写入 `png_ocr_output_txt_path`
- 每次启动前覆盖 `./log/png_regex_ocr.log`
- 每次启动前清空旧的统计 TXT

## llama.cpp OCR

项目现在支持通过 `llama.cpp` 直接做统一 OCR，并可在代码内自动拉起本地 `llama-server`。

关键配置位于 `config.yaml -> ocr`：

- `llamacpp_base_url`
- `llamacpp_model`
- `llamacpp_autostart`
- `llamacpp_server_path`
- `llamacpp_model_path`
- `llamacpp_mmproj_path`
- `llamacpp_n_gpu_layers`

当 `llamacpp_autostart=true` 时，OCR 初始化会：

- 优先复用已经运行的 `llama-server`
- 如果目标地址不可达，则自动启动 `llama-server`
- 自动校验当前模型是否具备 `multimodal` 能力
- 如果服务未加载 `mmproj`，会在初始化阶段直接报错，而不是等到 OCR 请求时再返回 500

项目默认按以下组合启动：

- 模型：`Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_S.gguf`
- 视觉投影：`mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf`

相关配置：

- `config.yaml`
  - `png_ocr_input_path`
  - `png_ocr_output_txt_path`
- `common.env`
  - `PNG_OCR_INPUT_PATH`
  - `PNG_OCR_OUTPUT_TXT_PATH`

### 当前已知问题

PNG OCR 工作流目前已经能处理一部分截图类图片，但在“微信截图 / 桌面附件列表截图”场景下仍有明显局限：

- 某些图片虽然可以正常读取，但 OCR 结果长度为 `0`，说明图片已解码成功，但模型没有检测到可用文字。
- 小字体、抗锯齿 UI 文本、图标干扰、浅色背景、文本截断，会显著降低默认 OCR 模型的识别效果。
- 当前 `regex_pattern` 是强结构匹配规则，对 `-`、`JZ`、`C2` 等字符识别错误非常敏感，只要 OCR 有轻微偏差就会完全匹配失败。
- OneDrive/桌面路径下的中文文件名问题已经通过 `numpy.fromfile + cv2.imdecode` 规避，但这只能解决“读图失败”，不能解决“识别为空”。
- 当前流程已经加入“原图 OCR + 增强 OCR”双通道策略，但依然是整图识别，对附件列表类截图仍可能不够稳定。

### 当前调试信息

`png_regex_ocr.log` 现在会记录以下调试信息，用于排查未命中原因：

- 图片路径、文件大小、读取字节数
- OCR 原始字符长度、去空白长度、有效行数
- 原图 OCR 与增强 OCR 的结果长度对比，以及最终选用哪一份结果
- 识别文本前 3 行
- 未命中时的候选文本长度
- 未命中时的完整 OCR 文本

### 下一步建议

如果后续要继续提升 PNG/截图识别准确率，优先建议：

- 增加“屏幕截图专用”局部裁剪识别，只针对文件名区域做 OCR
- 再根据截图场景单独设计更宽松的正则匹配策略
- 必要时引入更适合小号 UI 文本的 OCR 模型

## 日志与输出

- 日志目录：`./log/`
- 日志文件：
  - `split_pdf_keyword.log`
  - `rename_pdfs_by_regex.log`
  - `process_usb_pdfs.log`
  - `ocr_engine.log`
  - `splitter.log`
- 输出目录：`output_path`（默认 `./output/`）
- 本地输入目录：`input_path`（默认 `./input/`）

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
