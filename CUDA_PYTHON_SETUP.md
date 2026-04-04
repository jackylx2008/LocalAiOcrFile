# SplitPdfByKeyWord：CUDA + Python 环境配置指南（VS Code）

## 目标

在 Windows + VS Code 环境下，确保本项目满足以下运行条件：

- 统一使用项目解释器：`./.conda/python.exe`
- OCR 使用 `onnxruntime-gpu` + CUDA Provider
- `Run Code` 与终端运行行为一致
- 启动前可通过项目自检日志确认 GPU 与 DLL 状态

---

## 1. 前置检查

在 VS Code 终端执行：

```powershell
nvidia-smi
```

预期：能看到 NVIDIA 显卡、驱动版本、CUDA Version。

---

## 2. 统一 Python 解释器（关键）

本项目固定解释器：

- `./.conda/python.exe`

已在工作区配置（`Run Code` 也会走该解释器）：

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}\\.conda\\python.exe",
  "python.terminal.activateEnvironment": true,
  "code-runner.runInTerminal": true,
  "code-runner.executorMap": {
    "python": "& \".\\.conda\\python.exe\" -u"
  }
}
```

> 如果你看到 `GraphOptimizationLevel` 导入错误，通常是误用了 `base` 环境而不是 `./.conda/python.exe`。

---

## 3. 安装依赖

在项目根目录执行：

```powershell
.\.conda\python.exe -m pip install -U pip
.\.conda\python.exe -m pip install -r requirements.txt
```

安装 ONNX GPU Runtime（含 CUDA/cuDNN 运行时依赖）：

```powershell
.\.conda\python.exe -m pip install "onnxruntime-gpu[cuda,cudnn]==1.24.4"
```

---

## 4. 验证 ONNX Runtime GPU 可用

```powershell
.\.conda\python.exe -c "import onnxruntime as ort; print('ORT', ort.__version__); print('Device', ort.get_device()); print('Providers', ort.get_available_providers())"
```

预期至少包含：

- `CUDAExecutionProvider`
- `CPUExecutionProvider`

---

## 5. 运行项目内置启动前自检

```powershell
.\.conda\python.exe -c "from core.config import load_config; from services.ocr_service import run_startup_self_check; cfg=load_config('config.yaml'); proc=run_startup_self_check(cfg); print(proc.get_provider_status())"
```

自检会在日志中输出：

- GPU 可用性
- CUDA DLL 搜索路径
- 关键 DLL 路径（如 `cublas64_12.dll`、`cudnn64_9.dll`）
- `det/rec/cls` 三路 provider 状态

---

## 6. 正式运行

```powershell
.\.conda\python.exe split_pdf_keyword.py
```

或在 VS Code 中点击 `Run Code`（已绑定到 `.conda` 解释器）。

---

## 7. 常见故障与处理

### 7.1 `ImportError: cannot import name 'GraphOptimizationLevel' from 'onnxruntime'`

原因：使用了错误解释器（通常是 Anaconda `base`）。

处理：

1. 确认 VS Code 解释器是 `./.conda/python.exe`
2. 用 `Run Code` 或命令 `./.conda/python.exe split_pdf_keyword.py`

### 7.2 PowerShell 报 `Unexpected token '-u'`

原因：可执行路径被引号包裹但未使用调用符 `&`。

处理：确保 Code Runner Python 执行命令是：

```powershell
& ".\\.conda\\python.exe" -u
```

### 7.3 `CUDAExecutionProvider is not available`

原因：CUDA/cuDNN 运行时缺失或 DLL 路径未加入搜索路径。

处理：

1. 重新安装：`onnxruntime-gpu[cuda,cudnn]==1.24.4`
2. 重新执行启动前自检，检查关键 DLL 路径是否正常

---

## 8. 给 VS Code Copilot 的推荐提示词

用于让 Copilot 自动按本项目标准配置环境：

```text
请先读取 CUDA_PYTHON_SETUP.md，然后按文档步骤在当前工作区完成 CUDA + Python 环境配置与验证。
要求：
1) 强制使用 ./.conda/python.exe；
2) 安装 requirements 与 onnxruntime-gpu[cuda,cudnn]==1.24.4；
3) 运行 ONNX provider 检查与 run_startup_self_check；
4) 报告最终 det/rec/cls provider 状态。
```
