"""
文件说明：
- 提供项目统一的 OCR 抽象，支持 RapidOCR 与 llama.cpp 本地视觉模型两种识别引擎。

主要职责：
- 初始化并选择 OCR 引擎。
- 对 PDF 页面图像与普通图片执行文本识别。
- 为切分、重命名、PNG OCR 工作流提供统一接口。

运行方式：
- 分类：被依赖脚本
- 直接运行命令：不建议直接运行
- 直接运行用途：无独立业务入口，主要作为 OCR 能力模块被其他脚本导入。
- 被谁调用：workflows.split_workflow、services.pdf_rename_service、services.png_regex_ocr_service
- 作为依赖用途：为各业务流程提供统一 OCR 识别与启动自检能力。
"""

import base64
import json
import os
import site
import subprocess
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from urllib.parse import urlparse
import atexit

import cv2
import fitz  # PyMuPDF
import numpy as np
from core.logging_utils import setup_logger


KEY_CUDA_DLLS = [
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudart64_12.dll",
    "cudnn64_9.dll",
    "cufft64_11.dll",
]

DEFAULT_LLAMACPP_PROMPT = (
    "Perform OCR on this image and return only the recognized text. "
    "Preserve line breaks as much as possible. "
    "Do not summarize, explain, translate, or add markdown. "
    "If no readable text is visible, return an empty string."
)


def _inject_cuda_runtime_paths():
    if not hasattr(os, "add_dll_directory"):
        return []

    cuda_bin_paths = []
    for site_package in site.getsitepackages():
        nvidia_dir = os.path.join(site_package, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue

        for package_name in os.listdir(nvidia_dir):
            bin_dir = os.path.join(nvidia_dir, package_name, "bin")
            if os.path.isdir(bin_dir):
                cuda_bin_paths.append(bin_dir)

    cuda_bin_paths = list(dict.fromkeys(cuda_bin_paths))

    if not cuda_bin_paths:
        return []

    for bin_dir in cuda_bin_paths:
        try:
            os.add_dll_directory(bin_dir)
        except OSError:
            continue

    os.environ["PATH"] = ";".join(cuda_bin_paths) + ";" + os.environ.get("PATH", "")
    return cuda_bin_paths


def _collect_key_cuda_dll_paths():
    dll_paths = {}
    for dll_name in KEY_CUDA_DLLS:
        dll_paths[dll_name] = None
        for bin_dir in CUDA_RUNTIME_BIN_DIRS:
            candidate = os.path.join(bin_dir, dll_name)
            if os.path.exists(candidate):
                dll_paths[dll_name] = candidate
                break
    return dll_paths


CUDA_RUNTIME_BIN_DIRS = _inject_cuda_runtime_paths()

from rapidocr_onnxruntime import RapidOCR  # noqa: E402
from rapidocr_onnxruntime.utils import OrtInferSession  # noqa: E402

logger = setup_logger(log_file="./log/ocr_engine.log")
_LLAMACPP_MANAGED_PROCESS = None
_LLAMACPP_MANAGED_BASE_URL = None


def _ensure_parent_dir(file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def _read_log_tail(log_path, max_lines=40):
    path = Path(log_path)
    if not path.exists():
        return f"[日志文件不存在] {path}"

    try:
        with path.open("r", encoding="utf-8", errors="replace") as file:
            return "".join(deque(file, maxlen=max_lines)).strip() or f"[日志为空] {path}"
    except OSError as exc:
        return f"[读取日志失败] {path}, error={exc}"


def _terminate_managed_llamacpp_server():
    global _LLAMACPP_MANAGED_PROCESS
    if _LLAMACPP_MANAGED_PROCESS is None:
        return

    if _LLAMACPP_MANAGED_PROCESS.poll() is not None:
        logger.info("项目托管的 llama.cpp 服务已退出，无需重复关闭。")
        _LLAMACPP_MANAGED_PROCESS = None
        return

    try:
        logger.info("正在关闭项目托管的 llama.cpp 服务...")
        _LLAMACPP_MANAGED_PROCESS.terminate()
        _LLAMACPP_MANAGED_PROCESS.wait(timeout=10)
        logger.info("项目托管的 llama.cpp 服务已正常关闭。")
    except Exception:
        try:
            logger.warning("llama.cpp 服务未在超时时间内退出，尝试强制结束进程。")
            _LLAMACPP_MANAGED_PROCESS.kill()
            logger.info("项目托管的 llama.cpp 服务已被强制结束。")
        except Exception:
            logger.exception("关闭项目托管的 llama.cpp 服务失败。")
    finally:
        _LLAMACPP_MANAGED_PROCESS = None


atexit.register(_terminate_managed_llamacpp_server)


class BaseOCRProcessor:
    def __init__(self, config, app_logger):
        self.config = config
        self.logger = app_logger or logger

    def get_provider_status(self):
        raise NotImplementedError

    def ocr(self, image):
        raise NotImplementedError

    def pdf_to_images(self, pdf_path):
        doc = fitz.open(pdf_path)
        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.h, pix.w, pix.n
            )
            if pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            yield page_index, img
        doc.close()

    def process_pdf(self, pdf_path):
        results = []
        self.logger.info(f"开始处理 PDF: {pdf_path}")

        for page_index, img in self.pdf_to_images(pdf_path):
            self.logger.debug(f"正在识别第 {page_index + 1} 页...")
            result, _ = self.ocr(img)

            page_text = ""
            if result:
                for line in result:
                    page_text += line[1] + "\n"

            text_lines = [
                line.strip() for line in page_text.splitlines() if line.strip()
            ]
            preview_lines = text_lines[:3]
            if preview_lines:
                self.logger.info(
                    f"第 {page_index + 1} 页识别文本前 3 行: {' | '.join(preview_lines)}"
                )
            else:
                self.logger.info(f"第 {page_index + 1} 页识别文本前 3 行: [无识别文本]")

            results.append({"page": page_index, "text": page_text})

        return results


class RapidOCRProcessor(BaseOCRProcessor):
    def __init__(self, config, app_logger):
        super().__init__(config, app_logger)
        ocr_config = config.get("ocr", {})
        use_gpu = _to_bool(ocr_config.get("use_gpu", True))
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        det_model_path = ocr_config.get("det_model_path", "")
        rec_model_path = ocr_config.get("rec_model_path", "")
        cls_model_path = ocr_config.get("cls_model_path", "")

        try:
            self.ocr_engine = RapidOCR(
                det_use_cuda=use_gpu,
                rec_use_cuda=use_gpu,
                cls_use_cuda=use_gpu,
                det_model_path=det_model_path,
                rec_model_path=rec_model_path,
                cls_model_path=cls_model_path,
                det_providers=providers,
                rec_providers=providers,
                cls_providers=providers,
            )

            if use_gpu:
                self._rebuild_all_ort_sessions_with_cuda()

            self._log_provider_status(use_gpu)
        except Exception as exc:
            self.logger.warning(f"RapidOCR GPU 初始化失败，将回退 CPU 模式: {exc}")
            cpu_providers = ["CPUExecutionProvider"]
            self.ocr_engine = RapidOCR(
                det_use_cuda=False,
                rec_use_cuda=False,
                cls_use_cuda=False,
                det_model_path=det_model_path,
                rec_model_path=rec_model_path,
                cls_model_path=cls_model_path,
                det_providers=cpu_providers,
                rec_providers=cpu_providers,
                cls_providers=cpu_providers,
            )
            self._log_provider_status(False)

    def get_provider_status(self):
        provider_status = {}

        if getattr(self.ocr_engine, "use_text_det", False):
            provider_status["det"] = (
                self.ocr_engine.text_detector.infer.session.get_providers()
            )

        provider_status["rec"] = (
            self.ocr_engine.text_recognizer.session.session.get_providers()
        )

        if getattr(self.ocr_engine, "use_angle_cls", False):
            provider_status["cls"] = (
                self.ocr_engine.text_cls.infer.session.get_providers()
            )

        return provider_status

    def _rebuild_all_ort_sessions_with_cuda(self):
        if getattr(self.ocr_engine, "use_text_det", False):
            det_model = self.ocr_engine.text_detector.infer.session._model_path
            self.ocr_engine.text_detector.infer = OrtInferSession(
                {"model_path": det_model, "use_cuda": True}
            )

        rec_model = self.ocr_engine.text_recognizer.session.session._model_path
        self.ocr_engine.text_recognizer.session = OrtInferSession(
            {"model_path": rec_model, "use_cuda": True}
        )

        if getattr(self.ocr_engine, "use_angle_cls", False):
            cls_model = self.ocr_engine.text_cls.infer.session._model_path
            self.ocr_engine.text_cls.infer = OrtInferSession(
                {"model_path": cls_model, "use_cuda": True}
            )

    def _log_provider_status(self, target_gpu):
        provider_status = self.get_provider_status()

        if target_gpu:
            all_gpu = all(
                "CUDAExecutionProvider" in providers
                for providers in provider_status.values()
            )
            if all_gpu:
                self.logger.info(f"OCR 已启用 GPU 推理，providers: {provider_status}")
            else:
                self.logger.warning(
                    f"OCR 未完全启用 GPU，部分模型回退 CPU，providers: {provider_status}"
                )
        else:
            self.logger.info(f"OCR 使用 CPU 推理，providers: {provider_status}")

    def ocr(self, image):
        return self.ocr_engine(image)


class LlamaCppOCRProcessor(BaseOCRProcessor):
    def __init__(self, config, app_logger):
        super().__init__(config, app_logger)
        ocr_config = config.get("ocr", {})
        self.base_url = str(
            ocr_config.get(
                "llamacpp_base_url",
                ocr_config.get("lmstudio_base_url", "http://127.0.0.1:8080/v1"),
            )
        ).rstrip("/")
        self.root_url, self.api_url = self._normalize_base_urls(self.base_url)
        self.chat_url = f"{self.api_url}/chat/completions"
        self.models_url = f"{self.api_url}/models"
        self.health_url = f"{self.root_url}/health"
        self.timeout_sec = int(
            ocr_config.get(
                "llamacpp_timeout_sec", ocr_config.get("lmstudio_timeout_sec", 120)
            )
        )
        self.max_tokens = int(
            ocr_config.get(
                "llamacpp_max_tokens", ocr_config.get("lmstudio_max_tokens", 4096)
            )
        )
        self.temperature = float(
            ocr_config.get(
                "llamacpp_temperature", ocr_config.get("lmstudio_temperature", 0)
            )
        )
        self.image_max_side = int(
            ocr_config.get(
                "llamacpp_image_max_side",
                ocr_config.get("lmstudio_image_max_side", 1800),
            )
        )
        self.prompt = str(
            ocr_config.get(
                "llamacpp_ocr_prompt",
                ocr_config.get("lmstudio_ocr_prompt", DEFAULT_LLAMACPP_PROMPT),
            )
        )
        self.api_key = str(
            ocr_config.get("llamacpp_api_key", ocr_config.get("lmstudio_api_key", ""))
        )
        self.model_name = str(
            ocr_config.get(
                "llamacpp_model",
                ocr_config.get("lmstudio_model", "Qwen2.5-VL-7B-Instruct"),
            )
        ).strip()
        self.auto_start = _to_bool(ocr_config.get("llamacpp_autostart", False))
        self.server_path = str(ocr_config.get("llamacpp_server_path", "")).strip()
        self.model_path = str(ocr_config.get("llamacpp_model_path", "")).strip()
        self.mmproj_path = str(ocr_config.get("llamacpp_mmproj_path", "")).strip()
        self.n_gpu_layers = int(ocr_config.get("llamacpp_n_gpu_layers", 999))
        self.startup_timeout_sec = int(
            ocr_config.get("llamacpp_startup_timeout_sec", 180)
        )
        self.startup_poll_interval_sec = float(
            ocr_config.get("llamacpp_startup_poll_interval_sec", 1)
        )
        self.stdout_log_path = str(
            ocr_config.get("llamacpp_stdout_log_path", "./log/llama_server.out.log")
        ).strip()
        self.stderr_log_path = str(
            ocr_config.get("llamacpp_stderr_log_path", "./log/llama_server.err.log")
        ).strip()
        self._verify_connection()

    def get_provider_status(self):
        return {
            "engine": "llamacpp",
            "base_url": self.api_url,
            "model": self.model_name,
            "image_max_side": self.image_max_side,
            "auto_start": self.auto_start,
        }

    def _normalize_base_urls(self, configured_base_url):
        parsed = urlparse(configured_base_url)
        if not parsed.scheme or not parsed.netloc:
            raise RuntimeError(
                f"llama.cpp base_url 配置不合法: {configured_base_url}"
            )

        root_url = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        path = parsed.path.rstrip("/")
        api_url = f"{root_url}{path}" if path else root_url
        return root_url, api_url

    def _endpoint_healthcheck(self):
        health_payload = self._http_json(self.health_url, method="GET")
        models_payload = self._http_json(self.models_url, method="GET")
        return health_payload, models_payload

    def _resolve_mmproj_path(self):
        if self.mmproj_path:
            mmproj = Path(self.mmproj_path)
            if not mmproj.exists():
                raise RuntimeError(f"llama.cpp mmproj 文件不存在: {mmproj}")
            return str(mmproj)

        if not self.model_path:
            raise RuntimeError(
                "llama.cpp 自动启动缺少 llamacpp_model_path，无法自动发现 mmproj。"
            )

        model_dir = Path(self.model_path).resolve().parent
        candidates = sorted(model_dir.glob("mmproj*.gguf"))
        if not candidates:
            raise RuntimeError(
                f"未在模型目录找到 mmproj 文件: {model_dir}"
            )

        self.mmproj_path = str(candidates[0])
        self.logger.info(f"自动发现 mmproj 文件: {self.mmproj_path}")
        return self.mmproj_path

    def _build_server_command(self):
        if not self.server_path:
            raise RuntimeError(
                "llama.cpp 自动启动缺少 llamacpp_server_path 配置。"
            )
        if not self.model_path:
            raise RuntimeError(
                "llama.cpp 自动启动缺少 llamacpp_model_path 配置。"
            )

        server_path = Path(self.server_path)
        model_path = Path(self.model_path)
        if not server_path.exists():
            raise RuntimeError(f"llama.cpp 可执行文件不存在: {server_path}")
        if not model_path.exists():
            raise RuntimeError(f"llama.cpp 模型文件不存在: {model_path}")

        mmproj_path = self._resolve_mmproj_path()
        parsed = urlparse(self.api_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8080

        return [
            str(server_path),
            "-m",
            str(model_path),
            "--mmproj",
            str(mmproj_path),
            "-ngl",
            str(self.n_gpu_layers),
            "--host",
            host,
            "--port",
            str(port),
            "--verbose",
        ]

    def _format_startup_log_tail(self):
        stderr_tail = _read_log_tail(self.stderr_log_path)
        stdout_tail = _read_log_tail(self.stdout_log_path)
        return (
            f"stderr_tail=\n{stderr_tail}\n"
            f"stdout_tail=\n{stdout_tail}"
        )

    def _build_startup_failure_message(self, process):
        return (
            "llama.cpp 进程启动后提前退出。"
            f" returncode={process.returncode}; "
            f"stderr_log={self.stderr_log_path}; stdout_log={self.stdout_log_path}; "
            f"日志摘要: {self._format_startup_log_tail()}"
        )

    def _start_managed_server(self):
        global _LLAMACPP_MANAGED_PROCESS, _LLAMACPP_MANAGED_BASE_URL

        if _LLAMACPP_MANAGED_PROCESS is not None:
            if _LLAMACPP_MANAGED_PROCESS.poll() is None:
                if _LLAMACPP_MANAGED_BASE_URL == self.api_url:
                    self.logger.info(
                        f"复用已由项目启动的 llama.cpp 服务: {self.api_url}"
                    )
                    return
            else:
                _LLAMACPP_MANAGED_PROCESS = None
                _LLAMACPP_MANAGED_BASE_URL = None

        command = self._build_server_command()
        _ensure_parent_dir(self.stdout_log_path)
        _ensure_parent_dir(self.stderr_log_path)
        stdout_file = open(self.stdout_log_path, "a", encoding="utf-8")
        stderr_file = open(self.stderr_log_path, "a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=str(Path(command[0]).resolve().parent),
            )
        except Exception:
            stdout_file.close()
            stderr_file.close()
            raise

        process._codex_stdout_file = stdout_file
        process._codex_stderr_file = stderr_file

        _LLAMACPP_MANAGED_PROCESS = process
        _LLAMACPP_MANAGED_BASE_URL = self.api_url
        self.logger.info(
            "已启动项目托管的 llama.cpp 服务: "
            + " ".join(command)
        )

        deadline = time.time() + self.startup_timeout_sec
        last_error = None
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(self._build_startup_failure_message(process))

            try:
                self._endpoint_healthcheck()
                self.logger.info("llama.cpp 自动启动成功，healthcheck 已通过。")
                return
            except KeyboardInterrupt:
                self.logger.warning("启动等待期间收到 Ctrl+C，准备中断 llama.cpp 自启动流程。")
                raise
            except Exception as exc:
                last_error = exc
                time.sleep(self.startup_poll_interval_sec)

        raise RuntimeError(
            "等待 llama.cpp 自动启动超时，最后一次错误: "
            f"{last_error}. 启动日志摘要: {self._format_startup_log_tail()}"
        )

    def _extract_model_capabilities(self, models_payload):
        capability_map = {}
        for item in models_payload.get("models", []):
            model_id = item.get("model") or item.get("name") or item.get("id")
            if model_id:
                capability_map[model_id] = item.get("capabilities", []) or []
        for item in models_payload.get("data", []):
            model_id = item.get("id") or item.get("model") or item.get("name")
            if model_id and model_id not in capability_map:
                capability_map[model_id] = item.get("capabilities", []) or []
        return capability_map

    def _ensure_multimodal_capability(self, models_payload):
        capability_map = self._extract_model_capabilities(models_payload)
        current_capabilities = capability_map.get(self.model_name, [])
        if "multimodal" in current_capabilities:
            return

        if not current_capabilities:
            self.logger.warning(
                f"模型 {self.model_name} 未返回 capabilities 字段，将继续尝试 OCR。"
            )
            return

        raise RuntimeError(
            f"当前 llama.cpp 服务中的模型 {self.model_name} 不支持 multimodal，"
            f"当前 capabilities={current_capabilities}。"
            "这通常表示服务启动时没有加载 mmproj。"
        )

    def _verify_connection(self):
        try:
            health_payload, models_payload = self._endpoint_healthcheck()
            if health_payload:
                self.logger.info(f"llama.cpp health={health_payload}")
        except Exception as exc:
            if not self.auto_start:
                raise RuntimeError(
                    f"llama.cpp 连接失败，且未启用自动启动: {exc}"
                ) from exc
            self.logger.warning(
                f"llama.cpp 当前不可用，尝试自动启动本地服务: {exc}"
            )
            self._start_managed_server()
            health_payload, models_payload = self._endpoint_healthcheck()
            if health_payload:
                self.logger.info(f"llama.cpp health={health_payload}")

        models = self._get_models(models_payload)
        if not models:
            raise RuntimeError(
                f"llama.cpp 未返回可用模型，请检查 llama-server 是否已启动: {self.models_url}"
            )

        if not self.model_name:
            self.model_name = models[0]
            self.logger.info(
                f"llama.cpp 未显式指定模型，自动使用第一个可用模型: {self.model_name}"
            )
        elif self.model_name not in models:
            self.logger.warning(
                f"llama.cpp 配置模型 {self.model_name} 不在当前服务返回列表中，"
                f"当前可用模型: {models}"
            )

        self._ensure_multimodal_capability(models_payload)
        self.logger.info(
            f"llama.cpp OCR 已启用: base_url={self.api_url}, model={self.model_name}"
        )

    def _get_models(self, payload=None):
        if payload is None:
            payload = self._http_json(self.models_url, method="GET")
        model_items = payload.get("data", [])
        models = [item.get("id", "") for item in model_items if item.get("id")]
        if models:
            return models

        fallback_items = payload.get("models", [])
        return [
            item.get("model", "") or item.get("name", "")
            for item in fallback_items
            if item.get("model") or item.get("name")
        ]

    def _http_json(self, url, payload=None, method="POST"):
        body = None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                return json.loads(response.read().decode("utf-8"))
        except KeyboardInterrupt:
            self.logger.warning(
                f"收到 Ctrl+C，中断 llama.cpp 请求: method={method}, url={url}"
            )
            raise
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"llama.cpp HTTP 请求失败: status={exc.code}, body={error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"llama.cpp 连接失败: {exc.reason}") from exc

    def _prepare_image(self, image):
        if image is None:
            return None

        height, width = image.shape[:2]
        max_side = max(height, width)
        if max_side <= self.image_max_side:
            return image

        scale = self.image_max_side / float(max_side)
        resized = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        return resized

    def _encode_image_to_data_url(self, image):
        prepared = self._prepare_image(image)
        success, encoded = cv2.imencode(".png", prepared)
        if not success:
            raise RuntimeError("llama.cpp OCR 图片编码失败。")
        b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _extract_response_text(self, response_json):
        choices = response_json.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            fragments = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    fragments.append(str(item.get("text", "")))
            return "\n".join(fragment for fragment in fragments if fragment).strip()

        return str(content).strip()

    def ocr(self, image):
        start = time.perf_counter()
        image_url = self._encode_image_to_data_url(image)
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        }
        response_json = self._http_json(self.chat_url, payload=payload, method="POST")
        text = self._extract_response_text(response_json)
        elapsed = time.perf_counter() - start
        if not text:
            return [], elapsed

        result = []
        for line in text.splitlines():
            normalized = line.strip()
            if normalized:
                result.append([None, normalized, 1.0])
        return result, elapsed


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def run_startup_self_check(config, app_logger=None):
    check_logger = app_logger or logger
    ocr_config = config.get("ocr", {})
    engine = str(ocr_config.get("engine", "rapidocr")).strip().lower()
    use_gpu = _to_bool(ocr_config.get("use_gpu", True))

    check_logger.info("启动前自检开始...")
    check_logger.info(f"OCR 引擎={engine}, use_gpu={use_gpu}")

    if engine == "llamacpp":
        processor = LlamaCppOCRProcessor(config, check_logger)
        check_logger.info(f"Provider 状态: {processor.get_provider_status()}")
        check_logger.info("启动前自检结束。")
        return processor

    try:
        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        device = ort.get_device()
        gpu_available = (
            device == "GPU" and "CUDAExecutionProvider" in available_providers
        )
        check_logger.info(
            f"GPU 可用性: {gpu_available} (device={device}, available_providers={available_providers})"
        )
    except Exception as exc:
        check_logger.warning(f"读取 ONNX Runtime 状态失败: {exc}")

    if CUDA_RUNTIME_BIN_DIRS:
        check_logger.info(f"CUDA DLL 搜索路径: {CUDA_RUNTIME_BIN_DIRS}")
    else:
        check_logger.warning("未检测到 nvidia/*/bin DLL 路径。")

    dll_paths = _collect_key_cuda_dll_paths()
    check_logger.info(f"关键 DLL 路径: {dll_paths}")

    processor = RapidOCRProcessor(config, check_logger)
    check_logger.info(f"Provider 状态: {processor.get_provider_status()}")
    check_logger.info("启动前自检结束。")
    return processor
