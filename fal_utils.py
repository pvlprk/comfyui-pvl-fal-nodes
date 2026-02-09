import base64
import io
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import requests
import torch
from PIL import Image


# -------------------------
# FalConfig: API key lookup
# -------------------------
@dataclass
class FalConfig:
    """Holds configuration and helpers for fal.ai REST API access."""
    api_key_env_vars: Tuple[str, ...] = ("FAL_KEY", "FAL_API_KEY", "FAL_CLIENT_KEY")

    @classmethod
    def get_api_key(cls) -> Optional[str]:
        for key in cls.api_key_env_vars:
            val = os.environ.get(key)
            if val:
                return val
        return None


# -------------------------
# Image utilities
# -------------------------
class ImageUtils:
    @staticmethod
    def get_proxy_options(kwargs: Optional[dict]) -> Tuple[bool, bool]:
        """
        Extract proxy toggles from node kwargs.
        Returns: (use_mstudio_proxy, proxy_only_if_gt_1k)
        """
        if not isinstance(kwargs, dict):
            return False, False
        use_mstudio_proxy = bool(kwargs.get("use_mstudio_proxy", False))
        proxy_only_if_gt_1k = bool(
            kwargs.get(
                "Proxy Only if >1K",
                kwargs.get("proxy_only_if_gt_1k", kwargs.get("proxy_only_if_gt_1200px", False)),
            )
        )
        return use_mstudio_proxy, proxy_only_if_gt_1k

    @staticmethod
    def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
        """
        Convert a torch Tensor into an RGB PIL.Image.
        Accepts:
          - 4D: (B,H,W,C) or (B,C,H,W) -> uses the first frame
          - 3D: (H,W,C) or (C,H,W)
          - 2D: (H,W) -> expands to (H,W,1)
        Dtypes: float (0..1) or uint8.
        """
        if not isinstance(image_tensor, torch.Tensor):
            raise RuntimeError("expected torch.Tensor")

        img = image_tensor.detach().cpu()

        # Handle 4D batches: pick first sample
        if img.ndim == 4:
            img = img[0]

        # Handle 2D grayscale
        if img.ndim == 2:
            img = img.unsqueeze(-1)  # (H,W,1)

        if img.ndim != 3:
            raise RuntimeError(f"image tensor must be 2D/3D/4D, got shape {tuple(img.shape)}.")

        # If CHW -> permute to HWC
        if img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
            img = img.permute(1, 2, 0)

        # Normalize dtype -> uint8
        if img.dtype in (torch.float16, torch.float32, torch.float64):
            img = (img.clamp(0, 1) * 255.0).round().to(torch.uint8)
        elif img.dtype != torch.uint8:
            img = img.to(torch.uint8)

        np_img = img.numpy()

        # Expand/drop channels to RGB
        if np_img.shape[2] == 1:
            np_img = np.repeat(np_img, 3, axis=2)
        elif np_img.shape[2] == 4:
            np_img = np_img[:, :, :3]

        return Image.fromarray(np_img, mode="RGB")

    @staticmethod
    def upload_image(image_tensor: torch.Tensor) -> str:
        """
        Save a tensor as PNG and upload to fal.storage.
        Returns a URL usable in API requests.
        """
        pil_image = ImageUtils.tensor_to_pil(image_tensor)
        if pil_image is None:
            raise RuntimeError("FAL: input image conversion failed.")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            pil_image.save(tmp_path, format="PNG")

            fal_key = FalConfig.get_api_key()
            if not fal_key:
                raise RuntimeError("FAL_KEY environment variable not set")

            with open(tmp_path, "rb") as f:
                resp = requests.post(
                    "https://fal.run/storage/upload",
                    headers={"Authorization": f"Key {fal_key}"},
                    files={"file": f},
                    timeout=120,
                )

            if resp.status_code != 200:
                raise RuntimeError(f"FAL upload failed: {resp.status_code} {resp.text}")

            data = resp.json()
            url = data.get("url") or data.get("file_url") or data.get("signed_url")
            if not url:
                raise RuntimeError("FAL: upload returned no URL")

            return url
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    @staticmethod
    def image_to_data_uri(image_tensor: torch.Tensor) -> str:
        """
        Convert a torch.Tensor image into a Base64 PNG data URI.
        Useful for FAL endpoints that accept inline images instead of URLs.
        """
        pil_image = ImageUtils.tensor_to_pil(image_tensor)
        if pil_image is None:
            raise RuntimeError("FAL: input image conversion failed.")

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    @staticmethod
    def image_max_side(image_tensor: torch.Tensor) -> int:
        """
        Return max(width, height) for an image tensor.
        Supports 2D/3D/4D tensors with either HWC or CHW layout.
        """
        if not isinstance(image_tensor, torch.Tensor):
            raise RuntimeError("expected torch.Tensor")

        img = image_tensor.detach().cpu()
        if img.ndim == 4:
            img = img[0]
        if img.ndim == 2:
            h, w = int(img.shape[0]), int(img.shape[1])
            return max(h, w)
        if img.ndim != 3:
            raise RuntimeError(f"image tensor must be 2D/3D/4D, got shape {tuple(img.shape)}.")

        if img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
            # CHW
            h, w = int(img.shape[1]), int(img.shape[2])
        else:
            # HWC
            h, w = int(img.shape[0]), int(img.shape[1])
        return max(h, w)

    @staticmethod
    def image_pixel_area(image_tensor: torch.Tensor) -> int:
        """
        Return width * height for an image tensor.
        Supports 2D/3D/4D tensors with either HWC or CHW layout.
        """
        if not isinstance(image_tensor, torch.Tensor):
            raise RuntimeError("expected torch.Tensor")

        img = image_tensor.detach().cpu()
        if img.ndim == 4:
            img = img[0]
        if img.ndim == 2:
            h, w = int(img.shape[0]), int(img.shape[1])
            return h * w
        if img.ndim != 3:
            raise RuntimeError(f"image tensor must be 2D/3D/4D, got shape {tuple(img.shape)}.")

        if img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
            # CHW
            h, w = int(img.shape[1]), int(img.shape[2])
        else:
            # HWC
            h, w = int(img.shape[0]), int(img.shape[1])
        return h * w

    @staticmethod
    def upload_image_to_ministudio_proxy(image_tensor: torch.Tensor, timeout: int = 120) -> str:
        """
        Upload an image tensor to MiniStudio proxy and return the public URL.
        Required env vars:
          - MINISTUDIO_PROXY_API_KEY
          - MINISTUDIO_PROXY_HOST (or PROXY_HOST fallback)
        """
        proxy_token = os.environ.get("MINISTUDIO_PROXY_API_KEY")
        if not proxy_token:
            raise RuntimeError("MINISTUDIO_PROXY_API_KEY environment variable not set")

        proxy_host = os.environ.get("MINISTUDIO_PROXY_HOST") or os.environ.get("PROXY_HOST")
        if not proxy_host:
            raise RuntimeError(
                "MINISTUDIO_PROXY_HOST (or PROXY_HOST) environment variable not set"
            )

        proxy_host = proxy_host.strip().rstrip("/")
        if not proxy_host.startswith(("http://", "https://")):
            proxy_host = f"https://{proxy_host}"

        endpoint = f"{proxy_host}/upload/images"
        pil_image = ImageUtils.tensor_to_pil(image_tensor)
        if pil_image is None:
            raise RuntimeError("Proxy upload: input image conversion failed.")

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")

        resp = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {proxy_token}",
                "Content-Type": "image/png",
            },
            data=buf.getvalue(),
            timeout=timeout,
        )

        if not resp.ok:
            raise RuntimeError(f"Proxy upload failed: {resp.status_code} {resp.text}")

        try:
            payload = resp.json()
        except Exception as e:
            raise RuntimeError(f"Proxy upload returned non-JSON response: {e}") from e

        public_url = payload.get("public_url")
        if not public_url:
            raise RuntimeError("Proxy upload succeeded but response contained no public_url")
        return public_url

    @staticmethod
    def image_to_payload_uri(
        image_tensor: torch.Tensor,
        use_mstudio_proxy: bool = False,
        proxy_only_if_gt_1k: bool = False,
        timeout_sec: int = 120,
    ) -> str:
        """
        Encode image for payload as either:
        - proxy public URL (if enabled), or
        - inline base64 data URI.
        """
        use_proxy = bool(use_mstudio_proxy)
        if use_proxy and proxy_only_if_gt_1k:
            use_proxy = ImageUtils.image_pixel_area(image_tensor) > 1300000
        if use_proxy:
            return ImageUtils.upload_image_to_ministudio_proxy(
                image_tensor, timeout=int(timeout_sec)
            )
        return ImageUtils.image_to_data_uri(image_tensor)

    @staticmethod
    def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
        """
        Convert a PIL.Image into a ComfyUI IMAGE tensor (1,H,W,C) float32 in [0..1].
        """
        if not isinstance(pil_image, Image.Image):
            raise RuntimeError("expected PIL.Image")
        rgb = pil_image.convert("RGB")
        arr = np.array(rgb).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    @staticmethod
    def mask_to_image(mask_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize a ComfyUI MASK tensor into an IMAGE-like tensor for encoding.
        Returns a 2D float tensor in [0..1] suitable for tensor_to_pil().
        """
        if not isinstance(mask_tensor, torch.Tensor):
            raise RuntimeError("expected torch.Tensor for mask")

        mask = mask_tensor.detach().cpu().float()
        if mask.ndim == 4:
            mask = mask[0]

        if mask.ndim == 3:
            # Common forms seen in ComfyUI graphs:
            # - (B,H,W) batch of masks
            # - (H,W,C) where C is 1/3/4
            # - (C,H,W) where C is 1/3/4
            #
            # For encoding we need a single 2D mask. Prefer interpreting (H,W,C) when it's
            # unambiguous, otherwise treat as (B,H,W)/(C,H,W) and take the first plane.
            if mask.shape[2] in (1, 3, 4) and mask.shape[0] > 4 and mask.shape[1] > 4:
                # HWC
                mask = mask[:, :, 0]
            else:
                # BHW or CHW
                mask = mask[0]

        if mask.ndim != 2:
            raise RuntimeError(f"mask tensor must be 2D/3D/4D, got shape {tuple(mask.shape)}.")

        return mask.clamp(0, 1)


# -------------------------
# Result processor
# -------------------------
class ResultProcessor:
    @staticmethod
    def _pil_from_data_uri(data_uri: str) -> Image.Image:
        header, b64 = data_uri.split(",", 1)
        if ";base64" not in header:
            raise RuntimeError("non-base64 data URI not supported")
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    @staticmethod
    def process_image_result(result: Dict) -> Tuple[torch.Tensor]:
        if "images" not in result or not isinstance(result["images"], list) or not result["images"]:
            raise RuntimeError("FAL: response contained no images.")

        images_np = []
        for img_info in result["images"]:
            img_url = img_info.get("url")
            content = img_info.get("content")

            if isinstance(img_url, str) and img_url.startswith("data:"):
                pil = ResultProcessor._pil_from_data_uri(img_url)
            elif isinstance(content, str) and content.startswith("data:"):
                pil = ResultProcessor._pil_from_data_uri(content)
            elif isinstance(img_url, str):
                resp = requests.get(img_url, timeout=120)
                resp.raise_for_status()
                pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
            else:
                raise RuntimeError("FAL: image entry missing URL/content")

            arr = np.array(pil).astype(np.float32) / 255.0
            images_np.append(arr)

        stacked = np.stack(images_np, axis=0)  # (B, H, W, C)
        tensor = torch.from_numpy(stacked)
        return (tensor,)

    @staticmethod
    def create_blank_image(width: int = 256, height: int = 256) -> Tuple[torch.Tensor]:
        blank = torch.zeros((1, height, width, 3), dtype=torch.float32)
        return (blank,)


# -------------------------
# API handler
# -------------------------
class ApiHandler:
    @staticmethod
    def _is_content_policy_violation(message_or_json) -> bool:
        try:
            if isinstance(message_or_json, dict):
                et = str(message_or_json.get("type", "")).lower()
                if "content_policy_violation" in et:
                    return True
                err = message_or_json.get("error")
                if isinstance(err, dict):
                    et2 = str(err.get("type", "")).lower()
                    if "content_policy_violation" in et2:
                        return True
                if "content_policy_violation" in str(message_or_json).lower():
                    return True
            elif isinstance(message_or_json, str):
                s = message_or_json.lower()
                if "content_policy_violation" in s:
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def run_with_retries(action, retries: int, is_fatal=None, on_retry=None, base_delay_sec: float = 0.0):
        """
        Run an action with retry logic.
        - action: callable(attempt, total_attempts) -> result
        - retries: number of retries (total attempts = retries + 1)
        - is_fatal: optional callable(error) -> bool to stop retrying
        - on_retry: optional callable(attempt, total_attempts, error)
        - base_delay_sec: optional linear backoff multiplier
        """
        total_attempts = int(retries) + 1
        last_err = None
        for attempt in range(1, total_attempts + 1):
            try:
                return action(attempt, total_attempts)
            except Exception as e:
                last_err = e
                if on_retry:
                    on_retry(attempt, total_attempts, e)
                if is_fatal and is_fatal(e):
                    break
                if attempt < total_attempts and base_delay_sec > 0:
                    time.sleep(base_delay_sec * attempt)
        if last_err:
            raise last_err
        raise RuntimeError("Retry failed without an exception.")

    @staticmethod
    def submit_only(model_id: str, arguments: Dict, timeout: int = 120, debug: bool = False) -> Dict:
        """
        Submit request to FAL queue API and return request info.
        """
        fal_key = FalConfig.get_api_key()
        if not fal_key:
            raise RuntimeError("FAL_KEY environment variable not set")

        base = "https://queue.fal.run"
        submit_url = f"{base}/{model_id}"
        headers = {"Authorization": f"Key {fal_key}"}

        if debug:
            print(f"[FAL SUBMIT] url={submit_url} payload_keys={list(arguments.keys())}")

        resp = requests.post(submit_url, headers=headers, json=arguments, timeout=timeout)
        if not resp.ok:
            try:
                js = resp.json()
                if ApiHandler._is_content_policy_violation(js):
                    raise RuntimeError(f"FAL content_policy_violation: {js}")
            except Exception:
                pass
            raise RuntimeError(f"FAL submit error {resp.status_code}: {resp.text}")

        sub = resp.json()
        req_id = sub.get("request_id")
        if not req_id:
            raise RuntimeError("FAL did not return a request_id")

        status_url = sub.get("status_url") or f"{base}/{model_id}/requests/{req_id}/status"
        resp_url = sub.get("response_url") or f"{base}/{model_id}/requests/{req_id}"

        if debug:
            print(f"[FAL SUBMIT OK] request_id={req_id}")

        return {
            "request_id": req_id,
            "status_url": status_url,
            "response_url": resp_url,
        }

    @staticmethod
    def poll_and_get_result(request_info: Dict, timeout: int = 120, debug: bool = False) -> Dict:
        """
        Poll the FAL queue API for completion and return the JSON response.
        """
        fal_key = FalConfig.get_api_key()
        if not fal_key:
            raise RuntimeError("FAL_KEY environment variable not set")

        headers = {"Authorization": f"Key {fal_key}"}
        status_url = request_info["status_url"]
        resp_url = request_info["response_url"]
        req_id = request_info.get("request_id", "")[:16]

        deadline = time.time() + timeout
        completed = False
        while time.time() < deadline:
            try:
                sr = requests.get(status_url, headers=headers, timeout=min(10, timeout))
                if sr.ok:
                    js = sr.json()
                    st = js.get("status")
                    if debug:
                        print(f"[FAL POLL] req={req_id} status={st}")
                    if st == "COMPLETED":
                        completed = True
                        break
                    if st in ("ERROR", "FAILED"):
                        msg = js.get("error") or "Unknown FAL error"
                        payload = js.get("payload")
                        if payload:
                            raise RuntimeError(f"FAL status ERROR: {msg} | details: {payload}")
                        raise RuntimeError(f"FAL status ERROR: {msg}")
            except Exception as e:
                if debug:
                    print(f"[FAL POLL] req={req_id} status_check_error: {e}")
            time.sleep(0.6)

        if not completed:
            raise RuntimeError(f"FAL request {req_id} timed out after {timeout}s")

        rr = requests.get(resp_url, headers=headers, timeout=min(15, timeout))
        if not rr.ok:
            try:
                js = rr.json()
                if ApiHandler._is_content_policy_violation(js):
                    raise RuntimeError(f"FAL content_policy_violation: {js}")
            except Exception:
                pass
            raise RuntimeError(f"FAL result fetch error {rr.status_code}: {rr.text}")

        return rr.json().get("response", rr.json())

    @staticmethod
    def submit_and_get_result(model_id: str, arguments: Dict) -> Dict:
        """
        Submit request to fal.run REST API and return JSON result.
        """
        fal_key = FalConfig.get_api_key()
        if not fal_key:
            raise RuntimeError("FAL_KEY environment variable not set")

        url = f"https://fal.run/{model_id}"

        resp = requests.post(
            url,
            headers={
                "Authorization": f"Key {fal_key}",
                "Content-Type": "application/json"
            },
            json=arguments,
            timeout=300,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"FAL API error {resp.status_code}: {resp.text}")

        return resp.json()

    @staticmethod
    def handle_image_generation_error(api_name: str, error: Exception, width: int = 256, height: int = 256):
        print(f"[{api_name} ERROR] {str(error)}")
        return ResultProcessor.create_blank_image(width=width, height=height)
