import time
from typing import List

import torch

from .fal_utils import ImageUtils, ResultProcessor, ApiHandler


class PVL_fal_QwenImageEdit2511_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "output_format": (["jpeg", "png", "webp"], {"default": "png"}),
                "acceleration": (["none", "regular", "high"], {"default": "regular"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "retries": ("INT", {"default": 2, "min": 0, "max": 10}),
                "timeout_sec": ("INT", {"default": 120, "min": 5, "max": 600, "step": 5}),
                "debug_log": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # Up to 8 optional image inputs
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "image_size": (
                    [
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                        "custom",
                    ],
                    {"default": "custom"},
                ),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
                # ComfyUI can auto-randomize seeds beyond 32-bit; accept 64-bit inputs,
                # then clamp before sending to the FAL API.
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "use_mstudio_proxy": ("BOOLEAN", {"default": False}),
                "Proxy Only if >1K": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "PVL_tools_FAL"

    def _collect_image_urls(
        self,
        images: List[torch.Tensor],
        use_mstudio_proxy: bool = False,
        proxy_only_if_gt_1k: bool = False,
        timeout_sec: int = 120,
        debug: bool = False,
    ) -> List[str]:
        urls: List[str] = []
        for idx, img in enumerate(images):
            if img is None or not torch.is_tensor(img):
                continue
            tensor = img.detach()
            try:
                if tensor.ndim == 4:
                    for frame_idx in range(tensor.shape[0]):
                        frame = tensor[frame_idx]
                        use_proxy_for_frame = bool(use_mstudio_proxy)
                        if use_proxy_for_frame and proxy_only_if_gt_1k:
                            use_proxy_for_frame = ImageUtils.image_pixel_area(frame) > 1300000

                        if use_proxy_for_frame:
                            urls.append(
                                ImageUtils.upload_image_to_ministudio_proxy(
                                    frame, timeout=int(timeout_sec)
                                )
                            )
                        else:
                            urls.append(ImageUtils.image_to_data_uri(frame))
                        if debug:
                            print(
                                f"[Qwen Image Edit 2511] encoded image_{idx + 1} frame {frame_idx + 1} "
                                f"via {'proxy' if use_proxy_for_frame else 'base64'}"
                            )
                else:
                    use_proxy_for_image = bool(use_mstudio_proxy)
                    if use_proxy_for_image and proxy_only_if_gt_1k:
                        use_proxy_for_image = ImageUtils.image_pixel_area(tensor) > 1300000

                    if use_proxy_for_image:
                        urls.append(
                            ImageUtils.upload_image_to_ministudio_proxy(
                                tensor, timeout=int(timeout_sec)
                            )
                        )
                    else:
                        urls.append(ImageUtils.image_to_data_uri(tensor))
                    if debug:
                        print(
                            f"[Qwen Image Edit 2511] encoded image_{idx + 1} "
                            f"via {'proxy' if use_proxy_for_image else 'base64'}"
                        )
            except Exception as e:
                print(f"[Qwen Image Edit 2511] image_{idx + 1} encode error: {e}")
        if debug:
            print(f"[Qwen Image Edit 2511] total encoded images: {len(urls)}")
        return urls

    def _build_image_size(self, image_size, custom_width, custom_height):
        if image_size == "custom":
            if custom_width > 0 and custom_height > 0:
                return {"width": int(custom_width), "height": int(custom_height)}
            return None
        return image_size

    def generate_image(
        self,
        prompt,
        num_inference_steps,
        guidance_scale,
        num_images,
        enable_safety_checker,
        output_format,
        acceleration,
        sync_mode,
        retries=2,
        timeout_sec=120,
        debug_log=False,
        negative_prompt="",
        image_size="custom",
        custom_width=0,
        custom_height=0,
        seed=-1,
        use_mstudio_proxy=False,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        **kwargs,
    ):
        def action(attempt, total_attempts):
            if debug_log:
                print(
                    f"[Qwen Image Edit 2511] attempt {attempt}/{total_attempts} "
                    f"num_images={num_images} sync_mode={sync_mode}"
                )

            images = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8]
            proxy_only_if_gt_1k = bool(
                kwargs.get("Proxy Only if >1K", kwargs.get("proxy_only_if_gt_1200px", False))
            )
            image_urls = self._collect_image_urls(
                images,
                use_mstudio_proxy=use_mstudio_proxy,
                proxy_only_if_gt_1k=proxy_only_if_gt_1k,
                timeout_sec=timeout_sec,
                debug=debug_log,
            )

            arguments = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images": num_images,
                "enable_safety_checker": enable_safety_checker,
                "output_format": output_format,
                "acceleration": acceleration,
                "sync_mode": sync_mode,
            }

            size_payload = self._build_image_size(image_size, custom_width, custom_height)
            if size_payload is not None:
                arguments["image_size"] = size_payload

            if negative_prompt:
                arguments["negative_prompt"] = negative_prompt

            if seed != -1:
                arguments["seed"] = int(seed) & 0xFFFFFFFF

            if debug_log:
                print(f"[Qwen Image Edit 2511] image_urls={len(image_urls)} args={list(arguments.keys())}")

            # If no input images were provided, fall back to text-to-image.
            if not image_urls:
                if isinstance(prompt, str) and prompt.strip():
                    model_id = "fal-ai/qwen-image-2512"
                    if debug_log:
                        print(f"[Qwen Image Edit 2511] no images provided -> using {model_id}")
                else:
                    raise RuntimeError("No valid input images provided and prompt is empty.")
            else:
                model_id = "fal-ai/qwen-image-edit-2511"
                arguments["image_urls"] = image_urls

            if hasattr(ApiHandler, "submit_only") and hasattr(ApiHandler, "poll_and_get_result"):
                req_info = ApiHandler.submit_only(model_id, arguments, timeout=timeout_sec, debug=debug_log)
                result = ApiHandler.poll_and_get_result(req_info, timeout=timeout_sec, debug=debug_log)
            else:
                result = ApiHandler.submit_and_get_result(model_id, arguments)

            out = ResultProcessor.process_image_result(result)
            img_tensor = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(img_tensor) and img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)
            return img_tensor

        try:
            img_tensor = ApiHandler.run_with_retries(
                action,
                retries=retries,
                on_retry=lambda attempt, total_attempts, e: print(
                    f"[Qwen Image Edit 2511 ERROR] attempt {attempt}/{total_attempts} -> {e}"
                ),
            )
            return (img_tensor,)
        except Exception as e:
            print(f"Error generating image with Qwen Image Edit 2511: {str(e)}")
            return ApiHandler.handle_image_generation_error("Qwen Image Edit 2511", e)
