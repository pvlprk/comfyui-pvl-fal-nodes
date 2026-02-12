from typing import List

import torch

from .fal_utils import ApiHandler, ImageUtils, ResultProcessor


class PVL_fal_Flux2Klein9BEdit_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "num_inference_steps": ("INT", {"default": 4, "min": 4, "max": 8}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "output_format": (["jpeg", "png", "webp"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "retries": ("INT", {"default": 2, "min": 0, "max": 10}),
                "timeout_sec": ("INT", {"default": 120, "min": 5, "max": 600, "step": 5}),
                "debug_log": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
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
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "use_mstudio_proxy": ("BOOLEAN", {"default": False}),
                "Proxy Only if >1K": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "PVL_tools_FAL"

    def _build_image_size(self, image_size, custom_width, custom_height):
        if image_size == "custom":
            if custom_width > 0 and custom_height > 0:
                return {"width": int(custom_width), "height": int(custom_height)}
            return None
        return image_size

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
                        urls.append(
                            ImageUtils.image_to_payload_uri(
                                frame,
                                use_mstudio_proxy=use_mstudio_proxy,
                                proxy_only_if_gt_1k=proxy_only_if_gt_1k,
                                timeout_sec=int(timeout_sec),
                            )
                        )
                        if debug:
                            print(
                                f"[Flux.2 Klein 9B Edit] encoded image_{idx + 1} frame {frame_idx + 1}"
                            )
                else:
                    urls.append(
                        ImageUtils.image_to_payload_uri(
                            tensor,
                            use_mstudio_proxy=use_mstudio_proxy,
                            proxy_only_if_gt_1k=proxy_only_if_gt_1k,
                            timeout_sec=int(timeout_sec),
                        )
                    )
                    if debug:
                        print(f"[Flux.2 Klein 9B Edit] encoded image_{idx + 1}")
            except Exception as e:
                print(f"[Flux.2 Klein 9B Edit] image_{idx + 1} encode error: {e}")

        if debug:
            print(f"[Flux.2 Klein 9B Edit] total encoded images: {len(urls)}")
        return urls[:4]

    def generate_image(
        self,
        prompt,
        num_inference_steps,
        num_images,
        enable_safety_checker,
        output_format,
        sync_mode,
        retries=2,
        timeout_sec=120,
        debug_log=False,
        image_size="custom",
        custom_width=0,
        custom_height=0,
        seed=-1,
        use_mstudio_proxy=False,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        **kwargs,
    ):
        width = int(custom_width) if image_size == "custom" and custom_width > 0 else 256
        height = int(custom_height) if image_size == "custom" and custom_height > 0 else 256

        def action(attempt, total_attempts):
            if debug_log:
                print(
                    f"[Flux.2 Klein 9B Edit] attempt {attempt}/{total_attempts} "
                    f"num_images={num_images} sync_mode={sync_mode}"
                )

            proxy_only_if_gt_1k = bool(
                kwargs.get("Proxy Only if >1K", kwargs.get("proxy_only_if_gt_1200px", False))
            )
            image_urls = self._collect_image_urls(
                [image_1, image_2, image_3, image_4],
                use_mstudio_proxy=use_mstudio_proxy,
                proxy_only_if_gt_1k=proxy_only_if_gt_1k,
                timeout_sec=timeout_sec,
                debug=debug_log,
            )
            if not image_urls:
                raise RuntimeError("No valid input images provided for Flux.2 Klein 9B Edit.")

            arguments = {
                "prompt": prompt,
                "image_urls": image_urls,
                "num_inference_steps": int(num_inference_steps),
                "num_images": int(num_images),
                "enable_safety_checker": bool(enable_safety_checker),
                "output_format": output_format,
                "sync_mode": bool(sync_mode),
            }

            size_payload = self._build_image_size(image_size, custom_width, custom_height)
            if size_payload is not None:
                arguments["image_size"] = size_payload

            if int(seed) != -1:
                arguments["seed"] = int(seed) & 0xFFFFFFFF

            if debug_log:
                print(
                    f"[Flux.2 Klein 9B Edit] payload keys={list(arguments.keys())} "
                    f"image_urls={len(image_urls)}"
                )

            model_id = "fal-ai/flux-2/klein/9b/edit"
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
                    f"[Flux.2 Klein 9B Edit ERROR] attempt {attempt}/{total_attempts} -> {e}"
                ),
            )
            return (img_tensor,)
        except Exception as e:
            print(f"Error generating image with Flux.2 Klein 9B Edit: {str(e)}")
            return ApiHandler.handle_image_generation_error("Flux.2 Klein 9B Edit", e, width=width, height=height)
