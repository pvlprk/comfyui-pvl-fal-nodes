import torch

from .fal_utils import ImageUtils, ResultProcessor, ApiHandler


class PVL_fal_QwenImageEdit2511MultipleAngles_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "horizontal_angle": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "vertical_angle": ("FLOAT", {"default": 0.0, "min": -30.0, "max": 90.0, "step": 1.0}),
                "zoom": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "acceleration": (["none", "regular"], {"default": "regular"}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "retries": ("INT", {"default": 2, "min": 0, "max": 10}),
                "timeout_sec": ("INT", {"default": 120, "min": 5, "max": 600, "step": 5}),
                "debug_log": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "additional_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "image_size": (
                    [
                        "auto",
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                        "custom",
                    ],
                    {"default": "auto"},
                ),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
                # ComfyUI can auto-randomize seeds beyond 32-bit; accept 64-bit inputs,
                # then clamp before sending to the FAL API.
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "PVL_tools_FAL"

    def _collect_image_urls(self, image: torch.Tensor):
        if not torch.is_tensor(image):
            return []
        img = image.detach()
        if img.ndim == 4:
            return [ImageUtils.image_to_data_uri(img[i]) for i in range(img.shape[0])]
        return [ImageUtils.image_to_data_uri(img)]

    def _build_image_size(self, image_size, custom_width, custom_height):
        if image_size == "auto":
            # Omit image_size so API uses the input resolution
            return None
        if image_size == "custom":
            if custom_width > 0 and custom_height > 0:
                return {"width": int(custom_width), "height": int(custom_height)}
            return None
        return image_size

    def generate_image(
        self,
        image,
        horizontal_angle,
        vertical_angle,
        zoom,
        lora_scale,
        guidance_scale,
        num_inference_steps,
        acceleration,
        enable_safety_checker,
        output_format,
        num_images,
        sync_mode,
        retries=2,
        timeout_sec=120,
        debug_log=False,
        additional_prompt="",
        negative_prompt="",
        image_size="auto",
        custom_width=0,
        custom_height=0,
        seed=-1,
    ):
        def action(attempt, total_attempts):
            if debug_log:
                print(
                    f"[Qwen Image Edit 2511 Multiple Angles] attempt {attempt}/{total_attempts} "
                    f"num_images={num_images} sync_mode={sync_mode}"
                )

            image_urls = self._collect_image_urls(image)
            if not image_urls:
                raise RuntimeError("No valid input image provided.")

            arguments = {
                "image_urls": image_urls,
                "horizontal_angle": float(horizontal_angle),
                "vertical_angle": float(vertical_angle),
                "zoom": float(zoom),
                "lora_scale": float(lora_scale),
                "guidance_scale": float(guidance_scale),
                "num_inference_steps": int(num_inference_steps),
                "acceleration": acceleration,
                "enable_safety_checker": bool(enable_safety_checker),
                "output_format": output_format,
                "num_images": int(num_images),
                "sync_mode": bool(sync_mode),
            }

            size_payload = self._build_image_size(image_size, custom_width, custom_height)
            if size_payload is not None:
                arguments["image_size"] = size_payload

            if additional_prompt:
                arguments["additional_prompt"] = additional_prompt
            if negative_prompt:
                arguments["negative_prompt"] = negative_prompt
            if seed != -1:
                arguments["seed"] = int(seed) & 0xFFFFFFFF

            if debug_log:
                print(
                    "[Qwen Image Edit 2511 Multiple Angles] "
                    f"image_urls={len(image_urls)} args={list(arguments.keys())}"
                )

            if hasattr(ApiHandler, "submit_only") and hasattr(ApiHandler, "poll_and_get_result"):
                req_info = ApiHandler.submit_only(
                    "fal-ai/qwen-image-edit-2511-multiple-angles",
                    arguments,
                    timeout=timeout_sec,
                    debug=debug_log,
                )
                result = ApiHandler.poll_and_get_result(req_info, timeout=timeout_sec, debug=debug_log)
            else:
                result = ApiHandler.submit_and_get_result(
                    "fal-ai/qwen-image-edit-2511-multiple-angles",
                    arguments,
                )

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
                    f"[Qwen Image Edit 2511 Multiple Angles ERROR] "
                    f"attempt {attempt}/{total_attempts} -> {e}"
                ),
            )
            return (img_tensor,)
        except Exception as e:
            print(f"Error generating image with Qwen Image Edit 2511 Multiple Angles: {str(e)}")
            return ApiHandler.handle_image_generation_error("Qwen Image Edit 2511 Multiple Angles", e)
