import torch

from .fal_utils import ApiHandler, ResultProcessor


class PVL_fal_QwenDualLora_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": (
                    ["fal-ai/qwen-image", "fal-ai/qwen-image-2512/lora"],
                    {"default": "fal-ai/qwen-image"},
                ),
                "prompt": ("STRING", {"multiline": True}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "acceleration": (["none", "regular", "high"], {"default": "regular"}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "retries": ("INT", {"default": 2, "min": 0, "max": 10}),
                "timeout_sec": ("INT", {"default": 120, "min": 5, "max": 600, "step": 5}),
                "debug_log": ("BOOLEAN", {"default": False}),
            },
            "optional": {
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
                    {"default": "landscape_4_3"},
                ),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "lora1_path": ("STRING", {"default": ""}),
                "lora1_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "lora2_path": ("STRING", {"default": ""}),
                "lora2_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "lora3_path": ("STRING", {"default": ""}),
                "lora3_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "PVL_tools_FAL"

    def _build_image_size(self, image_size, custom_width, custom_height):
        if image_size == "custom":
            if int(custom_width) > 0 and int(custom_height) > 0:
                return {"width": int(custom_width), "height": int(custom_height)}
            return None
        return image_size

    def _build_loras(
        self,
        lora1_path="",
        lora1_scale=1.0,
        lora2_path="",
        lora2_scale=1.0,
        lora3_path="",
        lora3_scale=1.0,
    ):
        loras = []
        if isinstance(lora1_path, str) and lora1_path.strip():
            loras.append({"path": lora1_path.strip(), "scale": float(lora1_scale)})
        if isinstance(lora2_path, str) and lora2_path.strip():
            loras.append({"path": lora2_path.strip(), "scale": float(lora2_scale)})
        if isinstance(lora3_path, str) and lora3_path.strip():
            loras.append({"path": lora3_path.strip(), "scale": float(lora3_scale)})
        return loras[:3]

    def generate_image(
        self,
        qwen_model,
        prompt,
        num_inference_steps,
        guidance_scale,
        num_images,
        acceleration,
        enable_safety_checker,
        output_format,
        sync_mode,
        retries=2,
        timeout_sec=120,
        debug_log=False,
        negative_prompt="",
        image_size="landscape_4_3",
        custom_width=0,
        custom_height=0,
        seed=-1,
        lora1_path="",
        lora1_scale=1.0,
        lora2_path="",
        lora2_scale=1.0,
        lora3_path="",
        lora3_scale=1.0,
        **kwargs,
    ):
        width = int(custom_width) if image_size == "custom" and int(custom_width) > 0 else 256
        height = int(custom_height) if image_size == "custom" and int(custom_height) > 0 else 256

        def action(attempt, total_attempts):
            if debug_log:
                print(
                    f"[Qwen Dual LoRA] attempt {attempt}/{total_attempts} "
                    f"model={qwen_model} num_images={num_images} sync_mode={sync_mode}"
                )

            arguments = {
                "prompt": prompt,
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "num_images": int(num_images),
                "acceleration": acceleration,
                "enable_safety_checker": bool(enable_safety_checker),
                "output_format": output_format,
                "sync_mode": bool(sync_mode),
            }

            size_payload = self._build_image_size(image_size, custom_width, custom_height)
            if size_payload is not None:
                arguments["image_size"] = size_payload

            if isinstance(negative_prompt, str) and negative_prompt.strip():
                arguments["negative_prompt"] = negative_prompt

            if int(seed) != -1:
                arguments["seed"] = int(seed) & 0xFFFFFFFF

            loras = self._build_loras(
                lora1_path=lora1_path,
                lora1_scale=lora1_scale,
                lora2_path=lora2_path,
                lora2_scale=lora2_scale,
                lora3_path=lora3_path,
                lora3_scale=lora3_scale,
            )
            if loras:
                arguments["loras"] = loras

            if debug_log:
                print(
                    f"[Qwen Dual LoRA] payload keys={list(arguments.keys())} "
                    f"loras={len(loras)}"
                )

            if hasattr(ApiHandler, "submit_only") and hasattr(ApiHandler, "poll_and_get_result"):
                req_info = ApiHandler.submit_only(
                    qwen_model, arguments, timeout=timeout_sec, debug=debug_log
                )
                result = ApiHandler.poll_and_get_result(
                    req_info, timeout=timeout_sec, debug=debug_log
                )
            else:
                result = ApiHandler.submit_and_get_result(qwen_model, arguments)

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
                    f"[Qwen Dual LoRA ERROR] attempt {attempt}/{total_attempts} -> {e}"
                ),
            )
            return (img_tensor,)
        except Exception as e:
            print(f"Error generating image with Qwen Dual LoRA: {str(e)}")
            return ApiHandler.handle_image_generation_error(
                "Qwen Dual LoRA",
                e,
                width=width,
                height=height,
            )
