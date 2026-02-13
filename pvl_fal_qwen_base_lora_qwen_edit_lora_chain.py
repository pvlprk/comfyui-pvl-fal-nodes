import torch

from .fal_utils import ApiHandler, ImageUtils, ResultProcessor
from .pvl_fal_qwen_base_lora_edit_chain import PVL_fal_QwenBaseLoraEditChain_API


class PVL_fal_QwenBaseLoraQwenEditLoraChain_API(PVL_fal_QwenBaseLoraEditChain_API):
    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["required"]["edit_num_inference_steps"] = (
            "INT",
            {"default": 28, "min": 1, "max": 50},
        )
        base["required"]["edit_guidance_scale"] = (
            "FLOAT",
            {"default": 4.5, "min": 1.0, "max": 20.0, "step": 0.1},
        )
        base["required"]["edit_acceleration"] = (
            ["none", "regular", "high"],
            {"default": "regular"},
        )
        base["optional"]["edit_negative_prompt"] = ("STRING", {"multiline": True, "default": ""})
        base["optional"]["edit_image_size"] = (
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
        )
        base["optional"]["edit_custom_width"] = ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64})
        base["optional"]["edit_custom_height"] = ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64})
        base["optional"]["stage2_use_mstudio_proxy"] = ("BOOLEAN", {"default": False})
        base["optional"]["stage2_proxy_only_if_gt_1k"] = ("BOOLEAN", {"default": False})
        base["optional"]["stage2_lora1_path"] = ("STRING", {"default": ""})
        base["optional"]["stage2_lora1_scale"] = (
            "FLOAT",
            {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1},
        )
        base["optional"]["stage2_lora2_path"] = ("STRING", {"default": ""})
        base["optional"]["stage2_lora2_scale"] = (
            "FLOAT",
            {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1},
        )
        base["optional"]["stage2_lora3_path"] = ("STRING", {"default": ""})
        base["optional"]["stage2_lora3_scale"] = (
            "FLOAT",
            {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1},
        )
        return base

    def _build_edit_image_size(self, image_size, custom_width, custom_height):
        if image_size == "custom":
            if int(custom_width) > 0 and int(custom_height) > 0:
                return {"width": int(custom_width), "height": int(custom_height)}
            return None
        return image_size

    def _run_stage2_with_retries(
        self,
        item_index,
        stage1_url,
        prompt_edit,
        edit_num_inference_steps,
        edit_enable_safety_checker,
        edit_output_format,
        image,
        seed,
        retries,
        timeout_sec,
        debug_log,
    ):
        seed_for_item = seed if int(seed) == -1 else ((int(seed) + item_index) % 4294967296)
        edit_negative_prompt = getattr(self, "_edit_negative_prompt", "")
        stage2_loras = getattr(self, "_stage2_loras", [])
        edit_guidance_scale = float(getattr(self, "_edit_guidance_scale", 4.5))
        edit_acceleration = getattr(self, "_edit_acceleration", "regular")
        edit_image_size = getattr(self, "_edit_image_size", "custom")
        edit_custom_width = int(getattr(self, "_edit_custom_width", 0))
        edit_custom_height = int(getattr(self, "_edit_custom_height", 0))
        stage2_use_mstudio_proxy = bool(getattr(self, "_stage2_use_mstudio_proxy", False))
        stage2_proxy_only_if_gt_1k = bool(getattr(self, "_stage2_proxy_only_if_gt_1k", False))

        def action(attempt, total_attempts):
            if debug_log:
                print(
                    f"[Qwen Chain QwenEdit LoRA][Stage2] item={item_index} attempt={attempt}/{total_attempts} "
                    f"seed={seed_for_item if int(seed) != -1 else 'auto'}"
                )
            arguments = {
                "prompt": prompt_edit,
                "image_urls": [stage1_url],
                "num_inference_steps": int(edit_num_inference_steps),
                "guidance_scale": edit_guidance_scale,
                "num_images": 1,
                "enable_safety_checker": bool(edit_enable_safety_checker),
                "output_format": edit_output_format,
                "acceleration": edit_acceleration,
                "sync_mode": False,
            }
            size_payload = self._build_edit_image_size(
                edit_image_size, edit_custom_width, edit_custom_height
            )
            if size_payload is not None:
                arguments["image_size"] = size_payload
            if isinstance(edit_negative_prompt, str) and edit_negative_prompt.strip():
                arguments["negative_prompt"] = edit_negative_prompt
            if image is not None and torch.is_tensor(image):
                image_tensor = image[0] if image.ndim == 4 else image
                second_url = ImageUtils.image_to_payload_uri(
                    image_tensor,
                    use_mstudio_proxy=stage2_use_mstudio_proxy,
                    proxy_only_if_gt_1k=stage2_proxy_only_if_gt_1k,
                    timeout_sec=int(timeout_sec),
                )
                arguments["image_urls"].append(second_url)
            if int(seed) != -1:
                arguments["seed"] = int(seed_for_item) & 0xFFFFFFFF
            if stage2_loras:
                arguments["loras"] = stage2_loras
            if debug_log:
                print(
                    f"[Qwen Chain QwenEdit LoRA][Stage2] item={item_index} "
                    f"payload_keys={list(arguments.keys())} loras={len(stage2_loras)}"
                )
            result = self._submit_and_poll(
                "fal-ai/qwen-image-edit-2511/lora",
                arguments,
                timeout_sec=timeout_sec,
                debug=debug_log,
            )
            out = ResultProcessor.process_image_result(result)
            img_tensor = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(img_tensor) and img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)
            return img_tensor

        def on_retry(attempt, total_attempts, error):
            print(
                f"[Qwen Chain QwenEdit LoRA][Stage2 ERROR] item={item_index} "
                f"attempt={attempt}/{total_attempts} -> {error}"
            )

        try:
            img_tensor = ApiHandler.run_with_retries(
                action,
                retries=retries,
                is_fatal=lambda e: self._is_content_policy_violation(str(e)),
                on_retry=on_retry,
            )
            return True, img_tensor, ""
        except Exception as e:
            return False, None, str(e)

    def generate_image(
        self,
        qwen_model,
        prompt_base,
        prompt_edit,
        num_images,
        retries,
        timeout_sec,
        debug_log,
        output_stage1,
        base_num_inference_steps,
        base_guidance_scale,
        base_acceleration,
        base_enable_safety_checker,
        base_output_format,
        edit_num_inference_steps,
        edit_guidance_scale,
        edit_acceleration,
        edit_enable_safety_checker,
        edit_output_format,
        delimiter="[++]",
        base_negative_prompt="",
        base_image_size="landscape_4_3",
        base_custom_width=0,
        base_custom_height=0,
        edit_image_size="custom",
        edit_custom_width=0,
        edit_custom_height=0,
        stage2_use_mstudio_proxy=False,
        stage2_proxy_only_if_gt_1k=False,
        seed=-1,
        lora1_path="",
        lora1_scale=1.0,
        lora2_path="",
        lora2_scale=1.0,
        lora3_path="",
        lora3_scale=1.0,
        image=None,
        edit_negative_prompt="",
        stage2_lora1_path="",
        stage2_lora1_scale=1.0,
        stage2_lora2_path="",
        stage2_lora2_scale=1.0,
        stage2_lora3_path="",
        stage2_lora3_scale=1.0,
        **kwargs,
    ):
        self._edit_negative_prompt = edit_negative_prompt
        self._edit_guidance_scale = float(edit_guidance_scale)
        self._edit_acceleration = edit_acceleration
        self._edit_image_size = edit_image_size
        self._edit_custom_width = int(edit_custom_width)
        self._edit_custom_height = int(edit_custom_height)
        self._stage2_use_mstudio_proxy = bool(stage2_use_mstudio_proxy)
        self._stage2_proxy_only_if_gt_1k = bool(stage2_proxy_only_if_gt_1k)
        self._stage2_loras = self._build_loras(
            lora1_path=stage2_lora1_path,
            lora1_scale=stage2_lora1_scale,
            lora2_path=stage2_lora2_path,
            lora2_scale=stage2_lora2_scale,
            lora3_path=stage2_lora3_path,
            lora3_scale=stage2_lora3_scale,
        )
        return super().generate_image(
            qwen_model=qwen_model,
            prompt_base=prompt_base,
            prompt_edit=prompt_edit,
            num_images=num_images,
            retries=retries,
            timeout_sec=timeout_sec,
            debug_log=debug_log,
            output_stage1=output_stage1,
            base_num_inference_steps=base_num_inference_steps,
            base_guidance_scale=base_guidance_scale,
            base_acceleration=base_acceleration,
            base_enable_safety_checker=base_enable_safety_checker,
            base_output_format=base_output_format,
            edit_num_inference_steps=edit_num_inference_steps,
            edit_enable_safety_checker=edit_enable_safety_checker,
            edit_output_format=edit_output_format,
            delimiter=delimiter,
            base_negative_prompt=base_negative_prompt,
            base_image_size=base_image_size,
            base_custom_width=base_custom_width,
            base_custom_height=base_custom_height,
            seed=seed,
            lora1_path=lora1_path,
            lora1_scale=lora1_scale,
            lora2_path=lora2_path,
            lora2_scale=lora2_scale,
            lora3_path=lora3_path,
            lora3_scale=lora3_scale,
            image=image,
            **kwargs,
        )
