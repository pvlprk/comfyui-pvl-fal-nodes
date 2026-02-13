import json
import torch
from .fal_utils import ResultProcessor, ApiHandler

class PVL_fal_QwenImage_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "CFG": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "acceleration": (["none", "regular", "high"], {"default": "none"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "lora1_path": ("STRING", {"default": ""}),
                "lora1_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "lora2_path": ("STRING", {"default": ""}),
                "lora2_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "lora3_path": ("STRING", {"default": ""}),
                "lora3_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                # list of LoRAs: [{"path": "...", "scale": 1.0}, {...}]
                "loras": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "PVL_tools_FAL"

    def _build_loras_from_fields(
        self,
        lora1_path="",
        lora1_scale=1.0,
        lora2_path="",
        lora2_scale=1.0,
        lora3_path="",
        lora3_scale=1.0,
    ):
        lora_list = []
        if isinstance(lora1_path, str) and lora1_path.strip():
            lora_list.append({"path": lora1_path.strip(), "scale": float(lora1_scale)})
        if isinstance(lora2_path, str) and lora2_path.strip():
            lora_list.append({"path": lora2_path.strip(), "scale": float(lora2_scale)})
        if isinstance(lora3_path, str) and lora3_path.strip():
            lora_list.append({"path": lora3_path.strip(), "scale": float(lora3_scale)})
        return lora_list

    def _parse_loras_json(self, loras_text: str):
        if not isinstance(loras_text, str) or not loras_text.strip():
            return []
        try:
            parsed = json.loads(loras_text)
            if isinstance(parsed, list):
                return parsed
            print("Warning: 'loras' JSON is not a list; ignoring.")
        except Exception as e:
            print(f"Warning: could not parse LoRAs input: {e}")
        return []

    def generate_image(self, prompt, width, height, steps, CFG, seed,
                       num_images, enable_safety_checker, output_format,
                      sync_mode, acceleration, negative_prompt,
                      lora1_path="", lora1_scale=1.0,
                      lora2_path="", lora2_scale=1.0,
                      lora3_path="", lora3_scale=1.0,
                      loras=""):
        try:
            arguments = {
                "prompt": prompt,
                "num_inference_steps": steps,
                "guidance_scale": CFG,
                "num_images": num_images,
                "enable_safety_checker": enable_safety_checker,
                "output_format": output_format,
                "sync_mode": sync_mode,
                "image_size": {
                    "width": width,
                    "height": height
                },
                "acceleration": acceleration,
                "negative_prompt": negative_prompt,
            }

            if seed != -1:
                arguments["seed"] = seed

            field_loras = self._build_loras_from_fields(
                lora1_path=lora1_path,
                lora1_scale=lora1_scale,
                lora2_path=lora2_path,
                lora2_scale=lora2_scale,
                lora3_path=lora3_path,
                lora3_scale=lora3_scale,
            )
            json_loras = self._parse_loras_json(loras)
            all_loras = (field_loras + json_loras)[:3]
            if all_loras:
                arguments["loras"] = all_loras

            # Call FAL API for Qwen Image
            result = ApiHandler.submit_and_get_result("fal-ai/qwen-image", arguments)

            # Convert result into ComfyUI image tensor
            return ResultProcessor.process_image_result(result)

        except Exception as e:
            print(f"Error generating image with Qwen-Image: {str(e)}")
            return (torch.zeros((1, 64, 64, 3)),)  # fallback dummy image
