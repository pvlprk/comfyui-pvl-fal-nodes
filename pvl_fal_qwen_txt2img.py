import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

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
                "delimiter": ("STRING", {"default": "[++]", "multiline": False}),
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

    def _build_call_prompts(self, base_prompts, num_images):
        n = max(1, int(num_images))
        if not base_prompts:
            return []
        if len(base_prompts) >= n:
            return base_prompts[:n]
        print(
            f"[Qwen Txt2Img] Provided {len(base_prompts)} prompts but num_images={n}. "
            "Reusing the last prompt for remaining calls."
        )
        return base_prompts + [base_prompts[-1]] * (n - len(base_prompts))

    def _submit_and_poll(self, model_id, arguments):
        if hasattr(ApiHandler, "submit_only") and hasattr(ApiHandler, "poll_and_get_result"):
            req_info = ApiHandler.submit_only(model_id, arguments, timeout=120, debug=False)
            return ApiHandler.poll_and_get_result(req_info, timeout=120, debug=False)
        return ApiHandler.submit_and_get_result(model_id, arguments)

    def _run_one_call(
        self,
        item_index,
        prompt_text,
        width,
        height,
        steps,
        CFG,
        seed,
        enable_safety_checker,
        output_format,
        sync_mode,
        acceleration,
        negative_prompt,
        all_loras,
    ):
        arguments = {
            "prompt": prompt_text,
            "num_inference_steps": int(steps),
            "guidance_scale": float(CFG),
            "num_images": 1,
            "enable_safety_checker": bool(enable_safety_checker),
            "output_format": output_format,
            "sync_mode": bool(sync_mode),
            "image_size": {
                "width": int(width),
                "height": int(height),
            },
            "acceleration": acceleration,
            "negative_prompt": negative_prompt,
        }

        if int(seed) != -1:
            arguments["seed"] = (int(seed) + int(item_index)) % 4294967296

        if all_loras:
            arguments["loras"] = all_loras

        result = self._submit_and_poll("fal-ai/qwen-image", arguments)
        out = ResultProcessor.process_image_result(result)
        img_tensor = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(img_tensor) and img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    def generate_image(self, prompt, width, height, steps, CFG, seed,
                       num_images, enable_safety_checker, output_format,
                      sync_mode, acceleration, negative_prompt,
                      lora1_path="", lora1_scale=1.0,
                      lora2_path="", lora2_scale=1.0,
                      lora3_path="", lora3_scale=1.0,
                      loras="", delimiter="[++]"):
        try:
            prompt_text = str(prompt) if prompt is not None else ""
            try:
                # Preserve single-prompt behavior when the default delimiter token is absent.
                if str(delimiter) == "[++]" and "[++]" not in prompt_text:
                    base_prompts = [prompt_text.strip()] if prompt_text.strip() else []
                else:
                    base_prompts = [p.strip() for p in re.split(delimiter, prompt_text) if str(p).strip()]
            except re.error:
                print(f"[Qwen Txt2Img WARNING] Invalid regex delimiter '{delimiter}', using literal split.")
                base_prompts = [p.strip() for p in prompt_text.split(str(delimiter)) if str(p).strip()]

            if not base_prompts:
                raise RuntimeError("No valid prompts provided.")

            call_prompts = self._build_call_prompts(base_prompts, num_images)
            n = len(call_prompts)

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

            if n == 1:
                img_tensor = self._run_one_call(
                    item_index=0,
                    prompt_text=call_prompts[0],
                    width=width,
                    height=height,
                    steps=steps,
                    CFG=CFG,
                    seed=seed,
                    enable_safety_checker=enable_safety_checker,
                    output_format=output_format,
                    sync_mode=sync_mode,
                    acceleration=acceleration,
                    negative_prompt=negative_prompt,
                    all_loras=all_loras,
                )
                return (img_tensor,)

            print(f"[Qwen Txt2Img INFO] Submitting {n} requests in parallel...")
            results_map = {}
            errors_map = {}
            max_workers = min(n, 6)

            def worker(i):
                try:
                    img_tensor = self._run_one_call(
                        item_index=i,
                        prompt_text=call_prompts[i],
                        width=width,
                        height=height,
                        steps=steps,
                        CFG=CFG,
                        seed=seed,
                        enable_safety_checker=enable_safety_checker,
                        output_format=output_format,
                        sync_mode=sync_mode,
                        acceleration=acceleration,
                        negative_prompt=negative_prompt,
                        all_loras=all_loras,
                    )
                    return i, True, img_tensor, ""
                except Exception as e:
                    return i, False, None, str(e)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(worker, i) for i in range(n)]
                for fut in as_completed(futures):
                    i, ok, img_tensor, err = fut.result()
                    if ok and torch.is_tensor(img_tensor):
                        results_map[i] = img_tensor
                    else:
                        errors_map[i] = err or "Unknown error"

            if not results_map:
                sample_err = next(iter(errors_map.values()), "All FAL requests failed")
                raise RuntimeError(sample_err)

            all_images = [
                results_map[i] for i in sorted(results_map.keys()) if torch.is_tensor(results_map[i])
            ]
            if not all_images:
                raise RuntimeError("No images were generated from API calls.")

            final_tensor = torch.cat(all_images, dim=0)
            failed_idxs = sorted(set(range(n)) - set(results_map.keys()))
            for i in failed_idxs:
                print(f"[Qwen Txt2Img ERROR] Item {i + 1} failed: {errors_map.get(i, 'Unknown error')}")

            return (final_tensor,)

        except Exception as e:
            print(f"Error generating image with Qwen-Image: {str(e)}")
            return (torch.zeros((1, 64, 64, 3)),)  # fallback dummy image
