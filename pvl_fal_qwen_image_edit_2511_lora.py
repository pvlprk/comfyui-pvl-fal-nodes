import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import torch

from .fal_utils import ApiHandler, ImageUtils, ResultProcessor


class PVL_fal_QwenImageEdit2511Lora_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "output_format": (["jpeg", "png", "webp"], {"default": "png"}),
                "acceleration": (["none", "regular", "high"], {"default": "regular"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "retries": ("INT", {"default": 2, "min": 0, "max": 10}),
                "timeout_sec": ("INT", {"default": 120, "min": 5, "max": 600, "step": 5}),
                "debug_log": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "delimiter": ("STRING", {"default": "[++]", "multiline": False}),
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
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "lora1_path": ("STRING", {"default": ""}),
                "lora1_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "lora2_path": ("STRING", {"default": ""}),
                "lora2_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "lora3_path": ("STRING", {"default": ""}),
                "lora3_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
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
                                f"[Qwen Image Edit 2511 LoRA] encoded image_{idx + 1} frame {frame_idx + 1} "
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
                            f"[Qwen Image Edit 2511 LoRA] encoded image_{idx + 1} "
                            f"via {'proxy' if use_proxy_for_image else 'base64'}"
                        )
            except Exception as e:
                print(f"[Qwen Image Edit 2511 LoRA] image_{idx + 1} encode error: {e}")
        if debug:
            print(f"[Qwen Image Edit 2511 LoRA] total encoded images: {len(urls)}")
        return urls

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

    def _build_call_prompts(self, base_prompts, num_images, debug=False):
        n = max(1, int(num_images))
        if not base_prompts:
            return []
        if len(base_prompts) >= n:
            call_prompts = base_prompts[:n]
        else:
            if debug:
                print(
                    f"[Qwen Image Edit 2511 LoRA] Provided {len(base_prompts)} prompts but "
                    f"num_images={n}; reusing the last prompt."
                )
            call_prompts = base_prompts + [base_prompts[-1]] * (n - len(base_prompts))
        return call_prompts

    def _submit_and_poll(self, model_id, arguments, timeout_sec=120, debug=False):
        if hasattr(ApiHandler, "submit_only") and hasattr(ApiHandler, "poll_and_get_result"):
            req_info = ApiHandler.submit_only(model_id, arguments, timeout=timeout_sec, debug=debug)
            return ApiHandler.poll_and_get_result(req_info, timeout=timeout_sec, debug=debug)
        return ApiHandler.submit_and_get_result(model_id, arguments)

    def _run_one_with_retries(
        self,
        item_index,
        prompt_text,
        image_urls,
        num_inference_steps,
        guidance_scale,
        enable_safety_checker,
        output_format,
        acceleration,
        sync_mode,
        seed,
        negative_prompt,
        size_payload,
        loras,
        retries,
        timeout_sec,
        debug_log,
    ):
        seed_for_item = (
            int(seed) if int(seed) == -1 else ((int(seed) + int(item_index)) % 4294967296)
        )

        def action(attempt, total_attempts):
            if debug_log:
                print(
                    f"[Qwen Image Edit 2511 LoRA] item={item_index + 1} "
                    f"attempt {attempt}/{total_attempts}"
                )

            arguments = {
                "prompt": prompt_text,
                "image_urls": image_urls,
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "num_images": 1,
                "enable_safety_checker": bool(enable_safety_checker),
                "output_format": output_format,
                "acceleration": acceleration,
                "sync_mode": bool(sync_mode),
            }

            if size_payload is not None:
                arguments["image_size"] = size_payload
            if isinstance(negative_prompt, str) and negative_prompt.strip():
                arguments["negative_prompt"] = negative_prompt
            if int(seed) != -1:
                arguments["seed"] = int(seed_for_item) & 0xFFFFFFFF
            if loras:
                arguments["loras"] = loras

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

        try:
            img_tensor = ApiHandler.run_with_retries(
                action,
                retries=retries,
                on_retry=lambda attempt, total_attempts, e: print(
                    f"[Qwen Image Edit 2511 LoRA ERROR] item={item_index + 1} "
                    f"attempt {attempt}/{total_attempts} -> {e}"
                ),
            )
            return True, img_tensor, ""
        except Exception as e:
            return False, None, str(e)

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
        delimiter="[++]",
        negative_prompt="",
        image_size="custom",
        custom_width=0,
        custom_height=0,
        seed=-1,
        lora1_path="",
        lora1_scale=1.0,
        lora2_path="",
        lora2_scale=1.0,
        lora3_path="",
        lora3_scale=1.0,
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
        width = int(custom_width) if image_size == "custom" and int(custom_width) > 0 else 256
        height = int(custom_height) if image_size == "custom" and int(custom_height) > 0 else 256

        try:
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

            if not image_urls:
                raise RuntimeError(
                    "fal-ai/qwen-image-edit-2511/lora requires at least one input image."
                )

            prompt_text = str(prompt) if prompt is not None else ""
            try:
                # Preserve single-prompt behavior when the default delimiter token is absent.
                if str(delimiter) == "[++]" and "[++]" not in prompt_text:
                    base_prompts = [prompt_text.strip()] if prompt_text.strip() else []
                else:
                    base_prompts = [p.strip() for p in re.split(delimiter, prompt_text) if str(p).strip()]
            except re.error:
                print(
                    f"[Qwen Image Edit 2511 LoRA WARNING] Invalid regex delimiter "
                    f"'{delimiter}', using literal split."
                )
                base_prompts = [p.strip() for p in prompt_text.split(str(delimiter)) if str(p).strip()]

            if not base_prompts:
                raise RuntimeError("No valid prompts provided.")

            call_prompts = self._build_call_prompts(base_prompts, num_images, debug=debug_log)
            n = len(call_prompts)
            if debug_log:
                print(
                    f"[Qwen Image Edit 2511 LoRA] image_urls={len(image_urls)} "
                    f"num_prompts={len(base_prompts)} calls={n}"
                )
                if len(image_urls) == 1 and n > 1:
                    print(
                        "[Qwen Image Edit 2511 LoRA] single input image provided; "
                        f"requesting {n} output image(s) from the same input."
                    )

            size_payload = self._build_image_size(image_size, custom_width, custom_height)
            loras = self._build_loras(
                lora1_path=lora1_path,
                lora1_scale=lora1_scale,
                lora2_path=lora2_path,
                lora2_scale=lora2_scale,
                lora3_path=lora3_path,
                lora3_scale=lora3_scale,
            )

            if n == 1:
                ok, img_tensor, last_err = self._run_one_with_retries(
                    item_index=0,
                    prompt_text=call_prompts[0],
                    image_urls=image_urls,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    enable_safety_checker=enable_safety_checker,
                    output_format=output_format,
                    acceleration=acceleration,
                    sync_mode=sync_mode,
                    seed=seed,
                    negative_prompt=negative_prompt,
                    size_payload=size_payload,
                    loras=loras,
                    retries=retries,
                    timeout_sec=timeout_sec,
                    debug_log=debug_log,
                )
                if ok and torch.is_tensor(img_tensor):
                    return (img_tensor,)
                raise RuntimeError(last_err or "All attempts failed for single request.")

            print(f"[Qwen Image Edit 2511 LoRA INFO] Submitting {n} requests in parallel...")
            results_map = {}
            errors_map = {}
            max_workers = min(n, 6)

            def worker(i):
                return i, *self._run_one_with_retries(
                    item_index=i,
                    prompt_text=call_prompts[i],
                    image_urls=image_urls,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    enable_safety_checker=enable_safety_checker,
                    output_format=output_format,
                    acceleration=acceleration,
                    sync_mode=sync_mode,
                    seed=seed,
                    negative_prompt=negative_prompt,
                    size_payload=size_payload,
                    loras=loras,
                    retries=retries,
                    timeout_sec=timeout_sec,
                    debug_log=debug_log,
                )

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
                print(
                    f"[Qwen Image Edit 2511 LoRA ERROR] Item {i + 1} failed: "
                    f"{errors_map.get(i, 'Unknown error')}"
                )

            return (final_tensor,)
        except Exception as e:
            print(f"Error generating image with Qwen Image Edit 2511 LoRA: {str(e)}")
            return ApiHandler.handle_image_generation_error(
                "Qwen Image Edit 2511 LoRA",
                e,
                width=width,
                height=height,
            )
