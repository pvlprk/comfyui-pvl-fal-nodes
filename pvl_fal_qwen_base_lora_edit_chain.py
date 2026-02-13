import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from .fal_utils import ApiHandler, ImageUtils, ResultProcessor


class PVL_fal_QwenBaseLoraEditChain_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": (
                    ["fal-ai/qwen-image", "fal-ai/qwen-image-2512/lora"],
                    {"default": "fal-ai/qwen-image"},
                ),
                "prompt_base": ("STRING", {"multiline": True}),
                "prompt_edit": ("STRING", {"multiline": True}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "retries": ("INT", {"default": 2, "min": 0, "max": 10}),
                "timeout_sec": ("INT", {"default": 120, "min": 5, "max": 600, "step": 5}),
                "debug_log": ("BOOLEAN", {"default": False}),
                "output_stage1": ("BOOLEAN", {"default": False}),
                "base_num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 50}),
                "base_guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "base_acceleration": (["none", "regular", "high"], {"default": "regular"}),
                "base_enable_safety_checker": ("BOOLEAN", {"default": True}),
                "base_output_format": (["jpeg", "png"], {"default": "png"}),
                "edit_num_inference_steps": ("INT", {"default": 4, "min": 4, "max": 8}),
                "edit_enable_safety_checker": ("BOOLEAN", {"default": True}),
                "edit_output_format": (["jpeg", "png", "webp"], {"default": "png"}),
            },
            "optional": {
                "delimiter": ("STRING", {"default": "[++]", "multiline": False}),
                "base_negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "base_image_size": (
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
                "base_custom_width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
                "base_custom_height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "image": ("IMAGE",),
                "lora1_path": ("STRING", {"default": ""}),
                "lora1_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "lora2_path": ("STRING", {"default": ""}),
                "lora2_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "lora3_path": ("STRING", {"default": ""}),
                "lora3_scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image_stage1", "image_final")
    FUNCTION = "generate_image"
    CATEGORY = "PVL_tools_FAL"

    def _is_content_policy_violation(self, message_or_json) -> bool:
        checker = getattr(ApiHandler, "_is_content_policy_violation", None)
        if callable(checker):
            try:
                return bool(checker(message_or_json))
            except Exception:
                pass
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
                return "content_policy_violation" in str(message_or_json).lower()
            if isinstance(message_or_json, str):
                return "content_policy_violation" in message_or_json.lower()
        except Exception:
            pass
        return False

    def _build_base_image_size(self, base_image_size, base_custom_width, base_custom_height):
        if base_image_size == "custom":
            if int(base_custom_width) > 0 and int(base_custom_height) > 0:
                return {"width": int(base_custom_width), "height": int(base_custom_height)}
            return None
        return base_image_size

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
                    f"[Qwen Chain] base prompts={len(base_prompts)} num_images={n}; "
                    "reusing last base prompt."
                )
            call_prompts = base_prompts + [base_prompts[-1]] * (n - len(base_prompts))
        if debug:
            for i, p in enumerate(call_prompts):
                show = p if len(p) <= 140 else (p[:137] + "...")
                print(f"[Qwen Chain] item={i} base_prompt={show}")
        return call_prompts

    def _submit_and_poll(self, model_id, arguments, timeout_sec=120, debug=False):
        if hasattr(ApiHandler, "submit_only") and hasattr(ApiHandler, "poll_and_get_result"):
            req_info = ApiHandler.submit_only(model_id, arguments, timeout=timeout_sec, debug=debug)
            return ApiHandler.poll_and_get_result(req_info, timeout=timeout_sec, debug=debug)
        return ApiHandler.submit_and_get_result(model_id, arguments)

    def _extract_first_image_url(self, result):
        if not isinstance(result, dict):
            raise RuntimeError("Stage 1 returned a non-dict payload.")
        images = result.get("images")
        if not isinstance(images, list) or not images:
            raise RuntimeError("Stage 1 returned no images.")
        first = images[0]
        if not isinstance(first, dict):
            raise RuntimeError("Stage 1 first image entry is not an object.")
        image_url = first.get("url") or first.get("content")
        if not isinstance(image_url, str) or not image_url.strip():
            raise RuntimeError("Stage 1 returned image without URL/content.")
        return image_url

    def _run_stage1_with_retries(
        self,
        item_index,
        qwen_model,
        prompt_text,
        base_num_inference_steps,
        base_guidance_scale,
        base_acceleration,
        base_enable_safety_checker,
        base_output_format,
        base_negative_prompt,
        base_image_size,
        base_custom_width,
        base_custom_height,
        seed,
        loras,
        output_stage1,
        retries,
        timeout_sec,
        debug_log,
    ):
        seed_for_item = seed if int(seed) == -1 else ((int(seed) + item_index) % 4294967296)

        def action(attempt, total_attempts):
            if debug_log:
                print(
                    f"[Qwen Chain][Stage1] item={item_index} attempt={attempt}/{total_attempts} "
                    f"seed={seed_for_item if int(seed) != -1 else 'auto'}"
                )
            arguments = {
                "prompt": prompt_text,
                "num_inference_steps": int(base_num_inference_steps),
                "guidance_scale": float(base_guidance_scale),
                "num_images": 1,
                "acceleration": base_acceleration,
                "enable_safety_checker": bool(base_enable_safety_checker),
                "output_format": base_output_format,
                "sync_mode": False,
            }
            size_payload = self._build_base_image_size(base_image_size, base_custom_width, base_custom_height)
            if size_payload is not None:
                arguments["image_size"] = size_payload
            if isinstance(base_negative_prompt, str) and base_negative_prompt.strip():
                arguments["negative_prompt"] = base_negative_prompt
            if int(seed) != -1:
                arguments["seed"] = int(seed_for_item) & 0xFFFFFFFF
            if loras:
                arguments["loras"] = loras
            if debug_log:
                print(
                    f"[Qwen Chain][Stage1] item={item_index} model={qwen_model} "
                    f"payload_keys={list(arguments.keys())} loras={len(loras)}"
                )
            result = self._submit_and_poll(
                qwen_model,
                arguments,
                timeout_sec=timeout_sec,
                debug=debug_log,
            )
            image_url = self._extract_first_image_url(result)
            stage1_tensor = None
            if bool(output_stage1):
                out = ResultProcessor.process_image_result(result)
                stage1_tensor = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(stage1_tensor) and stage1_tensor.ndim == 3:
                    stage1_tensor = stage1_tensor.unsqueeze(0)
            if debug_log:
                print(f"[Qwen Chain][Stage1] item={item_index} got image URL.")
            return image_url, stage1_tensor

        def on_retry(attempt, total_attempts, error):
            print(
                f"[Qwen Chain][Stage1 ERROR] item={item_index} "
                f"attempt={attempt}/{total_attempts} -> {error}"
            )

        try:
            stage1_url = ApiHandler.run_with_retries(
                action,
                retries=retries,
                is_fatal=lambda e: self._is_content_policy_violation(str(e)),
                on_retry=on_retry,
            )
            if isinstance(stage1_url, tuple) and len(stage1_url) == 2:
                return True, stage1_url[0], stage1_url[1], ""
            return True, stage1_url, None, ""
        except Exception as e:
            return False, None, None, str(e)

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

        def action(attempt, total_attempts):
            if debug_log:
                print(
                    f"[Qwen Chain][Stage2] item={item_index} attempt={attempt}/{total_attempts} "
                    f"seed={seed_for_item if int(seed) != -1 else 'auto'}"
                )
            arguments = {
                "prompt": prompt_edit,
                "image_urls": [stage1_url],
                "num_inference_steps": int(edit_num_inference_steps),
                "num_images": 1,
                "enable_safety_checker": bool(edit_enable_safety_checker),
                "output_format": edit_output_format,
                "sync_mode": False,
            }
            if image is not None and torch.is_tensor(image):
                image_tensor = image[0] if image.ndim == 4 else image
                second_url = ImageUtils.image_to_payload_uri(
                    image_tensor,
                    use_mstudio_proxy=False,
                    proxy_only_if_gt_1k=False,
                    timeout_sec=int(timeout_sec),
                )
                arguments["image_urls"].append(second_url)
            if int(seed) != -1:
                arguments["seed"] = int(seed_for_item) & 0xFFFFFFFF
            if debug_log:
                print(
                    f"[Qwen Chain][Stage2] item={item_index} "
                    f"payload_keys={list(arguments.keys())}"
                )
            result = self._submit_and_poll(
                "fal-ai/flux-2/klein/9b/base/edit",
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
                f"[Qwen Chain][Stage2 ERROR] item={item_index} "
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

    def _run_one_chain_with_retries(
        self,
        item_index,
        qwen_model,
        prompt_base,
        prompt_edit,
        base_num_inference_steps,
        base_guidance_scale,
        base_acceleration,
        base_enable_safety_checker,
        base_output_format,
        base_negative_prompt,
        base_image_size,
        base_custom_width,
        base_custom_height,
        seed,
        edit_num_inference_steps,
        edit_enable_safety_checker,
        edit_output_format,
        image,
        loras,
        output_stage1,
        retries,
        timeout_sec,
        debug_log,
    ):
        ok1, stage1_url, stage1_tensor, err1 = self._run_stage1_with_retries(
            item_index=item_index,
            qwen_model=qwen_model,
            prompt_text=prompt_base,
            base_num_inference_steps=base_num_inference_steps,
            base_guidance_scale=base_guidance_scale,
            base_acceleration=base_acceleration,
            base_enable_safety_checker=base_enable_safety_checker,
            base_output_format=base_output_format,
            base_negative_prompt=base_negative_prompt,
            base_image_size=base_image_size,
            base_custom_width=base_custom_width,
            base_custom_height=base_custom_height,
            seed=seed,
            loras=loras,
            output_stage1=output_stage1,
            retries=retries,
            timeout_sec=timeout_sec,
            debug_log=debug_log,
        )
        if not ok1:
            return False, None, None, f"Stage1 failed: {err1}"

        ok2, img_tensor, err2 = self._run_stage2_with_retries(
            item_index=item_index,
            stage1_url=stage1_url,
            prompt_edit=prompt_edit,
            edit_num_inference_steps=edit_num_inference_steps,
            edit_enable_safety_checker=edit_enable_safety_checker,
            edit_output_format=edit_output_format,
            image=image,
            seed=seed,
            retries=retries,
            timeout_sec=timeout_sec,
            debug_log=debug_log,
        )
        if not ok2:
            return False, None, None, f"Stage2 failed: {err2}"
        return True, stage1_tensor, img_tensor, ""

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
        edit_enable_safety_checker,
        edit_output_format,
        delimiter="[++]",
        base_negative_prompt="",
        base_image_size="landscape_4_3",
        base_custom_width=0,
        base_custom_height=0,
        seed=-1,
        lora1_path="",
        lora1_scale=1.0,
        lora2_path="",
        lora2_scale=1.0,
        lora3_path="",
        lora3_scale=1.0,
        image=None,
        **kwargs,
    ):
        t0 = time.time()
        width = int(base_custom_width) if base_image_size == "custom" and int(base_custom_width) > 0 else 256
        height = int(base_custom_height) if base_image_size == "custom" and int(base_custom_height) > 0 else 256

        try:
            try:
                base_prompts = [p.strip() for p in re.split(delimiter, prompt_base) if str(p).strip()]
            except re.error:
                print(f"[Qwen Chain WARNING] Invalid regex delimiter '{delimiter}', using literal split.")
                base_prompts = [p.strip() for p in str(prompt_base).split(delimiter) if str(p).strip()]
            if not base_prompts:
                raise RuntimeError("No valid base prompts provided.")

            call_prompts = self._build_call_prompts(base_prompts, num_images, debug=debug_log)
            n = len(call_prompts)
            loras = self._build_loras(
                lora1_path=lora1_path,
                lora1_scale=lora1_scale,
                lora2_path=lora2_path,
                lora2_scale=lora2_scale,
                lora3_path=lora3_path,
                lora3_scale=lora3_scale,
            )

            print(
                f"[Qwen Chain INFO] Processing {n} pair(s) "
                f"with retries={retries}, timeout={timeout_sec}s"
            )

            if n == 1:
                ok, stage1_tensor, img_tensor, last_err = self._run_one_chain_with_retries(
                    item_index=0,
                    qwen_model=qwen_model,
                    prompt_base=call_prompts[0],
                    prompt_edit=prompt_edit,
                    base_num_inference_steps=base_num_inference_steps,
                    base_guidance_scale=base_guidance_scale,
                    base_acceleration=base_acceleration,
                    base_enable_safety_checker=base_enable_safety_checker,
                    base_output_format=base_output_format,
                    base_negative_prompt=base_negative_prompt,
                    base_image_size=base_image_size,
                    base_custom_width=base_custom_width,
                    base_custom_height=base_custom_height,
                    seed=seed,
                    edit_num_inference_steps=edit_num_inference_steps,
                    edit_enable_safety_checker=edit_enable_safety_checker,
                    edit_output_format=edit_output_format,
                    image=image,
                    loras=loras,
                    output_stage1=output_stage1,
                    retries=retries,
                    timeout_sec=timeout_sec,
                    debug_log=debug_log,
                )
                if ok and torch.is_tensor(img_tensor):
                    print(f"[Qwen Chain INFO] Successfully generated 1 image in {time.time() - t0:.2f}s")
                    if bool(output_stage1) and torch.is_tensor(stage1_tensor):
                        stage1_out = stage1_tensor if torch.is_tensor(stage1_tensor) else torch.zeros_like(img_tensor)
                    else:
                        stage1_out = torch.zeros_like(img_tensor)
                    return (stage1_out, img_tensor)
                raise RuntimeError(last_err or "All attempts failed for single chained request")

            print(f"[Qwen Chain INFO] Running {n} chained pairs in parallel...")
            results_map = {}
            stage1_map = {}
            errors_map = {}
            max_workers = min(n, 6)

            def worker(i):
                return i, *self._run_one_chain_with_retries(
                    item_index=i,
                    qwen_model=qwen_model,
                    prompt_base=call_prompts[i],
                    prompt_edit=prompt_edit,
                    base_num_inference_steps=base_num_inference_steps,
                    base_guidance_scale=base_guidance_scale,
                    base_acceleration=base_acceleration,
                    base_enable_safety_checker=base_enable_safety_checker,
                    base_output_format=base_output_format,
                    base_negative_prompt=base_negative_prompt,
                    base_image_size=base_image_size,
                    base_custom_width=base_custom_width,
                    base_custom_height=base_custom_height,
                    seed=seed,
                    edit_num_inference_steps=edit_num_inference_steps,
                    edit_enable_safety_checker=edit_enable_safety_checker,
                    edit_output_format=edit_output_format,
                    image=image,
                    loras=loras,
                    output_stage1=output_stage1,
                    retries=retries,
                    timeout_sec=timeout_sec,
                    debug_log=debug_log,
                )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(worker, i) for i in range(n)]
                for fut in as_completed(futures):
                    i, ok, stage1_tensor, img_tensor, last_err = fut.result()
                    if ok and torch.is_tensor(img_tensor):
                        if bool(output_stage1) and torch.is_tensor(stage1_tensor):
                            stage1_map[i] = stage1_tensor
                        results_map[i] = img_tensor
                    else:
                        errors_map[i] = last_err or "Unknown error"

            if not results_map:
                sample_err = next(iter(errors_map.values()), "All chained requests failed")
                raise RuntimeError(sample_err)

            all_images = [results_map[i] for i in sorted(results_map.keys()) if torch.is_tensor(results_map[i])]
            if not all_images:
                raise RuntimeError("No images were generated from chained API calls")

            try:
                final_tensor = torch.cat(all_images, dim=0)
            except RuntimeError:
                first = all_images[0]
                h, w = int(first.shape[1]), int(first.shape[2])
                fixed = []
                for t in all_images:
                    if t.shape[1] == h and t.shape[2] == w:
                        fixed.append(t)
                    else:
                        pil = ImageUtils.tensor_to_pil(t)
                        resized = pil.resize((w, h))
                        fixed.append(ImageUtils.pil_to_tensor(resized))
                final_tensor = torch.cat(fixed, dim=0)

            print(
                f"[Qwen Chain INFO] Successfully generated {final_tensor.shape[0]}/{n} "
                f"images in {time.time() - t0:.2f}s"
            )

            failed_idxs = sorted(set(range(n)) - set(results_map.keys()))
            if failed_idxs:
                for i in failed_idxs:
                    print(
                        f"[Qwen Chain ERROR] Item {i + 1} failed after {int(retries) + 1} attempt(s): "
                        f"{errors_map.get(i, 'Unknown error')}"
                    )
                print(
                    f"[Qwen Chain WARNING] Returning only {final_tensor.shape[0]}/{n} successful results."
                )

            if int(seed) != -1:
                seed_list = [(int(seed) + i) % 4294967296 for i in range(n)]
                print(f"[Qwen Chain INFO] Seeds used: {seed_list}")

            if bool(output_stage1):
                stage1_images = [
                    stage1_map[i]
                    for i in sorted(stage1_map.keys())
                    if torch.is_tensor(stage1_map[i])
                ]
                if stage1_images and len(stage1_images) == final_tensor.shape[0]:
                    try:
                        stage1_out = torch.cat(stage1_images, dim=0)
                    except RuntimeError:
                        first = stage1_images[0]
                        h, w = int(first.shape[1]), int(first.shape[2])
                        fixed = []
                        for t in stage1_images:
                            if t.shape[1] == h and t.shape[2] == w:
                                fixed.append(t)
                            else:
                                pil = ImageUtils.tensor_to_pil(t)
                                resized = pil.resize((w, h))
                                fixed.append(ImageUtils.pil_to_tensor(resized))
                        stage1_out = torch.cat(fixed, dim=0)
                else:
                    stage1_out = torch.zeros_like(final_tensor)
            else:
                stage1_out = torch.zeros_like(final_tensor)

            return (stage1_out, final_tensor)

        except Exception as e:
            print(f"Error generating image with Qwen Chain: {str(e)}")
            fallback = ApiHandler.handle_image_generation_error(
                "Qwen Base LoRA -> Flux Edit Chain",
                e,
                width=width,
                height=height,
            )
            fallback_img = fallback[0] if isinstance(fallback, tuple) else fallback
            return (fallback_img, fallback_img)
