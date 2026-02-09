import os
import io
import time
import torch
import requests
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from .fal_utils import FalConfig, ImageUtils


class PVL_fal_EvfSam_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "semantic_type": ("BOOLEAN", {"default": False}),
                "mask_only": ("BOOLEAN", {"default": True}),
                "use_grounding_dino": ("BOOLEAN", {"default": False}),
                "revert_mask": ("BOOLEAN", {"default": False}),
                "blur_mask": ("INT", {"default": 0, "min": 0, "max": 99, "step": 1}),
                "expand_mask": ("INT", {"default": 0, "min": 0, "max": 99, "step": 1}),
                "fill_holes": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "use_mstudio_proxy": ("BOOLEAN", {"default": False}),
                "Proxy Only if >1K": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "PVL_tools_FAL"

    def generate_image(self, *args, **kwargs):
        return self.process_image(*args, **kwargs)

    # -----------------------------
    # SINGLE REQUEST SUBMISSION
    # -----------------------------
    def _submit_single(
        self,
        image_tensor,
        prompt,
        negative_prompt,
        semantic_type,
        mask_only,
        use_grounding_dino,
        revert_mask,
        blur_mask,
        expand_mask,
        fill_holes,
        seed,
        debug,
        **kwargs,
    ):
        """
        Submit one image to FAL evf-sam endpoint and return request info.
        """
        use_mstudio_proxy, proxy_only_if_gt_1k = ImageUtils.get_proxy_options(kwargs)
        image_url = ImageUtils.image_to_payload_uri(
            image_tensor,
            use_mstudio_proxy=use_mstudio_proxy,
            proxy_only_if_gt_1k=proxy_only_if_gt_1k,
        )
        if debug:
            print(f"[PVL Segment DEBUG] Encoded image as inline data URI ({len(image_url)} chars)")

        # Prepare arguments
        arguments = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "semantic_type": semantic_type,
            "image_url": image_url,
            "mask_only": mask_only,
            "use_grounding_dino": use_grounding_dino,
            "revert_mask": revert_mask,
            "blur_mask": blur_mask if blur_mask % 2 == 1 or blur_mask == 0 else blur_mask + 1,  # must be odd
            "expand_mask": expand_mask,
            "fill_holes": fill_holes,
        }
        if seed != -1:
            arguments["seed"] = seed

        endpoint = "fal-ai/evf-sam"
        fal_key = FalConfig.get_api_key()
        if not fal_key:
            raise RuntimeError("FAL_KEY environment variable not set")

        base = "https://queue.fal.run"
        submit_url = f"{base}/{endpoint}"
        headers = {"Authorization": f"Key {fal_key}"}

        if debug:
            print(f"[PVL Segment DEBUG] Submitting to {endpoint} with args (keys): {list(arguments.keys())}")

        r = requests.post(submit_url, headers=headers, json=arguments, timeout=120)
        if not r.ok:
            raise RuntimeError(f"FAL submit error {r.status_code}: {r.text}")

        sub = r.json()
        req_id = sub.get("request_id")
        if not req_id:
            raise RuntimeError("FAL did not return a request_id")

        if debug:
            print(f"[PVL Segment DEBUG] Request submitted: {req_id}")

        return {
            "request_id": req_id,
            "status_url": sub.get("status_url") or f"{base}/{endpoint}/requests/{req_id}/status",
            "response_url": sub.get("response_url") or f"{base}/{endpoint}/requests/{req_id}",
        }

    # -----------------------------
    # POLLING UNTIL COMPLETED
    # -----------------------------
    def _poll_and_fetch(self, request_info, debug=False, timeout=180):
        fal_key = FalConfig.get_api_key()
        headers = {"Authorization": f"Key {fal_key}"}
        status_url = request_info["status_url"]
        resp_url = request_info["response_url"]

        if debug:
            print(f"[PVL Segment DEBUG] Polling status: {status_url}")

        start = time.time()
        while True:
            if time.time() - start > timeout:
                raise RuntimeError(f"FAL request timed out after {timeout}s")

            try:
                sr = requests.get(status_url, headers=headers, timeout=10)
                if sr.ok:
                    sdata = sr.json()
                    if sdata.get("status") == "COMPLETED":
                        break
                    if sdata.get("status") == "FAILED":
                        raise RuntimeError(f"FAL request failed: {sdata}")
            except Exception as e:
                if debug:
                    print(f"[PVL Segment DEBUG] Poll error: {e}")
            time.sleep(0.6)

        rr = requests.get(resp_url, headers=headers, timeout=30)
        if not rr.ok:
            raise RuntimeError(f"FAL result fetch error {rr.status_code}: {rr.text}")

        result = rr.json().get("response", rr.json())
        if debug:
            print(f"[PVL Segment DEBUG] Response keys: {list(result.keys())}")

        # Extract image result
        if "image" in result and isinstance(result["image"], dict):
            url = result["image"].get("url")
            if not url:
                raise RuntimeError("FAL returned no image URL")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
            arr = np.array(pil).astype("float32") / 255.0
            return torch.from_numpy(arr).unsqueeze(0)
        else:
            raise RuntimeError("Unexpected FAL result format")

    # -----------------------------
    # MAIN ENTRYPOINT
    # -----------------------------
    def process_image(
        self,
        image,
        prompt,
        negative_prompt,
        semantic_type,
        mask_only,
        use_grounding_dino,
        revert_mask,
        blur_mask,
        expand_mask,
        fill_holes,
        seed,
        debug,
        **kwargs,
    ):
        use_mstudio_proxy, proxy_only_if_gt_1k = ImageUtils.get_proxy_options(kwargs)
        if not torch.is_tensor(image):
            raise RuntimeError("Input must be IMAGE tensor")

        # Handle batch
        if image.ndim == 4:
            batch_count = image.shape[0]
            images = [image[i] for i in range(batch_count)]
        else:
            images = [image]

        if debug:
            print(f"[PVL Segment INFO] Processing {len(images)} image(s)")

        submit_info = []
        results = {}
        max_workers = min(len(images), 6)

        # ---- PHASE 1: Submit all ----
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    self._submit_single,
                    images[i],
                    prompt,
                    negative_prompt,
                    semantic_type,
                    mask_only,
                    use_grounding_dino,
                    revert_mask,
                    blur_mask,
                    expand_mask,
                    fill_holes,
                    (seed + i) % 4294967296 if seed != -1 else -1,
                    debug,
                    use_mstudio_proxy=use_mstudio_proxy,
                    proxy_only_if_gt_1k=proxy_only_if_gt_1k,
                ): i
                for i in range(len(images))
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    info = fut.result()
                    submit_info.append((idx, info))
                except Exception as e:
                    print(f"[PVL Segment ERROR] Submission failed for image {idx}: {e}")

        if not submit_info:
            raise RuntimeError("All submissions failed")

        if debug:
            print(f"[PVL Segment DEBUG] {len(submit_info)} requests submitted successfully")

        # ---- PHASE 2: Poll & Fetch ----
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(self._poll_and_fetch, info, debug): idx for idx, info in submit_info}
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    print(f"[PVL Segment ERROR] Poll failed for {idx}: {e}")

        if not results:
            raise RuntimeError("All FAL polling requests failed")

        ordered = [results[i] for i in sorted(results.keys()) if i in results]
        out = torch.cat(ordered, dim=0)

        if debug:
            print(f"[PVL Segment INFO] Successfully processed {out.shape[0]} image(s)")

        return (out,)
