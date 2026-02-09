from .fal_utils import ApiHandler, ResultProcessor, ImageUtils


class PVL_fal_FluxProFill_API:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "seed": ("INT", {"default": -1}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
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
        return self.fill_image(*args, **kwargs)

    # ------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------
    def fill_image(
        self,
        prompt,
        image,
        mask,
        seed,
        num_images,
        output_format,
        sync_mode,
        safety_tolerance,
        **kwargs,
    ):
        use_mstudio_proxy, proxy_only_if_gt_1k = ImageUtils.get_proxy_options(kwargs)
        image_b64 = ImageUtils.image_to_payload_uri(
            image,
            use_mstudio_proxy=use_mstudio_proxy,
            proxy_only_if_gt_1k=proxy_only_if_gt_1k,
        )
        mask_b64 = ImageUtils.image_to_payload_uri(
            ImageUtils.mask_to_image(mask),
            use_mstudio_proxy=use_mstudio_proxy,
            proxy_only_if_gt_1k=proxy_only_if_gt_1k,
        )

        args = {
            "prompt": prompt,
            "image_url": image_b64,
            "mask_url": mask_b64,
            "num_images": num_images,
            "output_format": output_format,
            "sync_mode": sync_mode,
            "safety_tolerance": safety_tolerance,
        }

        if seed != -1:
            args["seed"] = seed

        result = ApiHandler.submit_and_get_result(
            "fal-ai/flux-pro/v1/fill",
            args,
        )

        if not result or "images" not in result or not result["images"]:
            raise RuntimeError(f"FAL: no images returned ({result})")

        processed = ResultProcessor.process_image_result(result)
        return processed
