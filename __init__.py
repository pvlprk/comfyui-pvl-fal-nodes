from .pvl_fal_depth_anything_v2 import PVL_fal_DepthAnythingV2_API
from .pvl_fal_flux2_camera_ctrl import PVL_fal_Flux2CameraCtrl_API
from .pvl_fal_flux2_klein_9b_base_lora import PVL_fal_Flux2Klein9BBaseLora_API
from .pvl_fal_flux2_klein_9b_base_edit_lora import PVL_fal_Flux2Klein9BBaseEditLora_API
from .pvl_fal_flux2_klein_9b_base_lora_edit_chain import (
    PVL_fal_Flux2Klein9BBaseLoraBaseEditChain_API,
)
from .pvl_fal_flux2_dev import PVL_fal_Flux2Dev_API
from .pvl_fal_flux2_klein_9b_edit import PVL_fal_Flux2Klein9BBaseEdit_API
from .pvl_fal_flux2_flex import PVL_fal_Flux2Flex_API
from .pvl_fal_flux2_pro import PVL_fal_Flux2Pro_API
from .pvl_fal_flux_dev import PVL_fal_FluxDev_API
from .pvl_fal_flux_dev_inpaint import PVL_fal_FluxDevInpaint_API
from .pvl_fal_flux_general import PVL_fal_FluxGeneral_API
from .pvl_fal_flux_pro11_ultra import PVL_fal_FluxProV11Ultra_API
from .pvl_fal_flux_pro_fill import PVL_fal_FluxProFill_API
from .pvl_fal_flux_pulid import PVL_fal_FluxPulid_API
from .pvl_fal_flux_with_lora import PVL_fal_FluxWithLora_API
from .pvl_fal_flux_with_lora_pulid import PVL_fal_FluxWithLoraPulID_API
from .pvl_fal_kontext_dev import PVL_fal_KontextDev_API
from .pvl_fal_kontext_dev_inpaint import PVL_fal_KontextDevInpaint_API
from .pvl_fal_kontext_dev_lora import PVL_fal_KontextDevLora_API
from .pvl_fal_kontext_max_multi import PVL_fal_KontextMaxMulti_API
from .pvl_fal_kontext_max_single import PVL_fal_KontextMaxSingle_API
from .pvl_fal_kontext_pro import PVL_fal_KontextPro_API
from .pvl_fal_lumaphoton_flash_reframe import PVL_fal_LumaPhotonFlashReframe_API
from .pvl_fal_lumaphoton_reframe import PVL_fal_LumaPhotonReframe_API
from .pvl_fal_ministudio_avatar import PVL_fal_FluxDevPulidAvatar_API
from .pvl_fal_moondream3_segment import PVL_fal_Moondream3Segment_API
from .pvl_fal_nano_banana_edit import PVL_fal_NanoBanana_API
from .pvl_fal_qwen_image_edit_2511 import PVL_fal_QwenImageEdit2511_API
from .pvl_fal_qwen_image_edit_2511_lora import PVL_fal_QwenImageEdit2511Lora_API
from .pvl_fal_qwen_image_edit_2511_multiple_angles import (
    PVL_fal_QwenImageEdit2511MultipleAngles_API,
)
from .pvl_fal_qwen_dual_lora import PVL_fal_QwenDualLora_API
from .pvl_fal_qwen_base_lora_edit_chain import PVL_fal_QwenBaseLoraEditChain_API
from .pvl_fal_qwen_base_lora_qwen_edit_chain import PVL_fal_QwenBaseLoraQwenEditChain_API
from .pvl_fal_qwen_base_lora_qwen_edit_lora_chain import (
    PVL_fal_QwenBaseLoraQwenEditLoraChain_API,
)
from .pvl_fal_qwen_img_edit_inpaint import PVL_fal_Qwen_Img_Edit_Inpaint
from .pvl_fal_qwen_txt2img import PVL_fal_QwenImage_API
from .pvl_fal_remove_bg_v2 import PVL_fal_RemoveBackground_API
from .pvl_fal_sam3_segment import PVL_fal_Sam3Segmentation_API
from .pvl_fal_seedream4_edit import PVL_fal_Seedream4_API
from .pvl_fal_seedream_45 import PVL_fal_Seedream45_API
from .pvl_fal_seg_florence2 import PVL_fal_SegFlorence2_API
from .pvl_fal_segment import PVL_fal_EvfSam_API
from .pvl_fal_segment_x5 import PVL_fal_EvfSamX5_API

NODE_CLASS_MAPPINGS = {
    "PVL_fal_DepthAnythingV2_API": PVL_fal_DepthAnythingV2_API,
    "PVL_fal_Flux2CameraCtrl_API": PVL_fal_Flux2CameraCtrl_API,
    "PVL_fal_Flux2Klein9BBaseLora_API": PVL_fal_Flux2Klein9BBaseLora_API,
    "PVL_fal_Flux2Klein9BBaseEditLora_API": PVL_fal_Flux2Klein9BBaseEditLora_API,
    "PVL_fal_Flux2Klein9BBaseLoraBaseEditChain_API": PVL_fal_Flux2Klein9BBaseLoraBaseEditChain_API,
    "PVL_fal_Flux2Dev_API": PVL_fal_Flux2Dev_API,
    "PVL_fal_Flux2Klein9BBaseEdit_API": PVL_fal_Flux2Klein9BBaseEdit_API,
    "PVL_fal_Flux2Flex_API": PVL_fal_Flux2Flex_API,
    "PVL_fal_Flux2Pro_API": PVL_fal_Flux2Pro_API,
    "PVL_fal_FluxDev_API": PVL_fal_FluxDev_API,
    "PVL_fal_FluxDevInpaint_API": PVL_fal_FluxDevInpaint_API,
    "PVL_fal_FluxGeneral_API": PVL_fal_FluxGeneral_API,
    "PVL_fal_FluxProV11Ultra_API": PVL_fal_FluxProV11Ultra_API,
    "PVL_fal_FluxProFill_API": PVL_fal_FluxProFill_API,
    "PVL_fal_FluxPulid_API": PVL_fal_FluxPulid_API,
    "PVL_fal_FluxWithLora_API": PVL_fal_FluxWithLora_API,
    "PVL_fal_FluxWithLoraPulID_API": PVL_fal_FluxWithLoraPulID_API,
    "PVL_fal_KontextDev_API": PVL_fal_KontextDev_API,
    "PVL_fal_KontextDevInpaint_API": PVL_fal_KontextDevInpaint_API,
    "PVL_fal_KontextDevLora_API": PVL_fal_KontextDevLora_API,
    "PVL_fal_KontextMaxMulti_API": PVL_fal_KontextMaxMulti_API,
    "PVL_fal_KontextMaxSingle_API": PVL_fal_KontextMaxSingle_API,
    "PVL_fal_KontextPro_API": PVL_fal_KontextPro_API,
    "PVL_fal_LumaPhotonFlashReframe_API": PVL_fal_LumaPhotonFlashReframe_API,
    "PVL_fal_LumaPhotonReframe_API": PVL_fal_LumaPhotonReframe_API,
    "PVL_fal_FluxDevPulidAvatar_API": PVL_fal_FluxDevPulidAvatar_API,
    "PVL_fal_Moondream3Segment_API": PVL_fal_Moondream3Segment_API,
    "PVL_fal_NanoBanana_API": PVL_fal_NanoBanana_API,
    "PVL_fal_QwenImageEdit2511_API": PVL_fal_QwenImageEdit2511_API,
    "PVL_fal_QwenImageEdit2511Lora_API": PVL_fal_QwenImageEdit2511Lora_API,
    "PVL_fal_QwenImageEdit2511MultipleAngles_API": PVL_fal_QwenImageEdit2511MultipleAngles_API,
    "PVL_fal_QwenDualLora_API": PVL_fal_QwenDualLora_API,
    "PVL_fal_QwenBaseLoraEditChain_API": PVL_fal_QwenBaseLoraEditChain_API,
    "PVL_fal_QwenBaseLoraQwenEditChain_API": PVL_fal_QwenBaseLoraQwenEditChain_API,
    "PVL_fal_QwenBaseLoraQwenEditLoraChain_API": PVL_fal_QwenBaseLoraQwenEditLoraChain_API,
    "PVL_fal_Qwen_Img_Edit_Inpaint": PVL_fal_Qwen_Img_Edit_Inpaint,
    "PVL_fal_QwenImage_API": PVL_fal_QwenImage_API,
    "PVL_fal_RemoveBackground_API": PVL_fal_RemoveBackground_API,
    "PVL_fal_Sam3Segmentation_API": PVL_fal_Sam3Segmentation_API,
    "PVL_fal_Seedream4_API": PVL_fal_Seedream4_API,
    "PVL_fal_Seedream45_API": PVL_fal_Seedream45_API,
    "PVL_fal_SegFlorence2_API": PVL_fal_SegFlorence2_API,
    "PVL_fal_EvfSam_API": PVL_fal_EvfSam_API,
    "PVL_fal_EvfSamX5_API": PVL_fal_EvfSamX5_API,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_fal_DepthAnythingV2_API": "PVL Depth Anything V2 (fal.ai)",
    "PVL_fal_Flux2CameraCtrl_API": "PVL Flux.2 Camera Control (fal.ai)",
    "PVL_fal_Flux2Klein9BBaseLora_API": "PVL Flux.2 Klein 9B Base LoRA (fal.ai)",
    "PVL_fal_Flux2Klein9BBaseEditLora_API": "PVL Flux.2 Klein 9B Base Edit LoRA (fal.ai)",
    "PVL_fal_Flux2Klein9BBaseLoraBaseEditChain_API": "PVL Flux.2 Klein 9B Base LoRA -> Base Edit Chain (fal.ai)",
    "PVL_fal_Flux2Dev_API": "PVL Flux.2 Dev (fal.ai)",
    "PVL_fal_Flux2Klein9BBaseEdit_API": "PVL Flux.2 Klein 9B Base Edit (fal.ai)",
    "PVL_fal_Flux2Flex_API": "PVL Flux.2 Flex (fal.ai)",
    "PVL_fal_Flux2Pro_API": "PVL Flux.2 Pro (fal.ai)",
    "PVL_fal_FluxDev_API": "PVL FLUX DEV (fal.ai)",
    "PVL_fal_FluxDevInpaint_API": "PVL Flux Dev Inpaint (fal.ai)",
    "PVL_fal_FluxGeneral_API": "PVL FLUX General (fal.ai)",
    "PVL_fal_FluxProV11Ultra_API": "PVL FluxPro1.1 Ultra (fal.ai)",
    "PVL_fal_FluxProFill_API": "PVL FluxPro Fill (fal.ai)",
    "PVL_fal_FluxPulid_API": "PVL Flux PuLID (fal.ai)",
    "PVL_fal_FluxWithLora_API": "PVL FLUX DEV LORA (fal.ai)",
    "PVL_fal_FluxWithLoraPulID_API": "PVL Flux Lora PulID (fal.ai)",
    "PVL_fal_KontextDev_API": "PVL Kontext Dev (fal.ai)",
    "PVL_fal_KontextDevInpaint_API": "PVL Kontext Dev Inpaint (fal.ai)",
    "PVL_fal_KontextDevLora_API": "PVL Kontext Dev LoRA (fal.ai)",
    "PVL_fal_KontextMaxMulti_API": "PVL KONTEXT MAX MULTI (fal.ai)",
    "PVL_fal_KontextMaxSingle_API": "PVL Kontext Max Single (fal.ai)",
    "PVL_fal_KontextPro_API": "PVL Kontext Pro (fal.ai)",
    "PVL_fal_LumaPhotonFlashReframe_API": "PVL LumaPhoton Flash Reframe (fal.ai)",
    "PVL_fal_LumaPhotonReframe_API": "PVL LumaPhoton Reframe (fal.ai)",
    "PVL_fal_FluxDevPulidAvatar_API": "PVL Flux Dev Pulid Avatar (fal.ai)",
    "PVL_fal_Moondream3Segment_API": "PVL Moondream3 Segmentation (fal.ai)",
    "PVL_fal_NanoBanana_API": "PVL FAL Nano-Banana Edit",
    "PVL_fal_QwenImageEdit2511_API": "PVL Qwen Image Edit 2511 (fal.ai)",
    "PVL_fal_QwenImageEdit2511Lora_API": "PVL Qwen Image Edit 2511 LoRA (fal.ai)",
    "PVL_fal_QwenImageEdit2511MultipleAngles_API": (
        "PVL Qwen Image Edit 2511 Multiple Angles (fal.ai)"
    ),
    "PVL_fal_QwenDualLora_API": "PVL Qwen Dual LoRA (fal.ai)",
    "PVL_fal_QwenBaseLoraEditChain_API": "PVL Qwen Base LoRA -> Flux Edit Chain (fal.ai)",
    "PVL_fal_QwenBaseLoraQwenEditChain_API": "PVL Qwen Base LoRA -> Qwen Edit 2511 Chain (fal.ai)",
    "PVL_fal_QwenBaseLoraQwenEditLoraChain_API": (
        "PVL Qwen Base LoRA -> Qwen Edit 2511 LoRA Chain (fal.ai)"
    ),
    "PVL_fal_Qwen_Img_Edit_Inpaint": "PVL Qwen Image Edit Inpaint (fal.ai)",
    "PVL_fal_QwenImage_API": "PVL QwenImage txt2img (fal.ai)",
    "PVL_fal_RemoveBackground_API": "PVL Remove Background V2 (fal.ai)",
    "PVL_fal_Sam3Segmentation_API": "PVL Sam3 Segmentation (fal.ai)",
    "PVL_fal_Seedream4_API": "FAL SeeDream4 Edit (fal.ai)",
    "PVL_fal_Seedream45_API": "PVL SeeDream 4.5 (fal.ai)",
    "PVL_fal_SegFlorence2_API": "PVL Seg Florence2 (fal.ai)",
    "PVL_fal_EvfSam_API": "PVL Segment (fal.ai)",
    "PVL_fal_EvfSamX5_API": "PVL Segment X5 (fal.ai)",
}
