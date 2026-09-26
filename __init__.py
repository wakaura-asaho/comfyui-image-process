import os
import logging
import folder_paths
from comfy_api.latest import ComfyExtension, io
from .image_process import (
    ColorPatchFlatten, ColorPatchMerge,
    AchromaticStabilizer, LoadICCProfile,
    SaveImageAdvancedCustom, SaveImageJPG,
    SaveImageAdvancedJPG, SaveImageBMP, SaveImageAdvancedBMP,
    SaveImageTIFF, SaveImageAdvancedTIFF, SaveImageTGA,
    SaveImageAdvancedTGA, SaveImageAVIF, SaveImageAdvancedAVIF,
    SaveImageICO, SaveImageAdvancedICO
)
from .define import define

logger = logging.getLogger(define.logger_name)

icc_folder_path = os.path.join(folder_paths.models_dir, "icc_profiles")
if not os.path.exists(icc_folder_path):
    os.makedirs(icc_folder_path)

folder_paths.add_model_folder_path("icc_profiles", icc_folder_path)

WEB_DIRECTORY = "./web"
__all__ = ['WEB_DIRECTORY']

class ImageProcessExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ColorPatchFlatten,
            ColorPatchMerge,
            AchromaticStabilizer,
            LoadICCProfile,
            SaveImageAdvancedCustom,
            SaveImageJPG,
            SaveImageAdvancedJPG,
            SaveImageBMP,
            SaveImageAdvancedBMP,
            SaveImageTIFF,
            SaveImageAdvancedTIFF,
            SaveImageTGA,
            SaveImageAdvancedTGA,
            SaveImageAVIF,
            SaveImageAdvancedAVIF,
            SaveImageICO,
            SaveImageAdvancedICO,
        ]

async def comfy_entrypoint() -> ImageProcessExtension:
    return ImageProcessExtension()