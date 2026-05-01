import os
import folder_paths
from .image_process import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

icc_folder_path = os.path.join(folder_paths.models_dir, "icc_profiles")
if not os.path.exists(icc_folder_path):
    os.makedirs(icc_folder_path)

folder_paths.add_model_folder_path("icc_profiles", icc_folder_path)

WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']