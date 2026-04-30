from comfy_api.latest import io, ui
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch
import folder_paths
import os
import piexif
import json
import time
import random

class ImageSaveHelperExt():
    @staticmethod
    def dump_extra_info(cls: type[io.ComfyNode] | None):
        if cls is None:
            return None
        info_dict = {}
        if cls.hidden.extra_pnginfo:
            for x in cls.hidden.extra_pnginfo:
                info_dict[x] = json.dumps(cls.hidden.extra_pnginfo[x])
        return info_dict

    @staticmethod
    def create_metadata_exif(cls: type[io.ComfyNode] | None) -> bytes | None:
        if cls is None:
            return None
        exif_dict = {"Exif": {}}
        info_dict = {}
        if cls.hidden.prompt:
            info_dict = {"prompt": cls.hidden.prompt}
        if cls.hidden.extra_pnginfo:
            info_dict.update(ImageSaveHelperExt.dump_extra_info(cls))
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = json.dumps(info_dict).encode('utf-8')
        return piexif.dump(exif_dict)

    @staticmethod
    def create_metadata_png(cls: type[io.ComfyNode] | None) -> PngInfo | None:
        if cls is None:
            return None
        pnginfo = PngInfo()
        if cls.hidden.prompt:
            pnginfo.add_text("prompt", json.dumps(cls.hidden.prompt))
        if cls.hidden.extra_pnginfo:
            for x in cls.hidden.extra_pnginfo:
                pnginfo.add_text(x, json.dumps(cls.hidden.extra_pnginfo[x]))
        return pnginfo

    @staticmethod
    def create_metadata_tiff(cls: type[io.ComfyNode] | None):
        if cls is None:
            return None
        from PIL.TiffImagePlugin import ImageFileDirectory_v2
        tiffinfo = ImageFileDirectory_v2()
        info_dict = {}
        if cls.hidden.prompt:
            info_dict["prompt"] = cls.hidden.prompt
        if cls.hidden.extra_pnginfo:
            info_dict.update(ImageSaveHelperExt.dump_extra_info(cls))
        tiffinfo[270] = json.dumps(info_dict)
        return tiffinfo

    @staticmethod
    def get_save_result(
        image: torch.Tensor,
        mask: torch.Tensor | None,
        convert_mode: str,
        join_mask: bool,
        filename: str,
        full_output_folder: str,
        full_input_folder: str,
        subfolder: str,
        batch_number: int,
        counter: int,
        file_ext: str,
        save_to_input_folder: bool = False,
        save_kwargs: dict = {}
    ) -> ui.SavedResult:
        i = 255 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        img = img.convert(convert_mode)

        if mask is not None:
            if convert_mode != "RGBA" or join_mask is False:
                mask = None
            else: 
                m = 255 * mask.cpu().numpy()
                m_pil = Image.fromarray(m.astype(np.uint8), mode="L")
                if m.shape != (img.height, img.width):
                    m_pil = m_pil.resize((img.width, img.height), Image.LANCZOS)
                img.putalpha(m_pil)

        filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
        file = f"{filename_with_batch_num}_{counter:05}_.{file_ext}"

        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder)
        img.save(os.path.join(full_output_folder, file), **save_kwargs)
        if (save_to_input_folder):
            if not os.path.exists(full_input_folder):
                os.makedirs(full_input_folder)
            target_file = os.path.join(full_input_folder, file)
            if os.path.exists(target_file):
                target_file = os.path.join(full_input_folder, f"{filename_with_batch_num}_{counter:05}_{int(time.time())}.jpg")
            img.save(target_file, **save_kwargs)
        return ui.SavedResult(file, subfolder, io.FolderType.output)

    @staticmethod
    def get_save_result_temp(image: torch.Tensor) -> ui.SavedResult:
        tmp_file = "ComfyUI_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5)) + ".png"
        Image.fromarray(np.clip(255 * image.cpu().numpy(), 0, 255).astype(np.uint8)).convert("RGBA").save(os.path.join(folder_paths.get_temp_directory(), tmp_file))
        return ui.SavedResult(tmp_file, "", io.FolderType.temp)