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
        save_kwargs: dict = {},
        icc_profile: bytes | None = None
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
        img.save(os.path.join(full_output_folder, file), icc_profile=icc_profile, **save_kwargs)
        if (save_to_input_folder):
            if not os.path.exists(full_input_folder):
                os.makedirs(full_input_folder)
            target_file = os.path.join(full_input_folder, file)
            if os.path.exists(target_file):
                target_file = os.path.join(full_input_folder, f"{filename_with_batch_num}_{counter:05}_{int(time.time())}.jpg")
            img.save(target_file, icc_profile=icc_profile, **save_kwargs)
        return ui.SavedResult(file, subfolder, io.FolderType.output)

    @staticmethod
    def get_save_result_temp(image: torch.Tensor, mask: torch.Tensor | None = None) -> ui.SavedResult:
        tmp_file = "ComfyUI_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5)) + ".png"
        
        i = 255 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGBA")
        
        if mask is not None:
            m = 255 * mask.cpu().numpy()
            m_pil = Image.fromarray(m.astype(np.uint8), mode="L")
            if m.shape != (img.height, img.width):
                m_pil = m_pil.resize((img.width, img.height), Image.LANCZOS)
            img.putalpha(m_pil)

        img.save(os.path.join(folder_paths.get_temp_directory(), tmp_file))
        return ui.SavedResult(tmp_file, "", io.FolderType.temp)

    @staticmethod
    def to_pillow_image(image: torch.Tensor , number: int = 0) -> Image:
        """
        Explicitly extract an image from the tensor batch.
        """
        img_np = image.cpu().numpy()

        # (B, H, W, C)
        if img_np.ndim == 4:
            if img_np.shape[0] > 0:
                number = max(0, min(number, img_np.shape[0] - 1))
                img_np = img_np[number]
        
        if img_np.dtype in (np.float32, np.float64):
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
 
        if img_np.ndim == 3:
            if img_np.shape[2] == 4:
                pil_image = Image.fromarray(img_np, mode="RGBA")
            elif img_np.shape[2] == 3:
                pil_image = Image.fromarray(img_np, mode="RGB")
            else:
                pil_image = Image.fromarray(img_np.squeeze(), mode="L")
        else:
            pil_image = Image.fromarray(img_np, mode="L")

        return pil_image

    @staticmethod
    def to_pillow_images(image: torch.Tensor) -> list[Image]:
        """
        Convert the tensor batch to pillow image batch.
        """
        img_np = image.detach().cpu().numpy()

        if img_np.ndim == 3:
            img_np = img_np[None, ...]  # to (1, H, W, C)
        elif img_np.ndim == 2:
            img_np = img_np[None, ..., None] # to (1, H, W, 1)

        images = []
        if img_np.ndim == 4:
            for img in img_np:
                if img.dtype in (np.float32, np.float64):
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)

                if img.ndim == 3:
                    if img.shape[2] == 4:
                        pil_image = Image.fromarray(img, mode="RGBA")
                    elif img.shape[2] == 3:
                        pil_image = Image.fromarray(img, mode="RGB")
                    else:
                        pil_image = Image.fromarray(img.squeeze(), mode="L")
                else:
                    pil_image = Image.fromarray(img, mode="L")

                images.append(pil_image)

        return images
