from comfy_api.latest import io, ui
import numpy as np
from PIL import Image, ImageCms
import torch
import folder_paths
import os
from .image_helper import ImageSaveHelperExt
from .define import define

class ColorArtifactNormalizer(io.ComfyNode):
    """
    Fix achromatic color instability from AI-generated images.

    Pixels with near-zero saturation have their hue snapped to 0
    and their saturation zeroed out, then optionally smoothed.
    """

    debug_header = "[ComfyUI-ColorArtifactNormalizer]"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ColorArtifactNormalizer",
            display_name="Color Artifact Normalizer",
            category=define.author,
            is_experimental=True,
            description="Fix achromatic color instability from AI-generated images. "
                        "Pixels with near-zero saturation have their hue snapped to 0 and their saturation zeroed out, then optionally smoothed.",
            inputs=[
                io.Image.Input(
                    id="image",
                    display_name="Image",
                    tooltip="The image to be processed for color artifact normalization."
                ),
                io.Mask.Input(
                    id="mask",
                    display_name="Alpha Mask",
                    tooltip="Optional alpha channel mask. If provided and preserve_alpha is True, this mask will be used as the alpha channel.",
                    optional=True
                ),
                io.Boolean.Input(
                    id="smooth",
                    display_name="Smooth Transitions",
                    tooltip="Whether to apply a smoothing filter to the corrected areas.",
                    default=True
                ),
                io.Float.Input(
                    id="sat_threshold",
                    display_name="Saturation Threshold",
                    tooltip="The saturation level below which pixels are considered artifacts.",
                    step=0.01,
                    round=0.01,
                    default=0.08,
                    max=1.0,
                    min=0.0
                ),
                io.Int.Input(
                    id="kernel_size",
                    display_name="Smoothing Kernel Size",
                    tooltip="The size of the smoothing kernel to apply.",
                    default=3,
                    max=15,
                    min=1
                ),
                io.Boolean.Input(
                    id="preserve_alpha",
                    display_name="Preserve Alpha",
                    tooltip="Whether to preserve the alpha channel during processing.",
                    default=True
                ),
                io.Boolean.Input(
                    id="invert_alpha",
                    display_name="Invert Alpha",
                    tooltip="Whether to invert the alpha channel before saving.",
                    default=False
                )
            ],
            outputs=[
                io.Image.Output(
                    id="output",
                    display_name="OUTPUT"
                ),
                io.Mask.Output(
                    id="mask",
                    display_name="ALPHA"
                )
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, *args, **kwargs) -> int:
        """Fingerprint inputs for caching and consistent evaluation.

        The node does not use branch selection, so use the alpha-preserve
        and smoothing options to distinguish inputs.
        """
        preserve_alpha = bool(kwargs.get("preserve_alpha", False))
        smooth = bool(kwargs.get("smooth", False))
        sat_threshold = float(kwargs.get("sat_threshold", 0.08))
        kernel_size = int(kwargs.get("kernel_size", 3))
        mask = kwargs.get("mask")
        mask_id = id(mask) if mask is not None else 0
        return hash((preserve_alpha, smooth, sat_threshold, kernel_size, mask_id)) & 0xFFFFFFFF

    @classmethod
    def check_lazy_status(cls, *args, **kwargs) -> list[str]:
        """No alternate lazy input branches are exposed by this node."""
        return []

    @classmethod
    def validate_inputs(cls, *args, **kwargs) -> bool:
        preserve_alpha = bool(kwargs.get("preserve_alpha", False))
        image = kwargs.get("image")
        # Only validate if image is already a PIL Image
        if preserve_alpha and isinstance(image, Image.Image) and image.mode not in ("RGBA", "LA", "P"):
            print(
                f"{cls.debug_header} Warning: preserve_alpha requested but source image has no alpha channel. "
                f"   Output will be generated as an opaque image."
            )
        return True

    @classmethod
    def execute(
        cls, image: torch.Tensor,
        mask: torch.Tensor | None = None,
        smooth: bool = True,
        sat_threshold: float = 0.08,
        kernel_size: int = 3,
        preserve_alpha: bool = True,
        invert_alpha: bool = False,
        **kwargs
    ) -> io.NodeOutput:
        # Convert torch.Tensor to numpy array
        image = image.cpu().numpy()
        if image.ndim == 4:
            image = image[0]
        # Convert float images to uint8
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        if image.shape[2] == 4:
            image = Image.fromarray(image, mode="RGBA")
        elif image.shape[2] == 3:
            image = Image.fromarray(image, mode="RGB")
        else:
            image = Image.fromarray(image.squeeze(), mode="L")
        
        img = image
        if preserve_alpha and img.mode == "P":
            img = img.convert("RGBA")

        alpha = None
        # Use provided mask if preserve_alpha is True and mask is provided
        if preserve_alpha and mask is not None:
            # Convert mask input to numpy array
            alpha = mask.cpu().numpy()
            if alpha.ndim == 3:
                alpha = alpha[0]
            # Normalize to [0, 1] if needed
            if alpha.max() > 1.0:
                alpha = np.clip(alpha / 255.0, 0, 1).astype(np.float32)
            img = img.convert("RGB")
        elif preserve_alpha and img.mode in ("RGBA", "LA"):
            alpha = np.array(img.split()[-1], dtype=np.float32) / 255.0
            img = img.convert("RGB")
        else:
            img = img.convert("RGB")

        rgb = np.array(img, dtype=np.float32) / 255.0
        c_max = rgb.max(axis=-1)
        c_min = rgb.min(axis=-1)
        delta = c_max - c_min
        sat = np.where(c_max > 0, delta / c_max, 0.0)

        neutral_mask = sat < sat_threshold
        grey_value = rgb.mean(axis=-1, keepdims=True)
        rgb[neutral_mask] = np.repeat(grey_value, 3, axis=-1)[neutral_mask]

        if smooth:
            try:
                from scipy.ndimage import uniform_filter
                smoothed = uniform_filter(rgb, size=[kernel_size, kernel_size, 1])
                mask3 = np.stack([neutral_mask] * 3, axis=-1)
                rgb = np.where(mask3, smoothed, rgb)
            except Exception as exc:
                print(f"{cls.debug_header} Warning: smoothing failed ({exc}). Output will use unsmoothed correction.")

        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        if preserve_alpha and alpha is not None:
            if alpha.shape[:2] != rgb.shape[:2]:
                mask_image = Image.fromarray(alpha.astype(np.uint8), mode="L")
                mask_image = mask_image.resize(rgb.shape[:2][::-1], Image.LANCZOS)
                alpha = np.array(mask_image)
            if alpha.dtype != np.uint8:
                if alpha.max() <= 1.0:
                    alpha = alpha * 255
                alpha = np.clip(alpha, 0, 255).astype(np.uint8)
            if invert_alpha:
                alpha = 255 - alpha
            result_array = np.concatenate([rgb, alpha[..., np.newaxis]], axis=-1)
            output_image = Image.fromarray(result_array, mode="RGBA")
            mask_tensor = torch.from_numpy(alpha.astype(np.float32) / 255.0).unsqueeze(0)
            print(f"{cls.debug_header} Alpha channel preserved in output image.")
        else:
            output_image = Image.fromarray(rgb)
            mask_tensor = torch.zeros((1, rgb.shape[0], rgb.shape[1]), dtype=torch.float32)
            print(f"{cls.debug_header} No alpha channel preserved; output image is opaque.")

        output_tensor = torch.from_numpy(np.array(output_image, dtype=np.float32) / 255.0).unsqueeze(0)
        
        return io.NodeOutput(output_tensor, mask_tensor)

class LoadICCProfile(io.ComfyNode):
    """
    Loads an ICC color profile from the models/icc_profiles folder.
    """
    icc_folder = "icc_profiles"

    def get_valid_icc_profiles(cls, color_profiles: list[str]) -> list[str]:
        if color_profiles is None or len(color_profiles) < 0:
            return []

        valid_profiles = []
        for cp in color_profiles:
            if cp.lower().endswith(('.icc', '.icm')):
                try:
                    profile = ImageCms.getOpenProfile(folder_paths.get_full_path(cls.icc_folder, cp))
                    if profile:
                        valid_profiles.append(cp)
                except (IOError, TypeError, Exception):
                    print(f"Skipping invalid profile: {cp}")
                    
        return valid_profiles

    @staticmethod
    def get_icc_profile_info(profile: ImageCms.ImageCmsProfile) -> dict[str, str]:
        if profile and isinstance(profile, ImageCms.ImageCmsProfile):
            return {
                "model": profile.profile.model,
                "manufacturer": profile.profile.manufacturer,
                "description": profile.profile.profile_description,
                "copyright": profile.profile.copyright
            }
        else:
            return {}

    @staticmethod
    def get_icc_profile_info_plain_text(profile: ImageCms.ImageCmsProfile) -> str:
        if profile and isinstance(profile, ImageCms.ImageCmsProfile):
            return f"""
                Model: {profile.profile.model}\n
                Manufacturer: {profile.profile.manufacturer}\n
                Description: {profile.profile.profile_description}\n
                Copyright: {profile.profile.copyright}
            """
        else:
            return "no data"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadICCProfile",
            display_name="Load ICC Profile",
            category=define.author,
            inputs=[
                io.Combo.Input(
                    id="profile",
                    display_name="Profile",
                    options=cls.get_valid_icc_profiles(cls, folder_paths.get_filename_list(cls.icc_folder)),
                )
            ],
            outputs=[
                io.Custom("ICC_PROFILE").Output(
                    id="icc_profile",
                    display_name="ICC_PROFILE"
                ),
                io.String.Output(
                    id="icc_profile_info",
                    display_name="ICC_INFO"
                )
            ]
        )

    @classmethod
    def execute(cls, profile: str, **kwargs) -> io.NodeOutput:
        path = folder_paths.get_full_path(cls.icc_folder, profile)
        if not path:
            raise FileNotFoundError(f"ICC profile {profile} not found.")
        with open(path, "rb") as f:
            icc_data = f.read()
        icc_info = ImageCms.getOpenProfile(path)
        return io.NodeOutput(icc_data, cls.get_icc_profile_info_plain_text(icc_info))

class SaveImageAdvanced(io.ComfyNode):
    """
    Saves images to disk with additional options for metadata and compression.
    """
    output_dir = folder_paths.get_output_directory()
    prefix_append = ""

    debug_header = "[ComfyUI-SaveImageAdvanced]"

    web_unsupported_preview_formats = ["tiff", "tga"]
    alpha_supported_formats = ["png", "tiff", "webp", "bmp", "tga"]
    icc_unsupported_formats = ["tga", "bmp"]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageAdvanced",
            display_name="Save Image Advanced",
            category=define.author,
            description="Saves image to disk with additional options for metadata and compression.",
            inputs=[
                io.Image.Input(
                    id="images",
                    display_name="Images",
                    tooltip="The images to be saved."
                ),
                io.Mask.Input(
                    id="masks",
                    display_name="Alpha Masks",
                    tooltip="Optional alpha channel masks to clip the saved images. If provided, these masks will be applied to the corresponding images before saving.",
                    optional=True
                ),
                io.Custom("ICC_PROFILE").Input(
                    id="icc_profile",
                    display_name="ICC Profile",
                    tooltip="Optional ICC color profile to embed into the saved images.",
                    optional=True
                ),
                io.Boolean.Input(
                    id="disable_metadata",
                    display_name="Disable Metadata",
                    tooltip="Whether to include the metadata to the saved image.",
                    default=True
                ),
                io.Boolean.Input(
                    id="join_alpha",
                    display_name="Join Alpha Channel",
                    tooltip="Clip the image with the provided mask.\nPlease note this does not work on JPG format.",
                    default=False
                ),
                io.Boolean.Input(
                    id="invert_alpha",
                    display_name="Invert Alpha",
                    tooltip="Whether to invert the alpha channel before saving.",
                    default=False
                ),
                io.Int.Input(
                    id="dpi",
                    display_name="DPI",
                    tooltip="The DPI to use when saving images.",
                    default=define.printing_dpi,
                    min=1,
                    max=600,
                    display_mode=io.NumberDisplay.slider
                ),
                io.Int.Input(
                    id="compress_level",
                    display_name="Compression Level",
                    tooltip="The compression level to use when saving images as PNG (0-9).",
                    default=4,
                    min=0,
                    max=9,
                    display_mode=io.NumberDisplay.slider
                ),
                io.Int.Input(
                    id="quality",
                    display_name="Quality",
                    tooltip="The quality to use when saving images as JPEG or WebP (1-100).",
                    default=90,
                    min=1,
                    max=100,
                    display_mode=io.NumberDisplay.slider
                ),
                io.Combo.Input(
                    id="tiff_compression",
                    display_name="TIFF Compression",
                    tooltip="The compression to use when saving images as TIFF.",
                    options=["none", "tiff_lzw", "tiff_deflate", "tiff_adobe_deflate", "packbits", "jpeg", "tiff_jpeg", "tiff_ccitt"],
                    default="none"
                ),
                io.Boolean.Input(
                    id="tga_rle",
                    display_name="TGA RLE Compression",
                    tooltip="Whether to use RLE compression when saving as TGA.",
                    default=True
                ),
                io.Combo.Input(
                    id="format",
                    display_name="Format",
                    tooltip="The format to save the image in.",
                    options=["png", "jpg", "webp", "tiff", "bmp", "tga"],
                    default="png"
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="Prefix for the saved image filenames.\nEach image will be saved as {prefix}_{index}.{format}.",
                    default=define.default_file_name
                ),
                io.Boolean.Input(
                    id="save_to_input_folder",
                    display_name="Save to Input Folder",
                    tooltip="Whether to sync the image to the input folder.",
                    default=False
                )
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls, images: torch.Tensor,
        masks: torch.Tensor | None = None,
        filename_prefix: str = define.default_file_name,
        disable_metadata: bool = True,
        join_alpha: bool = False,
        invert_alpha: bool = False,
        compress_level: int = 4,
        tiff_compression: str = "none",
        format: str = "png",
        quality: int = 90,
        save_to_input_folder: bool = False,
        dpi: int = define.printing_dpi,
        tga_rle: bool = True,
        icc_profile: bytes | None = None,
        **kwargs
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0])
        f_input_folder = os.path.join(folder_paths.get_input_directory(), subfolder)

        results = []
        tmp_results = [] # For special file format that the web cannot display properly.
        metadata = not disable_metadata and ImageSaveHelperExt.create_metadata_png(cls) or None
        exif_data = not disable_metadata and ImageSaveHelperExt.create_metadata_exif(cls) or None
        for (batch_number, image) in enumerate(images):
            mode = "RGBA"
            save_options = {}
            if format == "jpg":
                mode = "RGB"
                save_options = {"quality": quality, "optimize": True, "dpi": (dpi, dpi)}
                if not disable_metadata:
                    save_options["exif"] = exif_data
            elif format == "webp":
                save_options = {"quality": quality, "lossless": False, "dpi": (dpi, dpi)}
                if not disable_metadata:
                    save_options["exif"] = exif_data
            elif format == "tiff":
                # CCITT compression (Group 3/4 fax) only supports 1-bit bilevel
                # images. Convert to mode "1" to avoid "encoder error -2".
                if tiff_compression == "tiff_ccitt":
                    print(f"{cls.debug_header} tiff_ccitt compression requires a bilevel (1-bit) image.")
                    mode = "1"
                save_options = {"compression": tiff_compression, "dpi": (dpi, dpi)}
                if not disable_metadata:
                    tiffinfo = ImageSaveHelperExt.create_metadata_tiff(cls)
                    if tiffinfo is not None:
                        save_options["tiffinfo"] = tiffinfo
            elif format == "bmp":
                save_options = {"dpi": (dpi, dpi)}
            elif format == "tga":
                save_options = {"compression": "tga_rle" if tga_rle else None}
            else: # PNG
                save_options = {"pnginfo": metadata, "compress_level": compress_level, "dpi": (dpi, dpi)}

            if (format in cls.icc_unsupported_formats) and icc_profile is not None:
                print(f"{cls.debug_header} ICC Profile discarded due to the incompatible format: {format.upper()}")

            mask = None
            if (format in cls.alpha_supported_formats) and join_alpha:
                mask_index = min(batch_number, masks.shape[0] - 1)
                mask = masks[mask_index]
                if invert_alpha:
                    mask = 1.0 - mask

            result = ImageSaveHelperExt.get_save_result(
                image=image, mask=mask, convert_mode=mode, join_mask=join_alpha, filename=filename,
                full_output_folder=f_output_folder, full_input_folder=f_input_folder, subfolder=subfolder,
                batch_number=batch_number, counter=c,
                file_ext=format, save_to_input_folder=save_to_input_folder,
                save_kwargs=save_options,
                icc_profile=icc_profile
            )

            if (format in cls.web_unsupported_preview_formats):
                tmp_results.append(ImageSaveHelperExt.get_save_result_temp(image, mask))
            else:
                results.append(result)

            c += 1

        return io.NodeOutput(ui=(ui.SavedImages(tmp_results) if format in cls.web_unsupported_preview_formats else ui.SavedImages(results)))

class SaveImageJPG(io.ComfyNode):
    """
    Saves images to disk as JPG files.
    """
    output_dir = folder_paths.get_output_directory()
    prefix_append = ""

    debug_header = "[ComfyUI-SaveImageJPG]"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageJPG",
            display_name="Save Image (JPG)",
            category=define.author,
            description="Saves images to disk as JPG files.",
            inputs=[
                io.Image.Input(
                    id="images",
                    display_name="Images",
                    tooltip="The images to be saved."
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="Prefix for the saved image filenames.\nEach image will be saved as {prefix}_{index}.{format}.",
                    default=define.default_file_name
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        filename_prefix: str = define.default_file_name,
        **kwargs
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0])
        results = []
        for (batch_number, image) in enumerate(images):
            result = ImageSaveHelperExt.get_save_result(
                image=image, mask=None, convert_mode="RGB", join_mask=False, filename=filename,
                full_output_folder=f_output_folder, full_input_folder=None, subfolder=subfolder,
                batch_number=batch_number, counter=c, file_ext="jpg", save_to_input_folder=False,
                save_kwargs={"quality": 95, "optimize": True, "dpi": (define.screen_dpi, define.screen_dpi)}
            )

            results.append(result)
            c += 1
            
        return io.NodeOutput(ui=ui.SavedImages(results))

class SaveImageAdvancedJPG(io.ComfyNode):
    """
    Saves images to disk as JPG files with additional options.
    """
    output_dir = folder_paths.get_output_directory()
    prefix_append = ""

    debug_header = "[ComfyUI-SaveImageAdvancedJPG]"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageAdvancedJPG",
            display_name="Save Image Advanced (JPG)",
            category=define.author,
            description="Saves images to disk as JPG files.",
            inputs=[
                io.Image.Input(
                    id="images",
                    display_name="Images",
                    tooltip="The images to be saved."
                ),
                io.Boolean.Input(
                    id="disable_metadata",
                    display_name="Disable Metadata",
                    tooltip="Disable embedding EXIF data into the saved images.\nEmbed into `UserComment`.",
                    default=False
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="The prefix to use for the filename.",
                    default=define.default_file_name
                ),
                io.Int.Input(
                    id="dpi",
                    display_name="DPI",
                    tooltip="The DPI to use when saving images.",
                    default=define.printing_dpi,
                    min=1,
                    max=600,
                    display_mode=io.NumberDisplay.slider
                ),
                io.Int.Input(
                    id="quality",
                    display_name="Quality",
                    tooltip="The quality of the saved images.",
                    default=95,
                    min=1,
                    max=100,
                    display_mode=io.NumberDisplay.slider
                ),
                io.Boolean.Input(
                    id="save_to_input_folder",
                    display_name="Save to Input Folder",
                    tooltip="Whether to sync the image to the input folder.",
                    default=False
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )
    
    @classmethod
    def execute(cls,
        images: torch.Tensor,
        filename_prefix: str = define.default_file_name,
        disable_metadata: bool = False,
        dpi: int = define.screen_dpi,
        quality: int = 95,
        save_to_input_folder: bool = False,
        **kwargs
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0])
        f_input_folder = os.path.join(folder_paths.get_input_directory(), subfolder)

        results = []
        for (batch_number, image) in enumerate(images):
            save_options = {
                "quality": quality,
                "optimize": True,
                "dpi": (dpi, dpi),
            }

            if not disable_metadata:
                exif_data = ImageSaveHelperExt.create_metadata_exif(cls)
                if exif_data is not None:
                    save_options["exif"] = exif_data

            result = ImageSaveHelperExt.get_save_result(
                image=image, mask=None, convert_mode="RGB", join_mask=False, filename=filename,
                full_output_folder=f_output_folder, full_input_folder=f_input_folder, subfolder=subfolder,
                batch_number=batch_number, counter=c,
                file_ext="jpg", save_to_input_folder=save_to_input_folder,
                save_kwargs=save_options
            )

            results.append(result)
            c += 1

        return io.NodeOutput(ui=ui.SavedImages(results))

class SaveImageBMP(io.ComfyNode):
    """
    Saves images to disk as BMP files.
    """
    output_dir = folder_paths.get_output_directory()
    prefix_append = ""

    debug_header = "[ComfyUI-SaveImageBMP]"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageBMP",
            display_name="Save Image (BMP)",
            category=define.author,
            description="Saves images to disk as BMP files.",
            inputs=[
                io.Image.Input(
                    id="images",
                    display_name="Images",
                    tooltip="The images to be saved."
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="Prefix for the saved image filenames.\nEach image will be saved as {prefix}_{index}.{format}.",
                    default=define.default_file_name
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        filename_prefix: str = define.default_file_name,
        **kwargs
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0])
        results = []
        for (batch_number, image) in enumerate(images):
            result = ImageSaveHelperExt.get_save_result(
                image=image, mask=None, convert_mode="RGB", join_mask=False, filename=filename,
                full_output_folder=f_output_folder, full_input_folder=None, subfolder=subfolder,
                batch_number=batch_number, counter=c, file_ext="bmp", save_to_input_folder=False,
                save_kwargs={"dpi": (define.screen_dpi, define.screen_dpi)}
            )

            results.append(result)
            c += 1
            
        return io.NodeOutput(ui=ui.SavedImages(results))

class SaveImageAdvancedBMP(io.ComfyNode):
    """
    Saves images to disk as BMP files with additional options.
    """
    output_dir = folder_paths.get_output_directory()
    prefix_append = ""

    debug_header = "[ComfyUI-SaveImageAdvancedBMP]"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageAdvancedBMP",
            display_name="Save Image Advanced (BMP)",
            category=define.author,
            description="Saves images to disk as BMP files.",
            inputs=[
                io.Image.Input(
                    id="images",
                    display_name="Images",
                    tooltip="The images to be saved."
                ),
                io.Mask.Input(
                    id="masks",
                    display_name="Alpha Masks",
                    tooltip="Optional alpha channel masks to clip the saved images. If provided, these masks will be applied to the corresponding images before saving.",
                    optional=True
                ),
                io.Combo.Input(
                    id="bit_depth",
                    display_name="Bit Depth",
                    tooltip="The bit depth to use when saving images as BMP.\n32bit supports alpha channel.",
                    options=["24bit", "32bit"],
                    default="24bit"
                ),
                io.Boolean.Input(
                    id="join_alpha",
                    display_name="Join Alpha Channel",
                    tooltip="Clip the image with the provided mask.\nPlease note this only works on 32bit bit depth.",
                    default=False
                ),
                io.Boolean.Input(
                    id="invert_alpha",
                    display_name="Invert Alpha",
                    tooltip="Whether to invert the alpha channel before saving.",
                    default=False
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="The prefix to use for the filename.",
                    default=define.default_file_name
                ),
                io.Int.Input(
                    id="dpi",
                    display_name="DPI",
                    tooltip="The DPI to use when saving images.",
                    default=define.printing_dpi,
                    min=1,
                    max=600,
                    display_mode=io.NumberDisplay.slider
                ),
                io.Boolean.Input(
                    id="save_to_input_folder",
                    display_name="Save to Input Folder",
                    tooltip="Whether to sync the image to the input folder.",
                    default=False
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )
    
    @classmethod
    def execute(cls,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
        bit_depth: str = "24bit",
        join_alpha: bool = False,
        invert_alpha: bool = False,
        filename_prefix: str = define.default_file_name,
        dpi: int = define.printing_dpi,
        save_to_input_folder: bool = False,
        **kwargs
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0])
        f_input_folder = os.path.join(folder_paths.get_input_directory(), subfolder)

        results = []
        for (batch_number, image) in enumerate(images):
            mode = "RGBA" if bit_depth == "32bit" else "RGB"
            save_options = {
                "dpi": (dpi, dpi),
            }

            mask = None
            if (bit_depth == "32bit") and join_alpha and masks is not None:
                mask_index = min(batch_number, masks.shape[0] - 1)
                mask = masks[mask_index]
                if invert_alpha:
                    mask = 1.0 - mask

            result = ImageSaveHelperExt.get_save_result(
                image=image, mask=mask, convert_mode=mode, join_mask=join_alpha, filename=filename,
                full_output_folder=f_output_folder, full_input_folder=f_input_folder, subfolder=subfolder,
                batch_number=batch_number, counter=c,
                file_ext="bmp", save_to_input_folder=save_to_input_folder,
                save_kwargs=save_options
            )

            results.append(result)
            c += 1

        return io.NodeOutput(ui=ui.SavedImages(results))


class SaveImageTIFF(io.ComfyNode):
    """
    Saves images to disk as TIFF files.
    """
    output_dir = folder_paths.get_output_directory()
    prefix_append = ""

    debug_header = "[ComfyUI-SaveImageTIFF]"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageTIFF",
            display_name="Save Image (TIFF)",
            category=define.author,
            description="Saves images to disk as TIFF files.",
            inputs=[
                io.Image.Input(
                    id="images",
                    display_name="Images",
                    tooltip="The images to be saved."
                ),
                io.Mask.Input(
                    id="masks",
                    display_name="Alpha Masks",
                    tooltip="Optional alpha channel masks to clip the saved images. If provided, these masks will be applied to the corresponding images before saving.",
                    optional=True
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="Prefix for the saved image filenames.\nEach image will be saved as {prefix}_{index}.{format}.",
                    default=define.default_file_name
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
        filename_prefix: str = define.default_file_name,
        **kwargs
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0])
        
        preview_results = []
        for (batch_number, image) in enumerate(images):
            mode = "RGB"
            mask = None
            if masks is not None:
                mask_index = min(batch_number, masks.shape[0] - 1)
                mask = masks[mask_index]
                mode = "RGBA"
            should_join_mask = mask is not None

            # Drop the result
            ImageSaveHelperExt.get_save_result(
                image=image, mask=mask, convert_mode=mode, join_mask=should_join_mask, filename=filename,
                full_output_folder=f_output_folder, full_input_folder=None, subfolder=subfolder,
                batch_number=batch_number, counter=c, file_ext="tiff", save_to_input_folder=False,
                save_kwargs={"compression": "none", "dpi": (define.screen_dpi, define.screen_dpi)}
            )

            preview_results.append(ImageSaveHelperExt.get_save_result_temp(image, mask))

            c += 1

        return io.NodeOutput(ui=ui.SavedImages(preview_results))

class SaveImageAdvancedTIFF(io.ComfyNode):
    """
    Saves images to disk as TIFF files with additional options.
    """
    output_dir = folder_paths.get_output_directory()
    prefix_append = ""

    debug_header = "[ComfyUI-SaveImageAdvancedTIFF]"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageAdvancedTIFF",
            display_name="Save Image Advanced (TIFF)",
            category=define.author,
            description="Saves images to disk as TIFF files.",
            inputs=[
                io.Image.Input(
                    id="images",
                    display_name="Images",
                    tooltip="The images to be saved."
                ),
                io.Mask.Input(
                    id="masks",
                    display_name="Alpha Masks",
                    tooltip="Optional alpha channel masks to clip the saved images. If provided, these masks will be applied to the corresponding images before saving.",
                    optional=True
                ),
                io.Boolean.Input(
                    id="disable_metadata",
                    display_name="Disable Metadata",
                    tooltip="Disable embedding EXIF data into the saved images.\nEmbed into `UserComment`.",
                    default=False
                ),
                io.Boolean.Input(
                    id="join_alpha",
                    display_name="Join Alpha Channel",
                    tooltip="Clip the image with the provided mask.\nPlease note this does not work on JPG format.",
                    default=False
                ),
                io.Boolean.Input(
                    id="invert_alpha",
                    display_name="Invert Alpha",
                    tooltip="Whether to invert the alpha channel before saving.",
                    default=False
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="The prefix to use for the filename.",
                    default=define.default_file_name
                ),
                io.Int.Input(
                    id="dpi",
                    display_name="DPI",
                    tooltip="The DPI to use when saving images.",
                    default=define.printing_dpi,
                    min=1,
                    max=600,
                    display_mode=io.NumberDisplay.slider
                ),
                io.Combo.Input(
                    id="tiff_compression",
                    display_name="Compression",
                    tooltip="The compression to use when saving images as TIFF.",
                    options=["none", "tiff_lzw", "tiff_deflate", "tiff_adobe_deflate", "packbits", "jpeg", "tiff_jpeg", "tiff_ccitt"],
                    default="none"
                ),
                io.Boolean.Input(
                    id="save_to_input_folder",
                    display_name="Save to Input Folder",
                    tooltip="Whether to sync the image to the input folder.",
                    default=False
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )
    
    @classmethod
    def execute(cls,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
        filename_prefix: str = define.default_file_name,
        disable_metadata: bool = False,
        join_alpha: bool = False,
        invert_alpha: bool = False,
        tiff_compression: str = "none",
        dpi: int = define.printing_dpi,
        save_to_input_folder: bool = False,
        **kwargs
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0])
        f_input_folder = os.path.join(folder_paths.get_input_directory(), subfolder)

        preview_results = []
        for (batch_number, image) in enumerate(images):
            mode = "RGBA"
            save_options = {
                "compression": tiff_compression,
                "dpi": (dpi, dpi),
            }

            if not disable_metadata:
                tiffinfo = ImageSaveHelperExt.create_metadata_tiff(cls)
                if tiffinfo is not None:
                    save_options["tiffinfo"] = tiffinfo

            if tiff_compression == "tiff_ccitt":
                print(f"{cls.debug_header} tiff_ccitt compression requires a bilevel (1-bit) image.")
                mode = "1"
                
            mask = None
            if join_alpha:
                mask_index = min(batch_number, masks.shape[0] - 1)
                mask = masks[mask_index]
                if invert_alpha:
                    mask = 1.0 - mask

            ImageSaveHelperExt.get_save_result(
                image=image, mask=mask, convert_mode=mode, join_mask=join_alpha, filename=filename,
                full_output_folder=f_output_folder, full_input_folder=f_input_folder, subfolder=subfolder,
                batch_number=batch_number, counter=c,
                file_ext="tiff", save_to_input_folder=save_to_input_folder,
                save_kwargs=save_options
            )

            preview_results.append(ImageSaveHelperExt.get_save_result_temp(image, mask))

            c += 1

        return io.NodeOutput(ui=ui.SavedImages(preview_results))


class SaveImageTGA(io.ComfyNode):
    """
    Saves images to disk as TGA files.
    """
    output_dir = folder_paths.get_output_directory()
    prefix_append = ""

    debug_header = "[ComfyUI-SaveImageTGA]"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageTGA",
            display_name="Save Image (TGA)",
            category=define.author,
            description="Saves images to disk as TGA files.",
            inputs=[
                io.Image.Input(
                    id="images",
                    display_name="Images",
                    tooltip="The images to be saved."
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="Prefix for the saved image filenames.\nEach image will be saved as {prefix}_{index}.{format}.",
                    default=define.default_file_name
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        filename_prefix: str = define.default_file_name,
        **kwargs
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0])
        
        preview_results = []
        for (batch_number, image) in enumerate(images):
            ImageSaveHelperExt.get_save_result(
                image=image, mask=None, convert_mode="RGB", join_mask=False, filename=filename,
                full_output_folder=f_output_folder, full_input_folder=None, subfolder=subfolder,
                batch_number=batch_number, counter=c, file_ext="tga", save_to_input_folder=False,
                save_kwargs={"compression": "tga_rle"}
            )

            preview_results.append(ImageSaveHelperExt.get_save_result_temp(image))

            c += 1

        return io.NodeOutput(ui=ui.SavedImages(preview_results))

class SaveImageAdvancedTGA(io.ComfyNode):
    """
    Saves images to disk as TGA files with additional options.
    """
    output_dir = folder_paths.get_output_directory()
    prefix_append = ""

    debug_header = "[ComfyUI-SaveImageAdvancedTGA]"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageAdvancedTGA",
            display_name="Save Image Advanced (TGA)",
            category=define.author,
            description="Saves images to disk as TGA files.",
            inputs=[
                io.Image.Input(
                    id="images",
                    display_name="Images",
                    tooltip="The images to be saved."
                ),
                io.Mask.Input(
                    id="masks",
                    display_name="Alpha Masks",
                    tooltip="Optional alpha channel masks to clip the saved images. If provided, these masks will be applied to the corresponding images before saving.",
                    optional=True
                ),
                io.Boolean.Input(
                    id="rle",
                    display_name="RLE Compression",
                    tooltip="Whether to use RLE compression when saving as TGA.",
                    default=True
                ),
                io.Boolean.Input(
                    id="join_alpha",
                    display_name="Join Alpha Channel",
                    tooltip="Clip the image with the provided mask.",
                    default=False
                ),
                io.Boolean.Input(
                    id="invert_alpha",
                    display_name="Invert Alpha",
                    tooltip="Whether to invert the alpha channel before saving.",
                    default=False
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="The prefix to use for the filename.",
                    default=define.default_file_name
                ),
                io.Boolean.Input(
                    id="save_to_input_folder",
                    display_name="Save to Input Folder",
                    tooltip="Whether to sync the image to the input folder.",
                    default=False
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )
    
    @classmethod
    def execute(cls,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
        filename_prefix: str = define.default_file_name,
        rle: bool = True,
        join_alpha: bool = False,
        invert_alpha: bool = False,
        save_to_input_folder: bool = False,
        **kwargs
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0])
        f_input_folder = os.path.join(folder_paths.get_input_directory(), subfolder)

        preview_results = []
        for (batch_number, image) in enumerate(images):
            mode = "RGBA" if join_alpha else "RGB"
            save_options = {
                "compression": "tga_rle" if rle else None,
            }
                
            mask = None
            if join_alpha and masks is not None:
                mask_index = min(batch_number, masks.shape[0] - 1)
                mask = masks[mask_index]
                if invert_alpha:
                    mask = 1.0 - mask

            ImageSaveHelperExt.get_save_result(
                image=image, mask=mask, convert_mode=mode, join_mask=join_alpha, filename=filename,
                full_output_folder=f_output_folder, full_input_folder=f_input_folder, subfolder=subfolder,
                batch_number=batch_number, counter=c,
                file_ext="tga", save_to_input_folder=save_to_input_folder,
                save_kwargs=save_options
            )

            preview_results.append(ImageSaveHelperExt.get_save_result_temp(image, mask))

            c += 1

        return io.NodeOutput(ui=ui.SavedImages(preview_results))

NODE_CLASS_MAPPINGS = {
    "ColorArtifactNormalizer": ColorArtifactNormalizer,
    "LoadICCProfile": LoadICCProfile,
    "SaveImageAdvanced": SaveImageAdvanced,
    "SaveImageJPG": SaveImageJPG,
    "SaveImageAdvancedJPG": SaveImageAdvancedJPG,
    "SaveImageBMP": SaveImageBMP,
    "SaveImageAdvancedBMP": SaveImageAdvancedBMP,
    "SaveImageTIFF": SaveImageTIFF,
    "SaveImageAdvancedTIFF": SaveImageAdvancedTIFF,
    "SaveImageTGA": SaveImageTGA,
    "SaveImageAdvancedTGA": SaveImageAdvancedTGA
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorArtifactNormalizer": "Color Artifact Normalizer",
    "LoadICCProfile": "Load ICC Profile",
    "SaveImageAdvanced": "Save Image Advanced",
    "SaveImageJPG": "Save Image (JPG)",
    "SaveImageAdvancedJPG": "Save Image Advanced (JPG)",
    "SaveImageBMP": "Save Image (BMP)",
    "SaveImageAdvancedBMP": "Save Image Advanced (BMP)",
    "SaveImageTIFF": "Save Image (TIFF)",
    "SaveImageAdvancedTIFF": "Save Image Advanced (TIFF)",
    "SaveImageTGA": "Save Image (TGA)",
    "SaveImageAdvancedTGA": "Save Image Advanced (TGA)",
}