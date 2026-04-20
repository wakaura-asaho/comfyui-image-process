from comfy_api.latest import io, ui
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import torch
import folder_paths
import os
import json

class ColorArtifactNormalizer(io.ComfyNode):
    """
    Fix achromatic color instability from AI-generated images.

    Pixels with near-zero saturation have their hue snapped to 0
    and their saturation zeroed out, then optionally smoothed.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ColorArtifactNormalizer",
            display_name="Color Artifact Normalizer",
            category="Wakaura",
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
                ),
                io.Boolean.Input(
                    id="smooth",
                    display_name="Smooth Transitions",
                    tooltip="Whether to apply a smoothing filter to the corrected areas.",
                    default=True
                ),
                io.Float.Input(
                    id="threshold",
                    display_name="Saturation Threshold",
                    tooltip="The saturation level below which pixels are considered artifacts.",
                    step=0.01,
                    round=0.01,
                    default=0.08,
                    max=1.0,
                    min=0.0,
                ),
                io.Int.Input(
                    id="kernal_size",
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
        threshold = float(kwargs.get("threshold", 0.08))
        kernel_size = int(kwargs.get("kernal_size", 3))
        mask = kwargs.get("mask")
        mask_id = id(mask) if mask is not None else 0
        return hash((preserve_alpha, smooth, threshold, kernel_size, mask_id)) & 0xFFFFFFFF

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
                "[ComfyUI-ColorArtifactNormalizer] Warning: preserve_alpha requested but source image has no alpha channel. "
                "   Output will be generated as an opaque image."
            )
        return True

    @classmethod
    def execute(cls, image, **kwargs) -> io.NodeOutput:
        preserve_alpha = bool(kwargs.get("preserve_alpha", False))
        smooth_transitions = bool(kwargs.get("smooth", False))
        sat_threshold = float(kwargs.get("threshold", 0.08))
        kernel_size = int(kwargs.get("kernal_size", 3))
        mask_input = kwargs.get("mask")

        if not isinstance(image, Image.Image):
            try:
                if isinstance(image, torch.Tensor):
                    image = image.cpu().numpy()
            except ImportError:
                pass
            
            if isinstance(image, np.ndarray):
                # Ensure single image (remove batch dimension if present)
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
        if preserve_alpha and mask_input is not None:
            # Convert mask input to numpy array
            if isinstance(mask_input, torch.Tensor):
                alpha = mask_input.cpu().numpy()
            elif isinstance(mask_input, np.ndarray):
                alpha = mask_input
            else:
                alpha = np.array(mask_input, dtype=np.float32)
            
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

        if smooth_transitions:
            try:
                from scipy.ndimage import uniform_filter
                smoothed = uniform_filter(rgb, size=[kernel_size, kernel_size, 1])
                mask3 = np.stack([neutral_mask] * 3, axis=-1)
                rgb = np.where(mask3, smoothed, rgb)
            except Exception as exc:
                print(f"[ComfyUI-ColorArtifactNormalizer] Warning: smoothing failed ({exc}). Output will use unsmoothed correction.")

        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        if alpha is not None:
            alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)
            result_array = np.concatenate([rgb, alpha[..., np.newaxis]], axis=-1)
            output_image = Image.fromarray(result_array, mode="RGBA")
            mask_image = Image.fromarray(alpha, mode="L")
            print("[ComfyUI-ColorArtifactNormalizer] Alpha channel preserved in output image.")
            mask_tensor = torch.from_numpy(np.array(mask_image, dtype=np.float32) / 255.0).unsqueeze(0)
        else:
            output_image = Image.fromarray(rgb)
            print("No alpha channel preserved; output image is opaque.")
            mask_tensor = torch.zeros((1, rgb.shape[0], rgb.shape[1]), dtype=torch.float32)

        output_tensor = torch.from_numpy(np.array(output_image, dtype=np.float32) / 255.0).unsqueeze(0)
        
        return io.NodeOutput(output_tensor, mask_tensor)

class SaveImageAdvanced(io.ComfyNode):
    """
    Saves images to disk with additional options for metadata and compression. Useful for saving AI-generated images with or without EXIF data, and for controlling compression levels.
    """
    output_dir = folder_paths.get_output_directory()
    prefix_append = ""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageAdvanced",
            display_name="Save Image Advanced",
            category="Wakaura",
            description="Saves image to disk with additional options for metadata and compression. Useful for saving AI-generated images with or without EXIF data, and for controlling compression levels.",
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
                    tooltip="If true, no metadata will be saved with the image. If false, prompt and extra_pnginfo from the hidden state will be saved as text chunks.",
                    default=True
                ),
                io.Boolean.Input(
                    id="join_alpha",
                    display_name="Join Alpha Channel",
                    tooltip="If true, the alpha masks will be joined with the images as an alpha channel before saving. If false, the alpha masks will be ignored and not included in the saved images.",
                    default=False
                ),
                io.Int.Input(
                    id="compress_level",
                    display_name="Compression Level",
                    tooltip="The compression level to use when saving images (0-9). Higher values result in smaller file sizes but longer save times.",
                    default=4,
                    min=0,
                    max=9
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="Prefix for the saved image filenames. Each image will be saved as {prefix}_{index}.png.",
                    default="ComfyUI"
                )
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images: torch.Tensor, masks: torch.Tensor | None = None, filename_prefix="ComfyUI", disable_metadata=True, join_alpha=False, compress_level=4, **kwargs) -> io.NodeOutput:
        filename_prefix += cls.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0])

        results = []
        metadata = not disable_metadata and ui.ImageSaveHelper._create_png_metadata(cls) or None
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            if join_alpha and masks is not None:
                # Clamp batch index so a single mask broadcasts across all images
                mask_index = min(batch_number, masks.shape[0] - 1)
                mask = masks[mask_index].cpu().numpy()   # shape: (H, W), range [0.0, 1.0]

                # Resize mask to match the image dimensions if they differ
                if mask.shape != (img.height, img.width):
                    mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
                    mask_pil = mask_pil.resize((img.width, img.height), Image.LANCZOS)
                else:
                    mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")

                img = img.convert("RGBA")
                img.putalpha(mask_pil)

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=compress_level)
            results.append(ui.SavedResult(file, subfolder, io.FolderType.output))
            counter += 1

        return io.NodeOutput(ui=ui.SavedImages(results))

NODE_CLASS_MAPPINGS = {
    "ColorArtifactNormalizer": ColorArtifactNormalizer,
    "SaveImageAdvanced": SaveImageAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorArtifactNormalizer": "Color Artifact Normalizer",
    "SaveImageAdvanced": "Save Image Advanced",
}