from comfy_api.latest import io, ui
import numpy as np
import logging
import time
import scipy.ndimage as nd
from PIL import Image, ImageCms
from skimage.color import rgb2lab, rgb2hsv, hsv2rgb
import torch
import folder_paths
import os
from .image_helper import ImageSaveHelperExt
from .define import define


logger = logging.getLogger("ComfyUI-ImageProcess")


class ColorPatchFlatten(io.ComfyNode):
    """
    Flatten HSV values across a color patch within a given tolerance.
    """

    debug_header = "[ComfyUI-ColorPatchFlatten]"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ColorPatchFlatten",
            display_name="Color Patch Flatten",
            category=define.author,
            is_experimental=True,
            description="Flatten HSV values across a color patch within a given tolerance.",
            inputs=[
                io.Image.Input(
                    id="image",
                    display_name="Image",
                    tooltip="The image to be processed.",
                ),
                io.Mask.Input(
                    id="mask",
                    display_name="Alpha Mask",
                    tooltip="Optional alpha channel mask.",
                    optional=True,
                ),
                io.Float.Input(
                    id="flatten_tolerance",
                    display_name="Flatten Tolerance",
                    tooltip="How different colors can be conisdered as a same color patch.\nLarger number means weaker flattening effect.",
                    step=0.01,
                    round=0.01,
                    default=0.05,
                    max=1.0,
                    min=0.00,
                ),
                io.Boolean.Input(id="flatten_hue", display_name="H", default=True),
                io.Boolean.Input(
                    id="flatten_saturation", display_name="S", default=True
                ),
                io.Boolean.Input(
                    id="flatten_brightness", display_name="V", default=True
                ),
            ],
            outputs=[
                io.Image.Output(id="output", display_name="OUTPUT"),
                io.Mask.Output(id="mask", display_name="ALPHA"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor | None = None,
        flatten_tolerance: float = 0.05,
        flatten_hue: bool = True,
        flatten_saturation: bool = True,
        flatten_brightness: bool = True,
        **kwargs,
    ) -> io.NodeOutput:
        pil_images = ImageSaveHelperExt.to_pillow_images(image)

        alpha_raw_batch = None
        if mask is not None:
            alpha_raw_batch = mask.cpu().numpy()

        output_images = []
        output_masks = []
        for i, pil_image in enumerate(pil_images):
            pil_image = pil_image.convert("RGB")

            alpha_f32 = None
            if alpha_raw_batch is not None:
                if alpha_raw_batch.ndim == 3:
                    alpha_idx = min(i, alpha_raw_batch.shape[0] - 1)
                    current_alpha_raw = alpha_raw_batch[alpha_idx]
                else:
                    current_alpha_raw = alpha_raw_batch

                if current_alpha_raw.max() > 1.0:
                    alpha_f32 = np.clip(current_alpha_raw / 255, 0, 1).astype(
                        np.float32
                    )
                else:
                    alpha_f32 = current_alpha_raw.astype(np.float32)

            rgb_np = np.array(pil_image, dtype=np.float32) / 255.0  # (H, W, 3)

            if flatten_tolerance > 0 and (
                flatten_hue or flatten_saturation or flatten_brightness
            ):
                hsv_np = rgb2hsv(rgb_np)

                # Determine patch for mean calculation
                if alpha_f32 is not None:
                    if alpha_f32.shape[:2] != rgb_np.shape[:2]:
                        mask_pil = Image.fromarray(
                            (alpha_f32 * 255).clip(0, 255).astype(np.uint8), mode="L"
                        )
                        mask_pil = mask_pil.resize(
                            (rgb_np.shape[1], rgb_np.shape[0]), Image.LANCZOS
                        )
                        alpha_f32 = np.array(mask_pil, dtype=np.float32) / 255.0

                    mask_indices = alpha_f32 > 0.05
                    if np.any(mask_indices):
                        mean_hsv = np.mean(hsv_np[mask_indices], axis=0)
                    else:
                        mean_hsv = np.mean(hsv_np.reshape(-1, 3), axis=0)
                else:
                    mean_hsv = np.mean(hsv_np.reshape(-1, 3), axis=0)

                # Flatten
                if flatten_hue:
                    h_diff = np.abs(hsv_np[..., 0] - mean_hsv[0])
                    # Circular hue diff
                    h_diff = np.minimum(h_diff, 1.0 - h_diff)
                    hsv_np[..., 0] = np.where(
                        h_diff <= flatten_tolerance, mean_hsv[0], hsv_np[..., 0]
                    )

                if flatten_saturation:
                    s_diff = np.abs(hsv_np[..., 1] - mean_hsv[1])
                    hsv_np[..., 1] = np.where(
                        s_diff <= flatten_tolerance, mean_hsv[1], hsv_np[..., 1]
                    )

                if flatten_brightness:
                    v_diff = np.abs(hsv_np[..., 2] - mean_hsv[2])
                    hsv_np[..., 2] = np.where(
                        v_diff <= flatten_tolerance, mean_hsv[2], hsv_np[..., 2]
                    )

                rgb_np = hsv2rgb(hsv_np)

            rgb_u8 = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)

            if alpha_f32 is not None:
                alpha_u8 = np.clip(alpha_f32 * 255, 0, 255).astype(np.uint8)
                rgba_array = np.concatenate(
                    [rgb_u8, alpha_u8[..., np.newaxis]], axis=-1
                )
                output_image_pil = Image.fromarray(rgba_array, mode="RGBA")
                output_mask_tensor = torch.from_numpy(alpha_f32)
            else:
                output_image_pil = Image.fromarray(rgb_u8)
                output_mask_tensor = torch.zeros(
                    (rgb_u8.shape[0], rgb_u8.shape[1]), dtype=torch.float32
                )

            output_tensor = torch.from_numpy(
                np.array(output_image_pil, dtype=np.float32) / 255.0
            )
            output_images.append(output_tensor)
            output_masks.append(output_mask_tensor)

        output_images_batch = torch.stack(output_images, dim=0)
        output_masks_batch = torch.stack(output_masks, dim=0)

        return io.NodeOutput(output_images_batch, output_masks_batch)


class ColorPatchMerge(io.ComfyNode):
    """
    Groups similar colors and replaces them with their local average.
    """

    debug_header = "[ComfyUI-ColorPatchMerge]"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ColorPatchMerge",
            display_name="Color Patch Merge",
            category=define.author,
            is_experimental=True,
            description="Groups similar colors and replaces them with their local averages.",
            inputs=[
                io.Image.Input(
                    id="image",
                    display_name="Image",
                    tooltip="The image to be processed.",
                ),
                io.Mask.Input(
                    id="mask",
                    display_name="Alpha Mask",
                    tooltip="Optional alpha channel mask.",
                    optional=True,
                ),
                io.Float.Input(
                    id="merge_tolerance",
                    display_name="Merge Tolerance",
                    tooltip="How different colors can be to stay in the same group.",
                    step=0.01,
                    round=0.01,
                    default=0.05,
                    max=1.0,
                    min=0.00,
                ),
                io.Int.Input(
                    id="neighborhood",
                    display_name="Neighborhood",
                    tooltip="The neighborhood size (diameter) for the bilateral filter.\nUsed for 'Smooth' merging solution.",
                    step=1,
                    default=3,
                    max=31,
                    min=3,
                ),
                io.Int.Input(
                    id="min_area",
                    display_name="Minimal Area",
                    tooltip="The minimum area size for color patches.\nUsed for 'Unify' merging solution.",
                    step=1,
                    default=20,
                    max=1024,
                    min=2,
                ),
                io.Int.Input(
                    id="iterations",
                    display_name="Iterations",
                    tooltip="Number of growth iterations for color patches.\nMore iterations result in larger, more uniform color regions.\nUsed for 'Unify' merging solution.",
                    step=1,
                    default=2,
                    max=16,
                    min=1,
                ),
                io.Boolean.Input(
                    id="use_lab",
                    display_name="Use Lab Colors",
                    tooltip="Convert RGB space to Lab colors.",
                    default=True,
                ),
                io.Combo.Input(
                    id="merge_solution",
                    display_name="Merge Solution",
                    tooltip="How to handle the color merging.",
                    options=["Smooth", "Unify"],
                    default="Smooth",
                ),
            ],
            outputs=[
                io.Image.Output(id="output", display_name="OUTPUT"),
                io.Mask.Output(id="mask", display_name="ALPHA"),
            ],
        )

    @classmethod
    def unify_colors(
        cls,
        rgb_np: np.ndarray,
        tolerance: float = 8.0,
        min_area: int = 20,
        iterations: int = 2,
        use_lab: bool = True,
    ) -> np.ndarray:
        orig = rgb_np.astype(np.float32)

        if use_lab:
            data_for_quant = rgb2lab(orig)
        else:
            data_for_quant = orig * 255.0

        # Coarser quantization
        q = (data_for_quant / tolerance).astype(np.int32)

        if use_lab:
            flat_colors = q[..., 0] + q[..., 1] * 1000 + q[..., 2] * 1000000
        else:
            flat_colors = q[..., 0] + q[..., 1] * 300 + q[..., 2] * 90000

        # Cap the growth footprint size to avoid performance issues
        footprint_size = min(min_area, 32)
        structure = np.ones((footprint_size, footprint_size), dtype=np.int32)
        connected = flat_colors.copy()

        for _ in range(iterations):
            dilated = nd.grey_dilation(connected, footprint=structure)
            connected = np.where(
                dilated == flat_colors, flat_colors, dilated
            )  # Grow regions

        labels, num_features = nd.label(
            connected, structure=np.ones((3, 3), dtype=np.int32)
        )

        index = np.arange(1, num_features + 1)
        mean_r = nd.mean(orig[..., 0], labels, index)
        mean_g = nd.mean(orig[..., 1], labels, index)
        mean_b = nd.mean(orig[..., 2], labels, index)
        means = np.column_stack((mean_r, mean_g, mean_b))

        lookup_table = np.vstack(([0, 0, 0], means))
        flattened = lookup_table[labels]

        # protect small areas or high-gradient (edge) pixels
        if min_area > 1:
            counts = nd.sum(np.ones_like(labels), labels, index)

            lookup_counts = np.zeros(num_features + 1, dtype=counts.dtype)
            lookup_counts[1:] = counts

            large_mask = lookup_counts[labels] >= min_area
            large_mask = large_mask[..., None]

            # high local variance areas keep original
            gray = np.dot(orig[..., :3], [0.299, 0.587, 0.114])
            local_var = nd.generic_filter(gray, np.var, size=3)
            edge_mask = local_var > np.percentile(
                local_var, 85
            )  # Top ~15% variance = edges

            result = np.where(large_mask & ~edge_mask[..., None], flattened, orig)
        else:
            result = flattened

        return result

    @classmethod
    def smooth_colors(
        cls, rgb_np: np.ndarray, tolerance: float = 0.05, neighborhood: int = 3
    ):
        import cv2

        simplified = cv2.bilateralFilter(
            (rgb_np * 255).astype(np.uint8),
            d=neighborhood,
            sigmaColor=tolerance * 255,
            sigmaSpace=neighborhood * 2,
        )
        return simplified.astype(np.float32) / 255.0

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor | None = None,
        merge_tolerance: float = 0.01,
        neighborhood: int = 3,
        min_area: int = 3,
        iterations: int = 2,
        use_lab: bool = True,
        merge_solution: str = "Smooth",
        **kwargs,
    ) -> io.NodeOutput:
        pil_images = ImageSaveHelperExt.to_pillow_images(image)

        alpha_raw_batch = None
        if mask is not None:
            alpha_raw_batch = mask.cpu().numpy()

        output_images = []
        output_masks = []
        for i, pil_image in enumerate(pil_images):
            pil_image = pil_image.convert("RGB")

            alpha_f32 = None
            if alpha_raw_batch is not None:
                if alpha_raw_batch.ndim == 3:
                    alpha_idx = min(i, alpha_raw_batch.shape[0] - 1)
                    current_alpha_raw = alpha_raw_batch[alpha_idx]
                else:
                    current_alpha_raw = alpha_raw_batch
                if current_alpha_raw.max() > 1.0:
                    alpha_f32 = np.clip(current_alpha_raw / 255, 0, 1).astype(
                        np.float32
                    )
                else:
                    alpha_f32 = current_alpha_raw.astype(np.float32)

            rgb_np = np.array(pil_image, dtype=np.float32) / 255.0  # (H, W, 3)

            if merge_tolerance > 0:
                if merge_solution == "Smooth" and neighborhood > 0:
                    rgb_np = cls.smooth_colors(rgb_np, merge_tolerance, neighborhood)
                elif merge_solution == "Unify" and min_area > 0:
                    # Scale tolerance appropriately for the unify_colors method
                    # (default merge_tolerance is 0.05, so * 255 gives ~12.75, which
                    # is reasonable for LAB 0-100 or RGB 0-255 ranges).
                    rgb_np = cls.unify_colors(
                        rgb_np, merge_tolerance * 255, min_area, iterations, use_lab
                    )
            else:
                logger.info(
                    f"{cls.debug_header} Tolerance is set to zero. The process will be skipped."
                )

            rgb_u8 = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)

            if alpha_f32 is not None:
                # Resize alpha to match image if needed
                if alpha_f32.shape[:2] != rgb_u8.shape[:2]:
                    mask_pil = Image.fromarray(
                        (alpha_f32 * 255).clip(0, 255).astype(np.uint8), mode="L"
                    )
                    mask_pil = mask_pil.resize(
                        (rgb_u8.shape[1], rgb_u8.shape[0]), Image.LANCZOS
                    )
                    alpha_f32 = np.array(mask_pil, dtype=np.float32) / 255.0

                alpha_u8 = np.clip(alpha_f32 * 255, 0, 255).astype(np.uint8)
                rgba_array = np.concatenate(
                    [rgb_u8, alpha_u8[..., np.newaxis]], axis=-1
                )
                output_image_pil = Image.fromarray(rgba_array, mode="RGBA")
                output_mask_tensor = torch.from_numpy(alpha_f32)
            else:
                output_image_pil = Image.fromarray(rgb_u8)
                output_mask_tensor = torch.zeros(
                    (rgb_u8.shape[0], rgb_u8.shape[1]), dtype=torch.float32
                )

            output_tensor = torch.from_numpy(
                np.array(output_image_pil, dtype=np.float32) / 255.0
            )
            output_images.append(output_tensor)
            output_masks.append(output_mask_tensor)

        output_images_batch = torch.stack(output_images, dim=0)
        output_masks_batch = torch.stack(output_masks, dim=0)

        return io.NodeOutput(output_images_batch, output_masks_batch)


class AchromaticStabilizer(io.ComfyNode):
    """
    Fix achromatic color instability from AI-generated images.

    Pixels with near-zero saturation have their hue snapped to 0
    and their saturation zeroed out, then optionally smoothed.
    """

    debug_header = "[ComfyUI-AchromaticStabilizer]"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AchromaticStabilizer",
            display_name="Achromatic Stabilizer",
            category=define.author,
            is_experimental=True,
            description="Fix achromatic color instability across the image."
            "Pixels with near-zero saturation have their hue snapped to 0 and their saturation zeroed out, then optionally smoothed.",
            inputs=[
                io.Image.Input(
                    id="image",
                    display_name="Image",
                    tooltip="The image to be processed.",
                ),
                io.Mask.Input(
                    id="mask",
                    display_name="Alpha Mask",
                    tooltip="Optional alpha channel mask.\nIf provided and preserve_alpha is True, this mask will be used as the alpha channel.",
                    optional=True,
                ),
                io.Boolean.Input(
                    id="smooth",
                    display_name="Smooth Transitions",
                    tooltip="Whether to apply a smoothing filter to the corrected areas.",
                    default=True,
                ),
                io.Float.Input(
                    id="sat_threshold",
                    display_name="Saturation Threshold",
                    tooltip="The saturation level below which pixels are considered artifacts.",
                    step=0.01,
                    round=0.01,
                    default=0.08,
                    max=1.0,
                    min=0.0,
                ),
                io.Int.Input(
                    id="kernel_size",
                    display_name="Smoothing Kernel Size",
                    tooltip="The size of the smoothing kernel to apply.",
                    default=3,
                    max=15,
                    min=1,
                ),
                io.Boolean.Input(
                    id="preserve_alpha",
                    display_name="Preserve Alpha",
                    tooltip="Whether to preserve the alpha channel during processing.",
                    default=True,
                ),
                io.Boolean.Input(
                    id="invert_alpha",
                    display_name="Invert Alpha",
                    tooltip="Whether to invert the alpha channel before saving.",
                    default=False,
                ),
            ],
            outputs=[
                io.Image.Output(id="output", display_name="OUTPUT"),
                io.Mask.Output(id="mask", display_name="ALPHA"),
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
        return (
            hash((preserve_alpha, smooth, sat_threshold, kernel_size, mask_id))
            & 0xFFFFFFFF
        )

    @classmethod
    def check_lazy_status(cls, *args, **kwargs) -> list[str]:
        """No alternate lazy input branches are exposed by this node."""
        return []

    @classmethod
    def validate_inputs(cls, *args, **kwargs) -> bool:
        preserve_alpha = bool(kwargs.get("preserve_alpha", False))
        image = kwargs.get("image")
        # Only validate if image is already a PIL Image
        if (
            preserve_alpha
            and isinstance(image, Image.Image)
            and image.mode not in ("RGBA", "LA", "P")
        ):
            logger.warning(
                f"{cls.debug_header} Warning: preserve_alpha requested but source image has no alpha channel. "
                f"   Output will be generated as an opaque image."
            )
        return True

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor | None = None,
        smooth: bool = True,
        sat_threshold: float = 0.08,
        kernel_size: int = 3,
        preserve_alpha: bool = True,
        invert_alpha: bool = False,
        **kwargs,
    ) -> io.NodeOutput:
        if kernel_size % 2 == 0:
            kernel_size += 1
            logger.warning(
                f"{cls.debug_header} kernel_size rounded up to {kernel_size} "
                f"(must be odd for a symmetric smoothing kernel)."
            )

        pil_images = ImageSaveHelperExt.to_pillow_images(image)

        alpha_raw_batch = None
        if preserve_alpha and mask is not None:
            alpha_raw_batch = mask.cpu().numpy()

        output_images = []
        output_masks = []
        alpha_preserved = False

        for i, pil_image in enumerate(pil_images):
            if preserve_alpha and pil_image.mode == "P":
                pil_image = pil_image.convert("RGBA")

            alpha_f32: np.ndarray | None = None
            if alpha_raw_batch is not None:
                # (B, H, W)
                if alpha_raw_batch.ndim == 3:
                    alpha_idx = min(i, alpha_raw_batch.shape[0] - 1)
                    current_alpha_raw = alpha_raw_batch[alpha_idx]
                else:
                    current_alpha_raw = alpha_raw_batch
                if current_alpha_raw.max() > 1.0:
                    alpha_f32 = np.clip(current_alpha_raw / 255, 0, 1).astype(
                        np.float32
                    )
                else:
                    alpha_f32 = current_alpha_raw.astype(np.float32)
                pil_image = pil_image.convert("RGB")
            elif preserve_alpha and pil_image.mode in ("RGBA", "LA"):
                alpha_f32 = np.array(pil_image.split()[-1], dtype=np.float32) / 255.0
                pil_image = pil_image.convert("RGB")
            else:
                pil_image = pil_image.convert("RGB")

            rgb = np.array(pil_image, dtype=np.float32) / 255  # (H, W, 3)

            c_max = rgb.max(axis=-1)
            c_min = rgb.min(axis=-1)
            chroma = c_max - c_min  # absolute chroma

            neutral_mask = chroma < sat_threshold

            rgb_original = rgb.copy()

            # Replace artifact pixels with their luminance grey.
            grey_value = rgb.mean(axis=-1, keepdims=True)  # (H, W, 1)
            rgb[neutral_mask] = np.repeat(grey_value, 3, axis=-1)[neutral_mask]

            if smooth:
                try:
                    smoothed_original = nd.uniform_filter(
                        rgb_original, size=[kernel_size, kernel_size, 1]
                    )
                    mask3 = np.stack([neutral_mask] * 3, axis=-1)
                    rgb = np.where(mask3, smoothed_original, rgb)
                except Exception as exc:
                    logger.warning(
                        f"{cls.debug_header} Warning: smoothing failed ({exc}). "
                        f"Output will use unsmoothed correction."
                    )

            rgb_u8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)  # (H, W, 3)
            if preserve_alpha and alpha_f32 is not None:
                alpha_preserved = True
                # Resize alpha to match image if needed.
                if alpha_f32.shape[:2] != rgb_u8.shape[:2]:
                    mask_pil = Image.fromarray(
                        (alpha_f32 * 255).clip(0, 255).astype(np.uint8), mode="L"
                    )
                    target_wh = (rgb_u8.shape[1], rgb_u8.shape[0])
                    mask_pil = mask_pil.resize(target_wh, Image.LANCZOS)
                    alpha_f32 = np.array(mask_pil, dtype=np.float32) / 255.0

                if invert_alpha:
                    alpha_f32 = 1.0 - alpha_f32

                alpha_u8 = np.clip(alpha_f32 * 255, 0, 255).astype(np.uint8)

                rgba_array = np.concatenate(
                    [rgb_u8, alpha_u8[..., np.newaxis]], axis=-1
                )
                output_image_pil = Image.fromarray(rgba_array, mode="RGBA")

                output_mask_tensor = torch.from_numpy(alpha_f32)
            else:
                output_image_pil = Image.fromarray(rgb_u8)
                output_mask_tensor = torch.zeros(
                    (rgb_u8.shape[0], rgb_u8.shape[1]), dtype=torch.float32
                )

            output_tensor = torch.from_numpy(
                np.array(output_image_pil, dtype=np.float32) / 255.0
            )

            output_images.append(output_tensor)
            output_masks.append(output_mask_tensor)

        if alpha_preserved:
            logger.info(f"{cls.debug_header} Alpha channel preserved in output image.")
        else:
            logger.info(
                f"{cls.debug_header} No alpha channel preserved; output image is opaque."
            )

        output_images_batch = torch.stack(output_images, dim=0)
        output_masks_batch = torch.stack(output_masks, dim=0)

        return io.NodeOutput(output_images_batch, output_masks_batch)


class LoadICCProfile(io.ComfyNode):
    """
    Loads an ICC color profile from the models/icc_profiles folder.
    """

    icc_folder = "icc_profiles"

    @classmethod
    def get_valid_icc_profiles(cls, color_profiles: list[str]) -> list[str]:
        if color_profiles is None or len(color_profiles) < 0:
            return []

        valid_profiles = []
        for cp in color_profiles:
            if cp.lower().endswith((".icc", ".icm")):
                try:
                    profile = ImageCms.getOpenProfile(
                        folder_paths.get_full_path(cls.icc_folder, cp)
                    )
                    if profile:
                        valid_profiles.append(cp)
                except (IOError, TypeError, Exception):
                    logger.warning(f"Skipping invalid profile: {cp}")

        return valid_profiles

    @staticmethod
    def get_icc_profile_info(profile: ImageCms.ImageCmsProfile) -> dict[str, str]:
        if profile and isinstance(profile, ImageCms.ImageCmsProfile):
            return {
                "model": profile.profile.model,
                "manufacturer": profile.profile.manufacturer,
                "description": profile.profile.profile_description,
                "copyright": profile.profile.copyright,
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
                    options=cls.get_valid_icc_profiles(
                        folder_paths.get_filename_list(cls.icc_folder)
                    ),
                )
            ],
            outputs=[
                io.Custom("ICC_PROFILE").Output(
                    id="icc_profile", display_name="ICC_PROFILE"
                ),
                io.String.Output(id="icc_profile_info", display_name="ICC_INFO"),
            ],
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


class SaveImageAdvancedCustom(io.ComfyNode):
    """
    Saves images to disk with additional options for metadata and compression.
    """

    output_dir = folder_paths.get_output_directory()
    prefix_append = ""

    debug_header = "[ComfyUI-SaveImageAdvanced]"

    web_unsupported_preview_formats = ["tiff", "tga"]
    alpha_supported_formats = ["png", "tiff", "webp", "bmp", "tga", "avif"]
    icc_unsupported_formats = ["tga", "bmp"]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageAdvancedCustom",
            display_name="Save Image Advanced",
            category=define.author,
            description="Saves image to disk with additional options for metadata and compression.",
            inputs=[
                io.Image.Input(
                    id="images",
                    display_name="Images",
                    tooltip="The images to be saved.",
                ),
                io.Mask.Input(
                    id="masks",
                    display_name="Alpha Masks",
                    tooltip="Optional alpha channel masks to clip the saved images. If provided, these masks will be applied to the corresponding images before saving.",
                    optional=True,
                ),
                io.Custom("ICC_PROFILE").Input(
                    id="icc_profile",
                    display_name="ICC Profile",
                    tooltip="Optional ICC color profile to embed into the saved images.",
                    optional=True,
                ),
                io.Boolean.Input(
                    id="disable_metadata",
                    display_name="Disable Metadata",
                    tooltip="Whether to include the metadata to the saved image.",
                    default=True,
                ),
                io.Boolean.Input(
                    id="join_alpha",
                    display_name="Join Alpha Channel",
                    tooltip="Clip the image with the provided mask.\nPlease note this does not work on JPG format.",
                    default=False,
                ),
                io.Boolean.Input(
                    id="invert_alpha",
                    display_name="Invert Alpha",
                    tooltip="Whether to invert the alpha channel before saving.",
                    default=False,
                ),
                io.Int.Input(
                    id="dpi",
                    display_name="DPI",
                    tooltip="The DPI to use when saving images.",
                    default=define.printing_dpi,
                    min=1,
                    max=600,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Int.Input(
                    id="compress_level",
                    display_name="Compression Level",
                    tooltip="The compression level to use when saving images as PNG (0-9).",
                    default=4,
                    min=0,
                    max=9,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Int.Input(
                    id="quality",
                    display_name="Quality",
                    tooltip="The quality to use when saving images as JPEG or WebP (1-100).",
                    default=90,
                    min=1,
                    max=100,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Combo.Input(
                    id="tiff_compression",
                    display_name="TIFF Compression",
                    tooltip="The compression to use when saving images as TIFF.",
                    options=[
                        "none",
                        "tiff_lzw",
                        "tiff_deflate",
                        "tiff_adobe_deflate",
                        "packbits",
                        "jpeg",
                        "tiff_jpeg",
                        "tiff_ccitt",
                    ],
                    default="none",
                ),
                io.Boolean.Input(
                    id="tga_rle",
                    display_name="TGA RLE Compression",
                    tooltip="Whether to use RLE compression when saving as TGA.",
                    default=True,
                ),
                io.Combo.Input(
                    id="format",
                    display_name="Format",
                    tooltip="The format to save the image in.",
                    options=["png", "jpg", "webp", "tiff", "bmp", "tga", "avif"],
                    default="png",
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    default=define.default_file_name,
                ),
                io.Boolean.Input(
                    id="save_to_input_folder",
                    display_name="Save to Input Folder",
                    tooltip="Whether to sync the image to the input folder.",
                    default=False,
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
        **kwargs,
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = (
            ImageSaveHelperExt.get_save_image_path(
                filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        f_input_folder = os.path.join(folder_paths.get_input_directory(), subfolder)

        results = []
        tmp_results = (
            []
        )  # For special file format that the web cannot display properly.
        metadata = (
            not disable_metadata and ImageSaveHelperExt.create_metadata_png(cls) or None
        )
        exif_data = (
            not disable_metadata
            and ImageSaveHelperExt.create_metadata_exif(cls)
            or None
        )
        for batch_number, image in enumerate(images):
            mode = "RGBA"
            save_options = {}
            if format == "jpg":
                mode = "RGB"
                save_options = {"quality": quality, "optimize": True, "dpi": (dpi, dpi)}
                if not disable_metadata:
                    save_options["exif"] = exif_data
            elif format == "webp":
                save_options = {
                    "quality": quality,
                    "lossless": False,
                    "dpi": (dpi, dpi),
                }
                if not disable_metadata:
                    save_options["exif"] = exif_data
            elif format == "tiff":
                # CCITT compression (Group 3/4 fax) only supports 1-bit bilevel
                # images. Convert to mode "1" to avoid "encoder error -2".
                if tiff_compression == "tiff_ccitt":
                    logger.warning(
                        f"{cls.debug_header} tiff_ccitt compression requires a bilevel (1-bit) image."
                    )
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
            elif format == "avif":
                mode = "RGBA" if join_alpha else "RGB"
                save_options = {
                    "quality": quality,
                    "speed": 6,
                    "dpi": (dpi, dpi),
                }
            else:  # PNG
                save_options = {
                    "pnginfo": metadata,
                    "compress_level": compress_level,
                    "dpi": (dpi, dpi),
                }

            if (format in cls.icc_unsupported_formats) and icc_profile is not None:
                logger.warning(
                    f"{cls.debug_header} ICC Profile discarded due to the incompatible format: {format.upper()}"
                )

            mask = None
            if (format in cls.alpha_supported_formats) and join_alpha:
                mask_index = min(batch_number, masks.shape[0] - 1)
                mask = masks[mask_index]
                if invert_alpha:
                    mask = 1.0 - mask

            result = ImageSaveHelperExt.get_save_result(
                image=image,
                mask=mask,
                convert_mode=mode,
                join_mask=join_alpha,
                filename=filename,
                full_output_folder=f_output_folder,
                full_input_folder=f_input_folder,
                subfolder=subfolder,
                batch_number=batch_number,
                counter=c,
                file_ext=format,
                save_to_input_folder=save_to_input_folder,
                save_kwargs=save_options,
                icc_profile=icc_profile,
            )

            if format in cls.web_unsupported_preview_formats:
                tmp_results.append(ImageSaveHelperExt.get_save_result_temp(image, mask))
            else:
                results.append(result)

            c += 1

        return io.NodeOutput(
            ui=(
                ui.SavedImages(tmp_results)
                if format in cls.web_unsupported_preview_formats
                else ui.SavedImages(results)
            )
        )


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
                    tooltip="The images to be saved.",
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    default=define.default_file_name,
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
        **kwargs,
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = (
            ImageSaveHelperExt.get_save_image_path(
                filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = []
        for batch_number, image in enumerate(images):
            result = ImageSaveHelperExt.get_save_result(
                image=image,
                mask=None,
                convert_mode="RGB",
                join_mask=False,
                filename=filename,
                full_output_folder=f_output_folder,
                full_input_folder=None,
                subfolder=subfolder,
                batch_number=batch_number,
                counter=c,
                file_ext="jpg",
                save_to_input_folder=False,
                save_kwargs={
                    "quality": 95,
                    "optimize": True,
                    "dpi": (define.screen_dpi, define.screen_dpi),
                },
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
                    tooltip="The images to be saved.",
                ),
                io.Boolean.Input(
                    id="disable_metadata",
                    display_name="Disable Metadata",
                    tooltip="Disable embedding EXIF data into the saved images.\nEmbed into `UserComment`.",
                    default=False,
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    default=define.default_file_name,
                ),
                io.Int.Input(
                    id="dpi",
                    display_name="DPI",
                    tooltip="The DPI to use when saving images.",
                    default=define.printing_dpi,
                    min=1,
                    max=600,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Int.Input(
                    id="quality",
                    display_name="Quality",
                    tooltip="The quality of the saved images.",
                    default=95,
                    min=1,
                    max=100,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Boolean.Input(
                    id="save_to_input_folder",
                    display_name="Save to Input Folder",
                    tooltip="Whether to sync the image to the input folder.",
                    default=False,
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
        disable_metadata: bool = False,
        dpi: int = define.screen_dpi,
        quality: int = 95,
        save_to_input_folder: bool = False,
        **kwargs,
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = (
            ImageSaveHelperExt.get_save_image_path(
                filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        f_input_folder = os.path.join(folder_paths.get_input_directory(), subfolder)

        results = []
        for batch_number, image in enumerate(images):
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
                image=image,
                mask=None,
                convert_mode="RGB",
                join_mask=False,
                filename=filename,
                full_output_folder=f_output_folder,
                full_input_folder=f_input_folder,
                subfolder=subfolder,
                batch_number=batch_number,
                counter=c,
                file_ext="jpg",
                save_to_input_folder=save_to_input_folder,
                save_kwargs=save_options,
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
                    tooltip="The images to be saved.",
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    default=define.default_file_name,
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
        **kwargs,
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = (
            ImageSaveHelperExt.get_save_image_path(
                filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = []
        for batch_number, image in enumerate(images):
            result = ImageSaveHelperExt.get_save_result(
                image=image,
                mask=None,
                convert_mode="RGB",
                join_mask=False,
                filename=filename,
                full_output_folder=f_output_folder,
                full_input_folder=None,
                subfolder=subfolder,
                batch_number=batch_number,
                counter=c,
                file_ext="bmp",
                save_to_input_folder=False,
                save_kwargs={"dpi": (define.screen_dpi, define.screen_dpi)},
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
                    tooltip="The images to be saved.",
                ),
                io.Mask.Input(
                    id="masks",
                    display_name="Alpha Masks",
                    tooltip="Optional alpha channel masks to clip the saved images. If provided, these masks will be applied to the corresponding images before saving.",
                    optional=True,
                ),
                io.Combo.Input(
                    id="bit_depth",
                    display_name="Bit Depth",
                    tooltip="The bit depth to use when saving images as BMP.\n32bit supports alpha channel.",
                    options=["24bit", "32bit"],
                    default="24bit",
                ),
                io.Boolean.Input(
                    id="join_alpha",
                    display_name="Join Alpha Channel",
                    tooltip="Clip the image with the provided mask.\nPlease note this only works on 32bit bit depth.",
                    default=False,
                ),
                io.Boolean.Input(
                    id="invert_alpha",
                    display_name="Invert Alpha",
                    tooltip="Whether to invert the alpha channel before saving.",
                    default=False,
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    default=define.default_file_name,
                ),
                io.Int.Input(
                    id="dpi",
                    display_name="DPI",
                    tooltip="The DPI to use when saving images.",
                    default=define.printing_dpi,
                    min=1,
                    max=600,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Boolean.Input(
                    id="save_to_input_folder",
                    display_name="Save to Input Folder",
                    tooltip="Whether to sync the image to the input folder.",
                    default=False,
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
        bit_depth: str = "24bit",
        join_alpha: bool = False,
        invert_alpha: bool = False,
        filename_prefix: str = define.default_file_name,
        dpi: int = define.printing_dpi,
        save_to_input_folder: bool = False,
        **kwargs,
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = (
            ImageSaveHelperExt.get_save_image_path(
                filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        f_input_folder = os.path.join(folder_paths.get_input_directory(), subfolder)

        results = []
        for batch_number, image in enumerate(images):
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
                image=image,
                mask=mask,
                convert_mode=mode,
                join_mask=join_alpha,
                filename=filename,
                full_output_folder=f_output_folder,
                full_input_folder=f_input_folder,
                subfolder=subfolder,
                batch_number=batch_number,
                counter=c,
                file_ext="bmp",
                save_to_input_folder=save_to_input_folder,
                save_kwargs=save_options,
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
                    tooltip="The images to be saved.",
                ),
                io.Mask.Input(
                    id="masks",
                    display_name="Alpha Masks",
                    tooltip="Optional alpha channel masks to clip the saved images. If provided, these masks will be applied to the corresponding images before saving.",
                    optional=True,
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    default=define.default_file_name,
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
        **kwargs,
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = (
            ImageSaveHelperExt.get_save_image_path(
                filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )

        preview_results = []
        for batch_number, image in enumerate(images):
            mode = "RGB"
            mask = None
            if masks is not None:
                mask_index = min(batch_number, masks.shape[0] - 1)
                mask = masks[mask_index]
                mode = "RGBA"
            should_join_mask = mask is not None

            # Drop the result
            ImageSaveHelperExt.get_save_result(
                image=image,
                mask=mask,
                convert_mode=mode,
                join_mask=should_join_mask,
                filename=filename,
                full_output_folder=f_output_folder,
                full_input_folder=None,
                subfolder=subfolder,
                batch_number=batch_number,
                counter=c,
                file_ext="tiff",
                save_to_input_folder=False,
                save_kwargs={
                    "compression": "none",
                    "dpi": (define.screen_dpi, define.screen_dpi),
                },
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
                    tooltip="The images to be saved.",
                ),
                io.Mask.Input(
                    id="masks",
                    display_name="Alpha Masks",
                    tooltip="Optional alpha channel masks to clip the saved images. If provided, these masks will be applied to the corresponding images before saving.",
                    optional=True,
                ),
                io.Boolean.Input(
                    id="disable_metadata",
                    display_name="Disable Metadata",
                    tooltip="Disable embedding EXIF data into the saved images.\nEmbed into `UserComment`.",
                    default=False,
                ),
                io.Boolean.Input(
                    id="join_alpha",
                    display_name="Join Alpha Channel",
                    tooltip="Clip the image with the provided mask.\nPlease note this does not work on JPG format.",
                    default=False,
                ),
                io.Boolean.Input(
                    id="invert_alpha",
                    display_name="Invert Alpha",
                    tooltip="Whether to invert the alpha channel before saving.",
                    default=False,
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    default=define.default_file_name,
                ),
                io.Int.Input(
                    id="dpi",
                    display_name="DPI",
                    tooltip="The DPI to use when saving images.",
                    default=define.printing_dpi,
                    min=1,
                    max=600,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Combo.Input(
                    id="tiff_compression",
                    display_name="Compression",
                    tooltip="The compression to use when saving images as TIFF.",
                    options=[
                        "none",
                        "tiff_lzw",
                        "tiff_deflate",
                        "tiff_adobe_deflate",
                        "packbits",
                        "jpeg",
                        "tiff_jpeg",
                        "tiff_ccitt",
                    ],
                    default="none",
                ),
                io.Boolean.Input(
                    id="save_to_input_folder",
                    display_name="Save to Input Folder",
                    tooltip="Whether to sync the image to the input folder.",
                    default=False,
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
        disable_metadata: bool = False,
        join_alpha: bool = False,
        invert_alpha: bool = False,
        tiff_compression: str = "none",
        dpi: int = define.printing_dpi,
        save_to_input_folder: bool = False,
        **kwargs,
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = (
            ImageSaveHelperExt.get_save_image_path(
                filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        f_input_folder = os.path.join(folder_paths.get_input_directory(), subfolder)

        preview_results = []
        for batch_number, image in enumerate(images):
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
                logger.warning(
                    f"{cls.debug_header} tiff_ccitt compression requires a bilevel (1-bit) image."
                )
                mode = "1"

            mask = None
            if join_alpha:
                mask_index = min(batch_number, masks.shape[0] - 1)
                mask = masks[mask_index]
                if invert_alpha:
                    mask = 1.0 - mask

            ImageSaveHelperExt.get_save_result(
                image=image,
                mask=mask,
                convert_mode=mode,
                join_mask=join_alpha,
                filename=filename,
                full_output_folder=f_output_folder,
                full_input_folder=f_input_folder,
                subfolder=subfolder,
                batch_number=batch_number,
                counter=c,
                file_ext="tiff",
                save_to_input_folder=save_to_input_folder,
                save_kwargs=save_options,
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
                    tooltip="The images to be saved.",
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    default=define.default_file_name,
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
        **kwargs,
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = (
            ImageSaveHelperExt.get_save_image_path(
                filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )

        preview_results = []
        for batch_number, image in enumerate(images):
            ImageSaveHelperExt.get_save_result(
                image=image,
                mask=None,
                convert_mode="RGB",
                join_mask=False,
                filename=filename,
                full_output_folder=f_output_folder,
                full_input_folder=None,
                subfolder=subfolder,
                batch_number=batch_number,
                counter=c,
                file_ext="tga",
                save_to_input_folder=False,
                save_kwargs={"compression": "tga_rle"},
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
                    tooltip="The images to be saved.",
                ),
                io.Mask.Input(
                    id="masks",
                    display_name="Alpha Masks",
                    tooltip="Optional alpha channel masks to clip the saved images. If provided, these masks will be applied to the corresponding images before saving.",
                    optional=True,
                ),
                io.Boolean.Input(
                    id="rle",
                    display_name="RLE Compression",
                    tooltip="Whether to use RLE compression when saving as TGA.",
                    default=True,
                ),
                io.Boolean.Input(
                    id="join_alpha",
                    display_name="Join Alpha Channel",
                    tooltip="Clip the image with the provided mask.",
                    default=False,
                ),
                io.Boolean.Input(
                    id="invert_alpha",
                    display_name="Invert Alpha",
                    tooltip="Whether to invert the alpha channel before saving.",
                    default=False,
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    default=define.default_file_name,
                ),
                io.Boolean.Input(
                    id="save_to_input_folder",
                    display_name="Save to Input Folder",
                    tooltip="Whether to sync the image to the input folder.",
                    default=False,
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
        rle: bool = True,
        join_alpha: bool = False,
        invert_alpha: bool = False,
        save_to_input_folder: bool = False,
        **kwargs,
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = (
            ImageSaveHelperExt.get_save_image_path(
                filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        f_input_folder = os.path.join(folder_paths.get_input_directory(), subfolder)

        preview_results = []
        for batch_number, image in enumerate(images):
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
                image=image,
                mask=mask,
                convert_mode=mode,
                join_mask=join_alpha,
                filename=filename,
                full_output_folder=f_output_folder,
                full_input_folder=f_input_folder,
                subfolder=subfolder,
                batch_number=batch_number,
                counter=c,
                file_ext="tga",
                save_to_input_folder=save_to_input_folder,
                save_kwargs=save_options,
            )

            preview_results.append(ImageSaveHelperExt.get_save_result_temp(image, mask))

            c += 1

        return io.NodeOutput(ui=ui.SavedImages(preview_results))


class SaveImageAVIF(io.ComfyNode):
    """
    Saves images to disk as AVIF files.
    """

    output_dir = folder_paths.get_output_directory()
    prefix_append = ""

    debug_header = "[ComfyUI-SaveImageAVIF]"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageAVIF",
            display_name="Save Image (AVIF)",
            category=define.author,
            description="Saves images to disk as AVIF files.",
            inputs=[
                io.Image.Input(
                    id="images",
                    display_name="Images",
                    tooltip="The images to be saved.",
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    default=define.default_file_name,
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
        **kwargs,
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = (
            ImageSaveHelperExt.get_save_image_path(
                filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )

        results = []
        for batch_number, image in enumerate(images):
            result = ImageSaveHelperExt.get_save_result(
                image=image,
                mask=None,
                convert_mode="RGB",
                join_mask=False,
                filename=filename,
                full_output_folder=f_output_folder,
                full_input_folder=None,
                subfolder=subfolder,
                batch_number=batch_number,
                counter=c,
                file_ext="avif",
                save_to_input_folder=False,
                save_kwargs={
                    "quality": 85,
                    "speed": 6,
                    "dpi": (define.screen_dpi, define.screen_dpi),
                },
            )

            results.append(result)
            c += 1

        return io.NodeOutput(ui=ui.SavedImages(results))


class SaveImageAdvancedAVIF(io.ComfyNode):
    """
    Saves images to disk as AVIF files with additional options.
    """

    output_dir = folder_paths.get_output_directory()
    prefix_append = ""

    debug_header = "[ComfyUI-SaveImageAdvancedAVIF]"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageAdvancedAVIF",
            display_name="Save Image Advanced (AVIF)",
            category=define.author,
            description="Saves images to disk as AVIF files.",
            inputs=[
                io.Image.Input(
                    id="images",
                    display_name="Images",
                    tooltip="The images to be saved.",
                ),
                io.Mask.Input(
                    id="masks",
                    display_name="Alpha Masks",
                    tooltip="Optional alpha channel masks to clip the saved images. If provided, these masks will be applied to the corresponding images before saving.",
                    optional=True,
                ),
                io.Boolean.Input(
                    id="join_alpha",
                    display_name="Join Alpha Channel",
                    tooltip="Clip the image with the provided mask.",
                    default=False,
                ),
                io.Boolean.Input(
                    id="invert_alpha",
                    display_name="Invert Alpha",
                    tooltip="Whether to invert the alpha channel before saving.",
                    default=False,
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    default=define.default_file_name,
                ),
                io.Int.Input(
                    id="dpi",
                    display_name="DPI",
                    tooltip="The DPI to use when saving images.",
                    default=define.printing_dpi,
                    min=1,
                    max=600,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Int.Input(
                    id="quality",
                    display_name="Quality",
                    tooltip="The quality of the saved images (0-100).",
                    default=80,
                    min=0,
                    max=100,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Int.Input(
                    id="speed",
                    display_name="Speed",
                    tooltip="AVIF encoding speed (0-10). 0 is slowest/best, 10 is fastest.",
                    default=6,
                    min=0,
                    max=10,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Boolean.Input(
                    id="save_to_input_folder",
                    display_name="Save to Input Folder",
                    tooltip="Whether to sync the image to the input folder.",
                    default=False,
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
        join_alpha: bool = False,
        invert_alpha: bool = False,
        dpi: int = define.printing_dpi,
        quality: int = 80,
        speed: int = 6,
        save_to_input_folder: bool = False,
        **kwargs,
    ) -> io.NodeOutput:
        f_output_folder, filename, c, subfolder, filename_prefix = (
            ImageSaveHelperExt.get_save_image_path(
                filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        f_input_folder = os.path.join(folder_paths.get_input_directory(), subfolder)

        results = []
        for batch_number, image in enumerate(images):
            mode = "RGBA" if join_alpha else "RGB"
            save_options = {
                "quality": quality,
                "speed": speed,
                "dpi": (dpi, dpi),
            }

            mask = None
            if join_alpha and masks is not None:
                mask_index = min(batch_number, masks.shape[0] - 1)
                mask = masks[mask_index]
                if invert_alpha:
                    mask = 1.0 - mask

            result = ImageSaveHelperExt.get_save_result(
                image=image,
                mask=mask,
                convert_mode=mode,
                join_mask=join_alpha,
                filename=filename,
                full_output_folder=f_output_folder,
                full_input_folder=f_input_folder,
                subfolder=subfolder,
                batch_number=batch_number,
                counter=c,
                file_ext="avif",
                save_to_input_folder=save_to_input_folder,
                save_kwargs=save_options,
            )

            results.append(result)
            c += 1

        return io.NodeOutput(ui=ui.SavedImages(results))


class SaveImageICO(io.ComfyNode):
    """
    Save multiple images to disk as a bundled ICO file.
    """

    output_dir = folder_paths.get_output_directory()
    prefix_append = ""

    debug_header = "[ComfyUI-SaveImageICO]"

    valid_ico_sizes = {16, 24, 32, 48, 64, 128, 256}
    max_images = 256

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageICO",
            display_name="Save Image (ICO)",
            category=define.author,
            description="Save the image batch to disk as a single bundled ICO file.\nImages must be square and one of the standard ICO sizes\n(16, 24, 32, 48, 64, 128, 256).\nInvalid images are optionally saved as fallback PNGs.",
            inputs=[
                io.Image.Input(
                    id="images",
                    display_name="Images",
                    tooltip="The images to be bundled into a single ICO file.\nEach image must be square and match a valid ICO size\n(16, 24, 32, 48, 64, 128, or 256 px).",
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    default=define.default_file_name,
                ),
                io.Boolean.Input(
                    id="save_invalid_as_png",
                    display_name="Save Invalid as PNG",
                    tooltip="When enabled, images that fail ICO validation are saved as\nindividual PNG files with an `_INVALID` suffix.\nWhen disabled, invalid images are silently discarded.",
                    default=True,
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def is_valid_ico_frame(cls, img: Image.Image) -> bool:
        return img.width == img.height and img.width in cls.valid_ico_sizes

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        filename_prefix: str = define.default_file_name,
        save_invalid_as_png: bool = True,
        **kwargs,
    ) -> io.NodeOutput:
        batch_size = images.shape[0]
        if batch_size > cls.max_images:
            raise ValueError(
                f"{cls.debug_header} Batch size {batch_size} exceeds the ICO limit of {cls.max_images} frames."
            )

        f_output_folder, filename, c, subfolder, filename_prefix = (
            ImageSaveHelperExt.get_save_image_path(
                filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        if not os.path.exists(f_output_folder):
            os.makedirs(f_output_folder)

        pil_images = ImageSaveHelperExt.to_pillow_images(images)

        valid_frames: list[Image.Image] = []
        invalid_frames: list[tuple[int, Image.Image]] = []
        for idx, img in enumerate(pil_images):
            img = img.convert("RGBA")
            if cls.is_valid_ico_frame(img):
                valid_frames.append(img)
            else:
                invalid_frames.append((idx, img))

        results = []

        if valid_frames:
            ico_file = f"{filename}_{c:05}_.ico"
            ico_path = os.path.join(f_output_folder, ico_file)
            valid_frames[0].save(
                ico_path,
                format="ICO",
                append_images=valid_frames[1:],
                sizes=[(img.width, img.height) for img in valid_frames],
            )
            results.append(ui.SavedResult(ico_file, subfolder, io.FolderType.output))
            c += 1

        if invalid_frames:
            if save_invalid_as_png:
                for idx, img in invalid_frames:
                    png_file = f"{filename}_{c:05}_INVALID.png"
                    img.save(os.path.join(f_output_folder, png_file))
                    results.append(
                        ui.SavedResult(png_file, subfolder, io.FolderType.output)
                    )
                    c += 1
            else:
                logger.warning(
                    f"{cls.debug_header} {len(invalid_frames)} image(s) failed ICO validation and were discarded "
                    f"(indices: {[i for i, _ in invalid_frames]}). "
                    f"Valid ICO frames must be square and one of: {sorted(cls.valid_ico_sizes)} px."
                )

        return io.NodeOutput(ui=ui.SavedImages(results))


class SaveImageAdvancedICO(io.ComfyNode):
    """
    Save multiple images to disk as a bundled ICO file with explicit size choices.
    """

    output_dir = folder_paths.get_output_directory()
    prefix_append = ""

    debug_header = "[ComfyUI-SaveImageICO]"

    max_images = 256

    valid_ico_sizes = {16, 24, 32, 48, 64, 128, 256}
    _SIZE_SLOTS = [16, 24, 32, 48, 64, 128, 256]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImageAdvancedICO",
            display_name="Save Image Advanced (ICO)",
            category=define.author,
            description="Save the image batch to disk as a single bundled ICO file.\nImages must be square and one of the standard ICO sizes\n(16, 24, 32, 48, 64, 128, 256).\nInvalid images are optionally saved as fallback PNGs.",
            inputs=[
                io.Image.Input(
                    id="images_16",
                    display_name="Images (16)",
                    tooltip="The images to be bundled into a single ICO file.\nThe input image must be 16x16 for this slot.",
                    optional=True,
                ),
                io.Mask.Input(
                    id="masks_16",
                    display_name="Alpha Masks (16)",
                    tooltip="Optional alpha channel masks to clip the saved images.\nIf provided, these masks will be applied to the corresponding images before saving.",
                    optional=True,
                ),
                io.Image.Input(
                    id="images_24",
                    display_name="Images (24)",
                    tooltip="The images to be bundled into a single ICO file.\nThe input image must be 24x24 for this slot.",
                    optional=True,
                ),
                io.Mask.Input(
                    id="masks_24",
                    display_name="Alpha Masks (24)",
                    tooltip="Optional alpha channel masks to clip the saved images.\nIf provided, these masks will be applied to the corresponding images before saving.",
                    optional=True,
                ),
                io.Image.Input(
                    id="images_32",
                    display_name="Images (32)",
                    tooltip="The images to be bundled into a single ICO file.\nThe input image must be 32x32 for this slot.",
                    optional=True,
                ),
                io.Mask.Input(
                    id="masks_32",
                    display_name="Alpha Masks (32)",
                    tooltip="Optional alpha channel masks to clip the saved images.\nIf provided, these masks will be applied to the corresponding images before saving.",
                    optional=True,
                ),
                io.Image.Input(
                    id="images_48",
                    display_name="Images (48)",
                    tooltip="The images to be bundled into a single ICO file.\nThe input image must be 48x48 for this slot.",
                    optional=True,
                ),
                io.Mask.Input(
                    id="masks_48",
                    display_name="Alpha Masks (48)",
                    tooltip="Optional alpha channel masks to clip the saved images.\nIf provided, these masks will be applied to the corresponding images before saving.",
                    optional=True,
                ),
                io.Image.Input(
                    id="images_64",
                    display_name="Images (64)",
                    tooltip="The images to be bundled into a single ICO file.\nThe input image must be 64x64 for this slot.",
                    optional=True,
                ),
                io.Mask.Input(
                    id="masks_64",
                    display_name="Alpha Masks (64)",
                    tooltip="Optional alpha channel masks to clip the saved images.\nIf provided, these masks will be applied to the corresponding images before saving.",
                    optional=True,
                ),
                io.Image.Input(
                    id="images_128",
                    display_name="Images (128)",
                    tooltip="The images to be bundled into a single ICO file.\nThe input image must be 128x128 for this slot.",
                    optional=True,
                ),
                io.Mask.Input(
                    id="masks_128",
                    display_name="Alpha Masks (128)",
                    tooltip="Optional alpha channel masks to clip the saved images.\nIf provided, these masks will be applied to the corresponding images before saving.",
                    optional=True,
                ),
                io.Image.Input(
                    id="images_256",
                    display_name="Images (256)",
                    tooltip="The images to be bundled into a single ICO file.\nThe input image must be 256x256 for this slot.",
                    optional=True,
                ),
                io.Mask.Input(
                    id="masks_256",
                    display_name="Alpha Masks (256)",
                    tooltip="Optional alpha channel masks to clip the saved images.\nIf provided, these masks will be applied to the corresponding images before saving.",
                    optional=True,
                ),
                io.Combo.Input(
                    id="color_depth",
                    display_name="Color Depth",
                    tooltip="The color depth of the images.",
                    options=["8bit", "16bit", "32bit"],
                    default="8bit",
                ),
                io.Combo.Input(
                    id="compression",
                    display_name="Compression",
                    tooltip="The compression of the images.",
                    options=["none", "RLE", "ZIP"],
                    default="none",
                ),
                io.Boolean.Input(
                    id="join_alpha",
                    display_name="Join Alpha Channel",
                    tooltip="Whether to join the alpha channel of the images.",
                    default=False,
                ),
                io.Boolean.Input(
                    id="invert_alpha",
                    display_name="Invert Alpha",
                    tooltip="Whether to invert the alpha channel of the images.",
                    default=False,
                ),
                io.String.Input(
                    id="filename_prefix",
                    display_name="Filename Prefix",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    default=define.default_file_name,
                ),
                io.Boolean.Input(
                    id="save_to_input_folder",
                    display_name="Save to Input Folder",
                    tooltip="Whether to sync the image to the input folder.",
                    default=False,
                ),
                io.Boolean.Input(
                    id="sort_by_size",
                    display_name="Sort by Size",
                    tooltip="When enabled, the images will be sorted by size before being bundled into the ICO file.",
                    default=False,
                ),
                io.Boolean.Input(
                    id="save_invalid_as_png",
                    display_name="Save Invalid as PNG",
                    tooltip="When enabled, images that fail ICO validation are saved as\nindividual PNG files with an `_INVALID` suffix.\nWhen disabled, invalid images are silently discarded.",
                    default=True,
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        images_16: torch.Tensor | None = None,
        images_24: torch.Tensor | None = None,
        images_32: torch.Tensor | None = None,
        images_48: torch.Tensor | None = None,
        images_64: torch.Tensor | None = None,
        images_128: torch.Tensor | None = None,
        images_256: torch.Tensor | None = None,
        masks_16: torch.Tensor | None = None,
        masks_24: torch.Tensor | None = None,
        masks_32: torch.Tensor | None = None,
        masks_48: torch.Tensor | None = None,
        masks_64: torch.Tensor | None = None,
        masks_128: torch.Tensor | None = None,
        masks_256: torch.Tensor | None = None,
        color_depth: str = "8bit",
        compression: str = "none",
        join_alpha: bool = False,
        invert_alpha: bool = False,
        filename_prefix: str = define.default_file_name,
        save_to_input_folder: bool = False,
        sort_by_size: bool = False,
        save_invalid_as_png: bool = True,
        **kwargs,
    ) -> io.NodeOutput:
        input_tensors: dict[int, torch.Tensor] = {
            size: tensor
            for size, tensor in zip(cls._SIZE_SLOTS, [
                images_16, images_24, images_32, images_48,
                images_64, images_128, images_256,
            ])
            if tensor is not None
        }
        input_masks: dict[int, torch.Tensor] = {
            size: mask
            for size, mask in zip(cls._SIZE_SLOTS, [
                masks_16, masks_24, masks_32, masks_48,
                masks_64, masks_128, masks_256,
            ])
            if mask is not None
        }

        if not input_tensors:
            raise ValueError(f"{cls.debug_header} No images provided to any size slot.")

        ref_size = min(input_tensors)
        ref_tensor = input_tensors[ref_size]
        f_output_folder, filename, c, subfolder, filename_prefix = (
            ImageSaveHelperExt.get_save_image_path(
                filename_prefix, cls.output_dir,
                ref_tensor.shape[2], ref_tensor.shape[1],
            )
        )
        f_input_folder = os.path.join(folder_paths.get_input_directory(), subfolder)
        os.makedirs(f_output_folder, exist_ok=True)

        # Resolve target color mode.
        # join_alpha / 32bit → RGBA; 16bit → RGB; 8bit → P (256-color palette).
        if join_alpha or color_depth == "32bit":
            target_mode = "RGBA"
        elif color_depth == "8bit":
            target_mode = "P"
        else:
            target_mode = "RGB"

        valid_frames: list[tuple[int, Image.Image]] = []
        invalid_frames: list[tuple[int, Image.Image]] = []

        ordered_sizes = sorted(input_tensors) if sort_by_size else list(input_tensors)
        for expected_size in ordered_sizes:
            img = ImageSaveHelperExt.to_pillow_image(input_tensors[expected_size], 0)

            img = img.convert("RGBA")

            if join_alpha:
                mask_tensor = input_masks.get(expected_size)
                if mask_tensor is not None:
                    m = mask_tensor[0] if mask_tensor.ndim == 3 else mask_tensor
                    if invert_alpha:
                        m = 1.0 - m
                    m_np = (m.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    m_pil = Image.fromarray(m_np, mode="L")
                    if m_pil.size != (img.width, img.height):
                        m_pil = m_pil.resize((img.width, img.height), Image.LANCZOS)
                    img.putalpha(m_pil)
                elif invert_alpha:
                    r, g, b, a = img.split()
                    a = a.point(lambda x: 255 - x)
                    img = Image.merge("RGBA", (r, g, b, a))

            img = img.convert(target_mode)

            if img.width == expected_size and img.height == expected_size:
                valid_frames.append((expected_size, img))
            else:
                invalid_frames.append((expected_size, img))

        results = []

        if valid_frames:
            pil_frames = [img for _, img in valid_frames]
            sizes = [(img.width, img.height) for img in pil_frames]
            ico_file = f"{filename}_{c:05}_.ico"
            ico_path = os.path.join(f_output_folder, ico_file)

            save_kwargs: dict = {"sizes": sizes}
            if compression == "ZIP":
                # PNG-compressed frames (Windows Vista+ / Pillow extension)
                save_kwargs["bitmap_format"] = "png"

            pil_frames[0].save(
                ico_path,
                format="ICO",
                append_images=pil_frames[1:],
                **save_kwargs,
            )
            results.append(ui.SavedResult(ico_file, subfolder, io.FolderType.output))

            if save_to_input_folder:
                os.makedirs(f_input_folder, exist_ok=True)
                target = os.path.join(f_input_folder, ico_file)
                if os.path.exists(target):
                    target = os.path.join(
                        f_input_folder,
                        f"{filename}_{c:05}_{int(time.time())}.ico",
                    )
                pil_frames[0].save(
                    target, format="ICO", append_images=pil_frames[1:], **save_kwargs
                )

            c += 1

        if invalid_frames:
            if save_invalid_as_png:
                for expected_size, img in invalid_frames:
                    png_file = f"{filename}_{c:05}_INVALID.png"
                    png_img = img.convert("RGBA") if img.mode == "P" else img
                    png_img.save(os.path.join(f_output_folder, png_file))
                    results.append(ui.SavedResult(png_file, subfolder, io.FolderType.output))
                    c += 1
            else:
                logger.warning(
                    f"{cls.debug_header} {len(invalid_frames)} image(s) failed ICO size validation "
                    f"and were discarded (expected sizes: {[s for s, _ in invalid_frames]}). "
                    f"Each slot only accepts an image whose dimensions exactly match its label."
                )

        return io.NodeOutput(ui=ui.SavedImages(results))


NODE_CLASS_MAPPINGS = {
    "ColorPatchFlatten": ColorPatchFlatten,
    "ColorPatchMerge": ColorPatchMerge,
    "AchromaticStabilizer": AchromaticStabilizer,
    "LoadICCProfile": LoadICCProfile,
    "SaveImageAdvancedCustom": SaveImageAdvancedCustom,  # Avoid naming conflicts with the official node.
    "SaveImageJPG": SaveImageJPG,
    "SaveImageAdvancedJPG": SaveImageAdvancedJPG,
    "SaveImageBMP": SaveImageBMP,
    "SaveImageAdvancedBMP": SaveImageAdvancedBMP,
    "SaveImageTIFF": SaveImageTIFF,
    "SaveImageAdvancedTIFF": SaveImageAdvancedTIFF,
    "SaveImageTGA": SaveImageTGA,
    "SaveImageAdvancedTGA": SaveImageAdvancedTGA,
    "SaveImageAVIF": SaveImageAVIF,
    "SaveImageAdvancedAVIF": SaveImageAdvancedAVIF,
    "SaveImageICO": SaveImageICO,
    "SaveImageAdvancedICO": SaveImageAdvancedICO,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorPatchFlatten": "Color Patch Flatten",
    "ColorPatchMerge": "Color Patch Merge",
    "AchromaticStabilizer": "Achromatic Stabilizer",
    "LoadICCProfile": "Load ICC Profile",
    "SaveImageAdvancedCustom": "Save Image (Custom Advanced)",
    "SaveImageJPG": "Save Image (JPG)",
    "SaveImageAdvancedJPG": "Save Image Advanced (JPG)",
    "SaveImageBMP": "Save Image (BMP)",
    "SaveImageAdvancedBMP": "Save Image Advanced (BMP)",
    "SaveImageTIFF": "Save Image (TIFF)",
    "SaveImageAdvancedTIFF": "Save Image Advanced (TIFF)",
    "SaveImageTGA": "Save Image (TGA)",
    "SaveImageAdvancedTGA": "Save Image Advanced (TGA)",
    "SaveImageAVIF": "Save Image (AVIF)",
    "SaveImageAdvancedAVIF": "Save Image Advanced (AVIF)",
    "SaveImageICO": "Save Image (ICO)",
    "SaveImageAdvancedICO": "Save Image Advanced (ICO)",
}
