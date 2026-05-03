# ComfyUI Image Process

A collection of image-processing nodes designed to enhance and refine AI-generated images within ComfyUI workflows.

<p align="center">
<img src="https://github.com/wakaura-asaho/comfyui-image-process/blob/main/docs/logo.png" alt="Logo" style="display: block; margin: 0 auto; text-align: center;">
</p>

## Nodes Included

### 1. Achromatic Stabilizer

Corrects achromatic color instability commonly found in AI-generated images.

* **Artifact Detection:** Identifies near-zero saturation pixels.
* **Saturation Threshold:** Adjusts sensitivity to define artifacts (0.0 - 1.0).
* **Smart Normalization:** Converts artifact pixels to neutral grey.
* **Optional Smoothing:** Blends corrected areas with the image.
* **Alpha Channel Preservation:** Maintains transparency during processing.
* **Invert Alpha:** Inverts the alpha channel.
* **Dual Output:** Returns both image and alpha mask separately.

### 2. Color Patch Flatten

Homogenizes colors by flattening their HSV values across patches within a given tolerance.

* **HSV Control:** Independently toggles flattening for Hue (H), Saturation (S), and Brightness (V).
* **Tolerance Handling:** Flattens values only within the specified tolerance. Larger numbers mean a weaker effect.
* **Alpha Mask Support:** Restricts the region for color calculation and flattening.

### 3. Color Patch Merge

Groups adjacent similar colors and replaces them with their local averages, clustering pixels into solid patches.

* **Merge Solutions:**
  * **Smooth:** Uses a bilateral filter to smooth colors while preserving edges (controlled by `Neighborhood`).
  * **Unify:** Iteratively grows solid color regions in Lab space (controlled by `Iterations` and `Minimal Area`).
* **Lab Color Space:** Optionally converts RGB to Lab space for perceptually accurate grouping.
* **Detail Preservation:** "Unify" protects high-gradient edges and small regions below the `Minimal Area`.

### 4. Load ICC Profile

Loads an ICC color profile from `models/icc_profiles`.

![Example_workflow](https://github.com/wakaura-asaho/comfyui-image-process/blob/main/docs/save_image_icc.png)

* **Profile Selection:** Choose valid `.icc` or `.icm` files.
* **Embedding:** Outputs profile data for the `Save Image Advanced` node.
* **Profile Information:** Outputs metadata (Model, Manufacturer, Description, Copyright).

> [!WARNING]
> `BMP` and `TGA` do not support ICC Profile embedding.

### 5. Save Image Advanced

Saves images with alpha channels and metadata.

* **ICC Profile:** Embeds a custom color profile.
* **Disable Metadata:** Discards workflow metadata to reduce file size.
* **Join Alpha Channel:** Clips the image with a mask, similar to `Join Image with Alpha`.
* **Invert Alpha:** Inverts the alpha channel.
* **Compression Level:** PNG compression (0-9).
* **Quality:** JPEG/WebP quality (1-100).
* **DPI:** Sets DPI metadata (1–600, default: 300).
* **TIFF Compression:** Supports LZW, Deflate, CCITT, etc.
* **TGA RLE Compression:** Toggles RLE compression for TGA.
* **Format:** Supports `JPG`, `PNG`, `WebP`, `TIFF`, `BMP`, and `TGA`.
* **Bit Depth:** `24bit` or `32bit` for BMP (32-bit supports alpha).
* **Save to Input Folder:** Copies the saved file to the input folder for reuse.

### 6. Dedicated Format Savers

Format-specific savers with simple and advanced variants.

![Example_workflow](https://github.com/wakaura-asaho/comfyui-image-process/blob/main/docs/save_image_simple.png)

* **JPG:** Quick saves or advanced EXIF/DPI control.
* **BMP:** Standard saves or 32-bit alpha/DPI support.
* **TIFF:** Standard saves or advanced compression/EXIF options.
* **TGA:** Standard saves or RLE compression/alpha support.

> [!WARNING]
> ComfyUI may not display TIFF or TGA files properly in the preview pane. The preview is a PNG copy, but the actual file is saved in your chosen format.

> [!TIP]
> `tiff_ccitt` is for black-and-white documents or line-art. For photography, use `tiff_lzw`, `tiff_deflate`, or `tiff_adobe_deflate`.

---

## Example Usage

The `Achromatic Stabilizer` is particularly useful for:

* **Removing color noise** from near-white or near-black regions.
* **Improving consistency** for models prone to desaturated, hue-unstable pixels.
* **Preserving transparency** while correcting color artifacts.

![Example_workflow](https://github.com/wakaura-asaho/comfyui-image-process/blob/main/docs/example.png)

To use the node:

1. Connect to **Image**.
2. (Optional) Connect an **Alpha Mask**.
3. Adjust **Saturation Threshold** (lower = more aggressive).
4. Enable **Smooth Transitions** to blend corrected areas.
5. Set **Smoothing Kernel Size** for blending intensity.
6. Toggle **Preserve Alpha** as needed.

The `Save Image Advanced` node can finalize your workflow with extra tweaks:

![Example_workflow](https://github.com/wakaura-asaho/comfyui-image-process/blob/main/docs/save_image_adv.png)

The example above removes the background and sends the mask to save with an alpha channel.
Enable `Disable Metadata` to reduce file size at the cost of workflow readability.

---

## Installation

### Method 1: ComfyUI Manager (Recommended)

1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager).
2. Click on **"Install via Git URL"**.
3. Paste the URL: `https://github.com/wakaura-asaho/comfyui-image-process`
4. Restart ComfyUI.

### Method 2: Manual Installation

1. Open a terminal in `ComfyUI/custom_nodes`.
2. Clone this repository:
```bash
git clone https://github.com/wakaura-asaho/comfyui-image-process.git
```
3. Install dependencies:
```bash
pip install -r comfyui-image-process/requirements.txt
```
4. Restart ComfyUI.

> [!WARNING]
> **Dependency Notice:** Use `pip install -r requirements.txt --upgrade-strategy only-if-needed` to avoid unnecessary version conflicts.

---

## File Structure

* `image_process.py`: Backend logic and node class definitions.

## Usage Tips

> [!TIP]
> **Saturation Threshold:** Start at 0.08. Use lower values (e.g., 0.05) for more aggressive correction, or higher values (e.g., 0.15) to be conservative.

> [!TIP]
> **Smoothing Kernel Size:** Use 3-5 for subtle corrections. Try 7-15 for pronounced smoothing, though it may blur regions.

> [!TIP]
> **Alpha Preservation:** Enable **Preserve Alpha** with an optional mask to maintain transparency.

## Compatible Versions and Notices

Designed for modern ComfyUI versions (V3 nodes).

* Tested Environment: Frontend >= v1.37.11, base >= 0.12.3
* Dependencies: NumPy >= 2.3.5, Pillow >= 12.1.0, SciPy >= 1.16.3, scikit-image >= 0.26.0
