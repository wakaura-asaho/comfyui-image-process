# ComfyUI Image Process

A collection of image-processing nodes designed to enhance and refine AI-generated images within ComfyUI workflows.

<p align="center">
<img src="https://github.com/wakaura-asaho/comfyui-image-process/blob/main/docs/logo.png" alt="Logo" style="display: block; margin: 0 auto; text-align: center;">
</p>

---

## What's New in v1.2.0

### Filename Prefix Token System

All save nodes now resolve dynamic tokens inside the **Filename Prefix** field, independently from ComfyUI core.

| Token | Output | Example |
|---|---|---|
| `%date:FORMAT%` | Date/time with custom format | `%date:yyyy-MM-dd%` → `2026-06-28` |
| `%year%` | 4-digit year | `2026` |
| `%month%` | 2-digit month | `06` |
| `%day%` | 2-digit day | `28` |
| `%hour%` | 2-digit hour (24 h) | `21` |
| `%minute%` | 2-digit minute | `05` |
| `%second%` | 2-digit second | `00` |
| `%width%` | Image width in pixels | `1024` |
| `%height%` | Image height in pixels | `1024` |

**`%date:FORMAT%` notation** uses `yyyy`, `yy`, `MM`, `dd`, `HH`, `hh`, `mm`, `ss`: e.g. `photos/%date:yyyy-MM-dd%/shot` produces `photos/2026-06-28/shot`.

![File Token Insertion](https://github.com/wakaura-asaho/comfyui-image-process/blob/main/docs/filename_token.png)

**Right-click "Insert Filename Token"**: any save node with a `filename_prefix` widget now shows a nested context menu.

### ICO Format Support

Two new nodes for saving Windows icon files:

- **Save Image (ICO)**: feeds a standard image batch; each frame is auto-validated against the seven legal ICO sizes and bundled into one `.ico` file.
- **Save Image Advanced (ICO)**: dedicated per-size slots (16 / 24 / 32 / 48 / 64 / 128 / 256 px), each with an optional alpha-mask input, plus color-depth, compression, and sort-by-size controls.

![Example workflow: Save Image ICO](https://github.com/wakaura-asaho/comfyui-image-process/blob/main/docs/save_image_ico.png)

An example workflow is included at [`workflows/example_workflow_imagesv_ico.json`](workflows/example_workflow_imagesv_ico.json).

> [!WARNING] Renaming the class name of `Save Image (Advanced)`
> Discovered in version 0.24.1, an official node with the same class name is introduced; to avoid the node from being confused upon fetching the schema, the internal class name has been renamed to `SaveImageAdvancedCustom`.

---

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
* **Filename Prefix Tokens:** Use `%date:FORMAT%`, `%year%`, `%month%`, `%width%`, etc. in the prefix field. Right-click the node for the **Insert Filename Token** helper menu.

### 6. Dedicated Format Savers

Format-specific savers with simple and advanced variants.

![Example_workflow](https://github.com/wakaura-asaho/comfyui-image-process/blob/main/docs/save_image_simple.png)

* **JPG:** Quick saves or advanced EXIF/DPI control.
* **BMP:** Standard saves or 32-bit alpha/DPI support.
* **TIFF:** Standard saves or advanced compression/EXIF options.
* **TGA:** Standard saves or RLE compression/alpha support.
* **AVIF:** Standard saves or advanced quality/speed control.

> [!WARNING]
> ComfyUI may not display TIFF or TGA files properly in the preview pane. The preview is a PNG copy, but the actual file is saved in your chosen format.

> [!TIP]
> `tiff_ccitt` is for black-and-white documents or line-art. For photography, use `tiff_lzw`, `tiff_deflate`, or `tiff_adobe_deflate`.

### 7. Save Image (ICO) / Save Image Advanced (ICO)

Saves images as Windows `.ico` files.

![Example workflow – Save Image ICO](https://github.com/wakaura-asaho/comfyui-image-process/blob/main/docs/save_image_ico.png)

**Save Image (ICO)**: simple batch input:

* Accepts any batch of images; each frame is automatically checked against valid ICO dimensions (16, 24, 32, 48, 64, 128, 256 px square).
* Valid frames are bundled into a single `.ico` file.
* Invalid frames are optionally saved as individual `_INVALID.png` files.

**Save Image Advanced (ICO)**: per-size slot inputs:

* Seven dedicated image slots, one per standard ICO size, each paired with an optional alpha-mask input.
* **Color Depth:** `8bit` (256-color palette), `16bit` (RGB), or `32bit` (RGBA).
* **Compression:** `none` (standard BMP frames), `ZIP` (PNG-compressed frames, Windows Vista+).
* **Join Alpha Channel / Invert Alpha:** Applies the per-slot mask as the alpha channel, with optional inversion.
* **Sort by Size:** Reorders frames from smallest to largest before bundling.
* **Save to Input Folder:** Copies the `.ico` to the input folder for immediate reuse.
* **Save Invalid as PNG:** Saves any size-mismatched images as fallback PNGs.

> [!TIP]
> Connect only the sizes you need: all slots are optional. The node raises an error only when no slot is connected at all.

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
* `image_helper.py`: Shared save helpers including the filename prefix resolver.
* `web/image_process.js`: Frontend extensions (widget visibility, filename token menu).
* `workflows/`: Example workflows.
* `docs/`: Screenshots referenced in this README.

## Usage Tips

> [!TIP]
> **Saturation Threshold:** Start at 0.08. Use lower values (e.g., 0.05) for more aggressive correction, or higher values (e.g., 0.15) to be conservative.

> [!TIP]
> **Smoothing Kernel Size:** Use 3-5 for subtle corrections. Try 7-15 for pronounced smoothing, though it may blur regions.

> [!TIP]
> **Alpha Preservation:** Enable **Preserve Alpha** with an optional mask to maintain transparency.

> [!TIP]
> **Filename tokens:** Right-click any save node and choose **Insert Filename Token** to append a token without typing. Combine tokens freely, e.g. `renders/%date:yyyy-MM-dd%/%hour%-%minute%`.

## Compatible Versions and Notices

Designed for modern ComfyUI versions (V3 nodes).

* Tested Environment: Frontend >= v1.37.11, base >= 0.12.3
* Dependencies: NumPy >= 2.3.5, Pillow >= 12.1.0, SciPy >= 1.16.3, scikit-image >= 0.26.0
