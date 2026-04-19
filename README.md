# ComfyUI-Image-Process

A collection of image processing nodes designed to enhance and refine AI-generated images in ComfyUI workflows. (may add more nodes to this repo in the future)

<p align="center">
<img src="https://github.com/wakaura-asaho/comfyui-image-process/blob/main/docs/logo.png" alt="Logo" style="display: block; margin: 0 auto; text-align: center;">
</p>

## Nodes Included

### 1. Color Artifact Normalizer

An intelligent image processing node that fixes achromatic color instability commonly found in AI-generated images.

* **Artifact Detection:** Automatically identifies pixels with near-zero saturation that represent color artifacts.
* **Saturation Threshold:** Adjustable threshold parameter to fine-tune which pixels are considered artifacts (range: 0.0 - 1.0).
* **Smart Normalization:** Snaps the hue of artifact pixels to 0 and zeroes out saturation, converting them to neutral grey.
* **Optional Smoothing:** Apply a configurable smoothing filter to blend corrected areas seamlessly with the rest of the image.
* **Alpha Channel Preservation:** Optionally preserve the alpha channel during processing for transparent images.
* **Dual Output:** Returns both the processed image and alpha mask separately for maximum flexibility.

---

## Example Usage

The Color Artifact Normalizer is particularly useful for:

* **Removing color noise** from AI-generated images that exhibit hue instability in near-white or near-black regions.
* **Improving consistency** in images generated with models prone to producing desaturated pixels with unstable hues.
* **Preserving transparency** in images that require alpha channel information while correcting color artifacts.

![Example_workflow](https://github.com/wakaura-asaho/comfyui-dynamic-selector/blob/main/docs/example.png)

To use the node in your workflow:

1. Connect an image to the **Image** input.
2. (Optional) Connect an alpha mask to the **Alpha Mask** input if you want to provide a custom alpha channel.
3. Adjust the **Saturation Threshold** to control sensitivity (lower values = more pixels corrected).
4. Enable **Smooth Transitions** to reduce visible artifacts in corrected regions.
5. Set the **Smoothing Kernel Size** for the intensity of the smoothing effect (higher values = more blending).
6. Toggle **Preserve Alpha** to keep or discard the alpha channel in the output.

The node will output the corrected image and its alpha channel separately.

---

## Installation

### Method 1: ComfyUI Manager (Recommended)

1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager).
2. Click on **"Install via Git URL"**.
3. Paste the URL of this repository: `https://github.com/wakaura-asaho/comfyui-image-process`
4. Restart ComfyUI.

### Method 2: Manual Installation

1. Open a terminal in your `ComfyUI/custom_nodes` folder.
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
> **Package Dependency Notice:** Consider using `pip install -r requirements.txt --upgrade-strategy only-if-needed` to install only missing or outdated packages to review your currently installed packages and avoid unnecessary version conflicts or environment breakage.

---

## File Structure

* `image_process.py`: Backend logic and node class definitions.

## Usage Tips

> [!TIP]
> **Saturation Threshold:** Start with the default value of 0.08 and adjust based on your image. Lower values (e.g., 0.05) will be more aggressive, while higher values (e.g., 0.15) will be more conservative.

> [!TIP]
> **Smoothing Kernel Size:** For subtle corrections, use smaller kernel sizes (3-5). For more pronounced smoothing, try larger sizes (7-15), but be aware this may blur the affected regions.

> [!TIP]
> **Alpha Preservation:** If you're working with transparent images, enable **Preserve Alpha** and optionally provide an alpha mask to ensure your transparency information is maintained through processing.

## Compatible Versions and Notices

The nodes are designed for use with modern versions of ComfyUI and are written as V3 nodes.

* Tested Environment: Frontend >= v1.37.11, base >= 0.12.3
* Dependencies: NumPy >= 2.3.5, Pillow >= 12.1.0, SciPy >= 1.16.3
