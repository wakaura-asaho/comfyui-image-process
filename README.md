# ComfyUI Image Process

A collection of image-processing nodes designed to enhance and refine AI-generated images within ComfyUI workflows. (may add more nodes to this repo in the future)

<p align="center">
<img src="https://github.com/wakaura-asaho/comfyui-image-process/blob/main/docs/logo.png" alt="Logo" style="display: block; margin: 0 auto; text-align: center;">
</p>

## Nodes Included

### 1. Color Artifact Normalizer

An image-processing node that corrects achromatic color instability commonly found in AI-generated images.

* **Artifact Detection:** Automatically identifies pixels with near-zero saturation that represent color artifacts.
* **Saturation Threshold:** Adjustable threshold parameter to fine-tune which pixels are considered artifacts (range: 0.0 - 1.0).
* **Smart Normalization:** Snaps the hue of artifact pixels to 0 and zeroes out saturation, converting them to neutral grey.
* **Optional Smoothing:** Apply a configurable smoothing filter to blend corrected areas seamlessly with the rest of the image.
* **Alpha Channel Preservation:** Optionally preserve the alpha channel during processing for transparent images.
* **Dual Output:** Returns both the processed image and alpha mask separately for maximum flexibility.

### 2. Save Image Advanced

An extended node that supports outputting the images with alpha channels and metadata.

* **Disable Metadata:** Normally, every image sent to the output folder will have the workflow embedded. You can toggle this option to discard the data and free up some disk space.
* **Join Alpha Channel:** Effectively the same as the official node `Join Image with Alpha`. You can connect a mask to the node to clip the image.
* **Compression Level:** The compression level to use when saving images as PNG (0-9).
* **Quality:** The quality to use when saving images as JPEG or WebP (1-100).
* **DPI:** Set the DPI (dots per inch) metadata embedded in the saved file (range: 1–600, default: 300). Applies to all formats.
* **TIFF Compression:** When saving as TIFF, choose the compression algorithm: `none`, `tiff_lzw`, `tiff_deflate`, `tiff_adobe_deflate`, `packbits`, `jpeg`, `tiff_jpeg`, or `tiff_ccitt`.
* **Format:** The format to save the image in. (Currently supports: `JPG`, `PNG`, `WebP`, `TIFF`)
* **Save to Input Folder:** When enabled, the saved file is also copied to the ComfyUI input folder (same filename and subfolder), making it immediately available for use in subsequent workflows.

> [!WARNING]
> If choose to save as TIFF file, ComfyUI may not be able to display the image properly.
> The image shown in the preview pane is a copy of the PNG version of the image. The actual file saved will be in the format and the folder path you choose.

> [!TIP]
> `tiff_ccitt` is only useful for scanned documents, fax images, or text/line-art that is already black and white.
> For photographic content, you might want to use `tiff_lzw`, `tiff_deflate`, or `tiff_adobe_deflate` instead.

---

## Example Usage

The `Color Artifact Normalizer` is particularly useful for:

* **Removing color noise** from AI-generated images that exhibit hue instability in near-white or near-black regions.
* **Improving consistency** in images generated with models prone to producing desaturated pixels with unstable hues.
* **Preserving transparency** in images that require alpha channel information while correcting color artifacts.

![Example_workflow](https://github.com/wakaura-asaho/comfyui-image-process/blob/main/docs/example.png)

To use the node in your workflow:

1. Connect an image to the **Image** input.
2. (Optional) Connect an alpha mask to the **Alpha Mask** input if you want to provide a custom alpha channel.
3. Adjust the **Saturation Threshold** to control sensitivity (lower values = more pixels corrected).
4. Enable **Smooth Transitions** to reduce visible artifacts in corrected regions.
5. Set the **Smoothing Kernel Size** for the intensity of the smoothing effect (higher values = more blending).
6. Toggle **Preserve Alpha** to keep or discard the alpha channel in the output.

The node will output the corrected image and its alpha channel separately.

The `Save Image Advanced` can be used at the end of the workflow to save your images with additional preprocessing tweaks:

![Example_workflow](https://github.com/wakaura-asaho/comfyui-image-process/blob/main/docs/save_image_adv.png)

The workflow screenshot above uses a custom node to remove the background from the generated image, sending the mask to the node to save the image with an alpha channel.

When the `Disable Metadata` is set to True, you will get a smaller file due to discarded metadata, and ComfyUI will no longer be able to read the workflow from the image. 

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
