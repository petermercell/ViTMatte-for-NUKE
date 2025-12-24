# ViTMatte for NUKE (External Workflow)

A standalone workflow for using [ViTMatte](https://github.com/hustvl/ViTMatte) with The Foundry's Nuke via external Python scripts and Miniconda.

## ⚠️ Looking for an Integrated Solution?

If you want a **fully integrated Nuke plugin** with easy installation, check out [Rafael Perez's ViTMatte-for-Nuke](https://github.com/rafaelperez/ViTMatte-for-Nuke). His implementation:

- Installs via Nuke's Cattery system
- Runs directly inside Nuke as an Inference node
- Includes optimized memory handling for 4K on 8GB GPUs
- Has built-in trimap generation controls

**This repository** takes a different approach — it provides an **external command-line workflow** using Miniconda and Hugging Face Transformers. This is what I do best: practical scripts and Nuke gizmos that get the job done without complex plugin builds.

## What's Included

| File | Description |
|------|-------------|
| `vitmatte.py` | Main Python script for batch processing images through ViTMatte |
| `trimap.nk` | Nuke gizmo for converting alpha to trimap |
| `TRIMAP_NEW_240224.nk` | Updated trimap generation gizmo |
| `ALPHA_TO_CV2_ALPHA.nk` | Converts Nuke alpha to CV2-compatible format |
| `MASK_TO_1_OBJECT.nk` | Utility for mask processing |
| `X2_prep.v002.nk` | Preprocessing gizmo |

## Performance

- **2K Image**: ~8 seconds
- **4K Image**: ~90 seconds

## Installation

### 1. Install Miniconda

Download and install [Miniconda 3](https://docs.anaconda.com/free/miniconda/).

### 2. Create Environment & Install Dependencies

```bash
conda create -n HFTF python=3.10
conda activate HFTF
pip install transformers torch pillow
pip install opencv-python  # 4x faster than PIL for saving images
```

### 3. Download the Model

Download the ViTMatte model from [Hugging Face](https://huggingface.co/hustvl/vitmatte-small-composition-1k/tree/main).

### 4. Install CV2 for Nuke (for trimap gizmos)

See the included `how to install cv2 for nuke` file for instructions on getting OpenCV working inside Nuke for the trimap tools.

## Usage

### Prepare Your Files

1. Create three folders on your desktop (or adjust paths in script):
   - `IMAGE/` — source images
   - `TRIMAP/` — trimap images exported from Nuke
   - `OUTPUT/` — results will be saved here

### Generate Trimap in Nuke

1. Load your source image with an alpha channel (garbage matte)
2. Use `ALPHA_TO_CV2_ALPHA.nk` to convert alpha to CV2 format
3. Use `trimap.nk` or `TRIMAP_NEW_240224.nk` to generate the trimap
4. Export trimap sequence to your `TRIMAP/` folder

### Run ViTMatte

```bash
conda activate HFTF
cd C:\Users\WORKSTATION\Desktop\
python vitmatte.py
```

Enter the number of frames when prompted.

## What is a Trimap?

A trimap is a grayscale guidance image that tells ViTMatte where to focus:

- **White (255)** — Definite foreground (opaque)
- **Black (0)** — Definite background (transparent)  
- **Gray (128)** — Unknown region (needs matting)

The quality of your trimap directly affects the output matte quality.

## More Information

For visual examples and detailed walkthrough, visit: [petermercell.com/rvm-vs-vitmatte](https://petermercell.com/rvm-vs-vitmatte/)

## Related Tools

- [RobustVideoMatting for Nuke](https://github.com/hkaramian/RobustVideoMatting-For-Nuke) — Human segmentation without trimap
- [ViTMatte-for-Nuke by Rafael Perez](https://github.com/rafaelperez/ViTMatte-for-Nuke) — Integrated Cattery plugin (recommended for most users)

## License

This workflow is provided as-is for the Nuke community. The underlying ViTMatte model is subject to its own licensing terms — see [hustvl/ViTMatte](https://github.com/hustvl/ViTMatte) for details.

## Author

**Peter Mercell**  
[petermercell.com](https://www.petermercell.com)  


Article you could find here: https://petermercell.com/rvm-vs-vitmatte/
