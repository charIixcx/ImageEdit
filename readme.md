# ImageEdit

This project provides a minimal command line interface for experimenting with
image effects using OpenCV and PyQt6. Depth maps are generated using the
[MiDaS](https://github.com/isl-org/MiDaS) model which produces more accurate
results than the previous luminance based placeholder. If the loader cannot
download the model it falls back to a local copy located in ``models/midas``.
Clone the MiDaS repository into this folder if you want offline support.
results than the previous luminance based placeholder.

## Running

```
python app_terminal.py --photo PATH/TO/IMAGE --gallery
```

Launch the GUI to pick an image and preview effects:

```
python app_gui.py
```

Use `--effect` to apply a single effect and save the result:

```
python app_terminal.py --photo input.jpg --effect swirl --output swirl.jpg
```

## Available Effects

- **bokeh** – blurs the background using a Gaussian filter.
- **particles** – placeholder for a particle overlay effect.
- **displacement** – remaps pixels based on the generated depth map.
- **distortion** – applies a depth driven distortion warp.
- **wave** – warps the image with a sine wave function.
- **swirl** – rotates pixels around the centre with exponential falloff.
