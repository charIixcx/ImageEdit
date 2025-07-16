"""Terminal interface for previewing and applying image effects."""
import cv2
import numpy as np
import argparse
from typing import Tuple, Dict
from PyQt6.QtCore import Qt
from PyQt6 import QtWidgets, QtGui
from features.effects import (
    load_midas_model,
    generate_depth_map,
    prepare_displacement,
    apply_bokeh_effect,
    apply_particles_effect,
    PRESETS,
    N_FRAMES,
)

def generate_effect_preview(photo_path: str, depth_arr: np.ndarray, effect_name: str) -> np.ndarray:
    """
    Generates a preview of a specific effect and returns it as a NumPy array.
    Args:
        photo_path: Path to the input photo.
        depth_arr: Depth map as a NumPy array.
        effect_name: Name of the effect to apply.
    Returns:
        Preview image as a NumPy array.
    """
    photo_bgr = cv2.imread(photo_path)
    if photo_bgr is None:
        raise FileNotFoundError(f"Could not read image at {photo_path}")

    # Apply the selected effect
    if effect_name == "bokeh":
        img = apply_bokeh_effect(photo_bgr, depth_arr, focus_depth=0.5, strength=0.5)
    elif effect_name == "particles":
        img = apply_particles_effect(photo_bgr, depth_arr, frame_idx=0, total_frames=N_FRAMES, density=0.0005, size=2, speed=0.02)
    elif effect_name == "displacement":
        dx, dy = prepare_displacement(depth_arr, PRESETS["fullwrap"])
        # Normalize displacement maps
        dx = np.clip(dx, 0, photo_bgr.shape[1] - 1)
        dy = np.clip(dy, 0, photo_bgr.shape[0] - 1)
        img = cv2.remap(photo_bgr, dy.astype(np.float32), dx.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    elif effect_name == "distortion":
        img = apply_distortion_effect(photo_bgr, depth_arr, PRESETS["fullwrap"])
    else:
        raise ValueError(f"Unknown effect: {effect_name}")

    # Ensure output is valid uint8 and within [0,255]
    if img.dtype != np.uint8:
        print(f"Converting output to uint8 for {effect_name}, original dtype was {img.dtype}")
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        # Sometimes remap returns float even if input is uint8
        if np.issubdtype(img.dtype, np.floating):
            print(f"Warning: Output for {effect_name} is float, clipping and converting to uint8")
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def apply_distortion_effect(photo_bgr: np.ndarray, depth_arr: np.ndarray, preset: Dict[str, float]) -> np.ndarray:
    """
    Applies a distortion effect to an image using a depth map and displacement maps.
    Args:
        photo_bgr: Original image in BGR format (NumPy array).
        depth_arr: Depth map as a NumPy array.
        preset: Preset dictionary containing displacement parameters.
    Returns:
        Distorted image as a NumPy array.
    """
    # Generate displacement maps
    dx, dy = prepare_displacement(depth_arr, preset)

    # Print for debug
    print("distortion dx min/max:", dx.min(), dx.max())
    print("distortion dy min/max:", dy.min(), dy.max())

    # Normalize displacement maps
    dx = np.clip(dx, 0, photo_bgr.shape[1] - 1)  # Clip to valid width range
    dy = np.clip(dy, 0, photo_bgr.shape[0] - 1)  # Clip to valid height range

    # Remap the image using the displacement maps
    distorted_image = cv2.remap(
        photo_bgr,
        dy.astype(np.float32),
        dx.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    # Ensure result is uint8 and values are in [0,255]
    if distorted_image.dtype != np.uint8:
        print("Converting distortion result to uint8")
        distorted_image = np.clip(distorted_image, 0, 255).astype(np.uint8)
    return distorted_image

def display_gallery(photo_path: str, depth_arr: np.ndarray):
    """
    Opens a temporary gallery to display previews of all effects.
    Args:
        photo_path: Path to the input photo.
        depth_arr: Depth map as a NumPy array.
    """
    app = QtWidgets.QApplication([])

    # Create a main window
    gallery_window = QtWidgets.QWidget()
    gallery_window.setWindowTitle("Effect Previews Gallery")
    layout = QtWidgets.QGridLayout(gallery_window)

    # Generate previews for all effects
    effects = ["bokeh", "particles", "displacement", "distortion"]
    for i, effect_name in enumerate(effects):
        preview = generate_effect_preview(photo_path, depth_arr, effect_name)
        preview_pixmap = numpy_to_qpixmap(preview)

        # Create a vertical layout for the image and its caption
        effect_layout = QtWidgets.QVBoxLayout()

        # Create a label to display the preview
        label = QtWidgets.QLabel()
        label.setPixmap(preview_pixmap)
        label.setScaledContents(True)
        label.setFixedSize(300, 300)  # Set a fixed size for each preview
        effect_layout.addWidget(label)

        # Add a caption below the preview
        caption = QtWidgets.QLabel(effect_name.capitalize())
        caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        effect_layout.addWidget(caption)

        # Add the vertical layout to the grid
        layout.addLayout(effect_layout, i // 2, i % 2)

    gallery_window.setLayout(layout)
    gallery_window.show()
    app.exec()

def numpy_to_qpixmap(np_array: np.ndarray) -> QtGui.QPixmap:
    """
    Converts a NumPy array (representing an image) to a QPixmap.
    Handles grayscale, RGB, and RGBA images.
    """
    if np_array.ndim == 2:
        np_array = cv2.cvtColor(np_array, cv2.COLOR_GRAY2RGB)
    
    height, width, channel = np_array.shape
    bytes_per_line = channel * width
    
    if channel == 3:  # BGR to RGB
        rgb_image = cv2.cvtColor(np_array, cv2.COLOR_BGR2RGB)
        qimage = QtGui.QImage(rgb_image.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
    elif channel == 4:  # BGRA to RGBA
        rgba_image = cv2.cvtColor(np_array, cv2.COLOR_BGRA2RGBA)
        qimage = QtGui.QImage(rgba_image.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGBA8888)
    else:
        raise ValueError(f"Unsupported channel count: {channel}. Expected 2, 3, or 4.")
        
    return QtGui.QPixmap.fromImage(qimage)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preview or apply simple image effects"
    )
    parser.add_argument("--photo", required=True, help="Path to the input photo")
    parser.add_argument(
        "--gallery",
        action="store_true",
        help="Open a gallery to view previews of all effects",
    )
    parser.add_argument(
        "--effect",
        choices=["bokeh", "particles", "displacement", "distortion"],
        help="Effect to apply and save",
    )
    parser.add_argument(
        "--output",
        help="Output path for the processed image when using --effect",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading MiDaS model...")
    midas, transform, device = load_midas_model()

    print(f"Generating depth map for {args.photo}...")
    depth_arr = generate_depth_map(args.photo, midas, transform, device)

    if args.gallery:
        display_gallery(args.photo, depth_arr)
        return

    if args.effect:
        result = generate_effect_preview(args.photo, depth_arr, args.effect)
        output_path = args.output or f"{args.effect}_output.png"
        cv2.imwrite(output_path, result)
        print(f"Saved {args.effect} result to {output_path}")
    else:
        print("No action specified. Use --gallery or --effect.")


if __name__ == "__main__":
    main()
