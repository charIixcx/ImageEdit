# Stub implementations for missing features
import numpy as np
import cv2
from typing import Tuple, Dict

# Constants
PRESETS = {"fullwrap": {"scale": 1.0}}
TILE_SIZE = 32
FONT_PATHS = {}
FPS = 30
N_FRAMES = 1


def load_midas_model():
    """Stub for loading MiDaS model."""
    return None, None, "cpu"


def generate_depth_map(photo_path: str, midas, transform, device):
    """Generate a fake depth map using image luminance."""
    image = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(photo_path)
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.float32) / 255.0


def create_pink_halftone_texture(size: Tuple[int, int]):
    return np.zeros((size[1], size[0], 3), dtype=np.uint8)


def make_collage_word_strip(*args, **kwargs):
    return np.zeros((100, 100, 3), dtype=np.uint8)


def prepare_displacement(depth_map: np.ndarray, preset: Dict[str, float]):
    h, w = depth_map.shape[:2]
    dx = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    dy = np.tile(np.arange(h, dtype=np.float32).reshape(h, 1), (1, w))
    return dx, dy


def apply_bokeh_effect(
    image: np.ndarray, depth_map: np.ndarray, focus_depth: float, strength: float
):
    return cv2.GaussianBlur(image, (5, 5), 0)


def apply_particles_effect(
    image: np.ndarray,
    depth_map: np.ndarray,
    frame_idx: int,
    total_frames: int,
    density: float,
    size: int,
    speed: float,
):
    return image


def apply_wave_effect(
    image: np.ndarray,
    amplitude: float = 10.0,
    wavelength: float = 20.0,
    direction: str = "horizontal",
) -> np.ndarray:
    """Apply a simple sine wave distortion to the image."""

    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    if direction == "horizontal":
        dx = x.astype(np.float32)
        dy = y + amplitude * np.sin(2 * np.pi * x / wavelength)
    else:
        dx = x + amplitude * np.sin(2 * np.pi * y / wavelength)
        dy = y.astype(np.float32)

    dx = np.clip(dx, 0, w - 1)
    dy = np.clip(dy, 0, h - 1)

    return cv2.remap(
        image,
        dx.astype(np.float32),
        dy.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def apply_swirl_effect(
    image: np.ndarray,
    strength: float = 1.5,
    radius: float = 120.0,
    center: Tuple[float, float] | None = None,
) -> np.ndarray:
    """Apply a swirl distortion to the image using polar coordinates."""

    h, w = image.shape[:2]
    cx, cy = center if center else (w / 2.0, h / 2.0)

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_shifted = x - cx
    y_shifted = y - cy

    r = np.sqrt(x_shifted**2 + y_shifted**2)
    theta = np.arctan2(y_shifted, x_shifted)
    swirl_amount = strength * np.exp(-r / radius)
    theta += swirl_amount

    x_new = cx + r * np.cos(theta)
    y_new = cy + r * np.sin(theta)

    x_new = np.clip(x_new, 0, w - 1)
    y_new = np.clip(y_new, 0, h - 1)

    return cv2.remap(
        image,
        x_new.astype(np.float32),
        y_new.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
