# Stub implementations for missing features
import numpy as np
import cv2
from typing import Tuple, Dict

# Constants
PRESETS = {
    "fullwrap": {"scale": 1.0}
}
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

def apply_bokeh_effect(image: np.ndarray, depth_map: np.ndarray, focus_depth: float, strength: float):
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_particles_effect(image: np.ndarray, depth_map: np.ndarray, frame_idx: int, total_frames: int, density: float, size: int, speed: float):
    return image
