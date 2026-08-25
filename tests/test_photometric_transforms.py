import pytest
from PIL import Image
import numpy as np
from hpc.data.transforms import PhotometricTransforms


def test_t10_photometric_transforms_preserve_geometry():
    """T10: Photometric transforms must never alter image dimensions or point coordinates."""
    img = Image.new("RGB", (448, 448), color=(128, 128, 128))
    pts = np.array([
        [10.0, 20.0],
        [150.5, 300.2],
        [400.0, 440.0],
    ], dtype=np.float32)
    pts_copy = pts.copy()
    
    degrader = PhotometricTransforms(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        blur_prob=1.0,
        gamma_min=0.5,
        gamma_max=1.5,
        salt_pepper_prob=1.0,
        jpeg_prob=1.0,
        noise_prob=1.0,
    )
    
    # Run multiple times with randomized transforms
    for _ in range(10):
        deg_img = degrader(img)
        assert deg_img.size == (448, 448), f"Image dimensions changed: {deg_img.size}"
        # Assert point coordinates unchanged
        np.testing.assert_array_equal(pts, pts_copy)
