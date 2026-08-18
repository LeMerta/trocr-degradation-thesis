"""
degradation.py

Functions to degrade images in different types and intensities. 
Contains four methods: AWGN, Gaussian blur, JPEG compression, and downscaling.

Each function takes a PIL Image and returns a degraded PIL Image.
"""

import io
import numpy as np
from PIL import Image, ImageFilter


def apply_awgn(image: Image.Image, sigma: float) -> Image.Image:
    """
    Adds Additive White Gaussian Noise to an image.

    Args:
        image:  PIL Image converted to RGB.
        sigma:  Standard deviation of the noise distribution (sigma > 0). 
                Higher values produce stronger noise.

    Returns:
        Degraded PIL Image.
    """
    img_array = np.array(image, dtype=np.float32)
    noise = np.random.normal(loc=0.0, scale=sigma, size=img_array.shape)
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)


def apply_gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """
    Applies Gaussian blur to an image.

    Args:
        image:  PIL Image converted to RGB.
        sigma:  Standard deviation of the Gaussian kernel in pixels (sigma > 0).
                Higher values produce stronger blurring.

    Returns:
        Degraded PIL Image.
    """
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """
    Applies JPEG compression artifacts to an image.

    Args:
        image:      PIL Image converted to RGB.
        quality:    JPEG quality level (1-95). 
                    Lower values produce stronger compression.

    Returns:
        Degraded PIL Image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


def apply_downscaling(image: Image.Image, target_height: int) -> Image.Image:
    """
    Downscales image to target_height then upscales back to original size.
    
    Args:
        image:         PIL Image converted to RGB.
        target_height: Target height in pixels to downscale to 
    """
    original_size = image.size
    scale_factor  = target_height / original_size[1]
    small_w = max(1, int(original_size[0] * scale_factor))
    return image.resize((small_w, target_height), resample=Image.NEAREST)
