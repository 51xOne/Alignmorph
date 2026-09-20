"""Image loading, interpolation weights, and Fourier noise injection."""

import numpy as np
import scipy
import torch
import torch.fft as fft
from PIL import Image


def center_crop(im: Image) -> Image:
    width, height = im.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    im = im.crop((left, top, right, bottom))
    return im


def load_im_from_path(im_path, image_resolution) -> torch.Tensor:
    if not isinstance(im_path, list):
        im_path = [im_path]
    im_list = []
    for path in im_path:
        image = Image.open(path).convert("RGB")
        if image_resolution[0] == image_resolution[1]:
            image = center_crop(image)
        image = image.resize((image_resolution[1], image_resolution[0]), Image.LANCZOS)
        image = np.array(image) / 255.0 * 2.0 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        im_list.append(image)
    image = torch.stack(im_list, dim=0)
    return image


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]


def linear_interpolation(
    l1: torch.Tensor, l2: torch.Tensor, size: int = 5, weights=None
) -> torch.Tensor:
    l1 = l1.unsqueeze(0).repeat_interleave(size, 0)
    l2 = l2.unsqueeze(0).repeat_interleave(size, 0)
    if weights is None:
        weights = torch.linspace(0, 1, size).to(device=l1.device, dtype=l1.dtype)

    result = torch.lerp(l1, l2, append_dims(weights, l1.ndim))
    result = result.transpose(0, 1).squeeze(0)
    return result


def generate_beta_tensor(size: int, alpha: float = 3, beta: float = 3) -> torch.Tensor:
    """Sample evenly spaced quantiles of a Beta distribution."""
    prob_values = [i / (size - 1) for i in range(size)]
    inverse_cdf_values = scipy.stats.beta.ppf(prob_values, alpha, beta)
    return torch.tensor(inverse_cdf_values, dtype=torch.float32)


def fourier_filter(x: torch.Tensor, y: torch.Tensor, threshold) -> torch.Tensor:
    """Keep x's central frequencies and replace its outer frequencies with y's."""
    if isinstance(threshold, int):
        threshold = [threshold, threshold]
    original_dtype = x.dtype
    x = x.float()
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    y = y.float()
    y_freq = fft.fftn(y, dim=(-2, -1))
    y_freq = fft.fftshift(y_freq, dim=(-2, -1))

    H, W = x_freq.shape[-2:]
    mask = torch.ones(x_freq.shape).to(x.device)

    crow, ccol = H // 2, W // 2

    mask[
        ...,
        crow - threshold[0] : crow + threshold[0],
        ccol - threshold[1] : ccol + threshold[1],
    ] = 0
    x_freq = torch.where(mask == 0, x_freq, y_freq)

    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(original_dtype)
