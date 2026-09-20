import numpy as np
import scipy
import torch
import torch.fft as fft
from PIL import Image


def center_crop(im: Image) -> Image:
    # Get dimensions
    width, height = im.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    # Crop the center of the image
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


def linear_interpolation(l1: torch.Tensor, l2: torch.Tensor, size: int = 5, weights=None) -> torch.Tensor:
    l1 = l1.unsqueeze(0).repeat_interleave(size, 0)
    l2 = l2.unsqueeze(0).repeat_interleave(size, 0)
    if weights is None:
        weights = torch.linspace(0, 1, size).to(device=l1.device, dtype=l1.dtype)

    result = torch.lerp(
        l1,
        l2,
        append_dims(
            weights,
            l1.ndim,
        ),
    )
    result = result.transpose(0, 1).squeeze(0)
    return result


def spherical_interpolation(l1: torch.Tensor, l2: torch.Tensor, size: int = 5) -> torch.Tensor:
    """
    Spherical interpolation

    Args:
        l1: Starting vector: (1, *)
        l2: Final vector: (1, *)
        size: int, number of interpolation points including l1 and l2

    Returns:
        Interpolated vectors: (size, *)
    """
    assert l1.shape == l2.shape, "shapes of l1 and l2 must match"
    res = []
    for i in range(size):
        t = i / (size - 1)
        li = slerp(l1, l2, t)
        res.append(li)
    res = torch.cat(res, dim=0)
    return res


def slerp(v0: torch.Tensor, v1: torch.Tensor, t, threshold=0.9995):
    """
    Spherical linear interpolation
    Args:
        v0: Starting vector
        v1: Final vector
        t: Float value between 0.0 and 1.0
        threshold: Threshold for considering the two vectors as colinear. Not recommended to alter this.
    Returns:
        Interpolation vector between v0 and v1
    """
    assert v0.shape == v1.shape, "shapes of v0 and v1 must match"

    # Normalize the vectors to get the directions and angles
    v0_norm: torch.Tensor = torch.norm(v0, dim=-1)
    v1_norm: torch.Tensor = torch.norm(v1, dim=-1)

    v0_normed: torch.Tensor = v0 / v0_norm.unsqueeze(-1)
    v1_normed: torch.Tensor = v1 / v1_norm.unsqueeze(-1)

    # Dot product with the normalized vectors
    dot: torch.Tensor = (v0_normed * v1_normed).sum(-1)
    dot_mag: torch.Tensor = dot.abs()

    # if dp is NaN, it's because the v0 or v1 row was filled with 0s
    # If absolute value of dot product is almost 1, vectors are ~colinear, so use torch.lerp
    gotta_lerp: torch.Tensor = dot_mag.isnan() | (dot_mag > threshold)
    can_slerp: torch.Tensor = ~gotta_lerp

    t_batch_dim_count: int = max(0, t.dim() - v0.dim()) if isinstance(t, torch.Tensor) else 0
    t_batch_dims = t.shape[:t_batch_dim_count] if isinstance(t, torch.Tensor) else torch.Size([])
    out: torch.Tensor = torch.zeros_like(v0.expand(*t_batch_dims, *[-1] * v0.dim()))

    # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
    if gotta_lerp.any():
        lerped: torch.Tensor = torch.lerp(v0, v1, t)

        out: torch.Tensor = lerped.where(gotta_lerp.unsqueeze(-1), out)

    # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
    if can_slerp.any():
        # Calculate initial angle between v0 and v1
        theta_0: torch.Tensor = dot.arccos().unsqueeze(-1)
        sin_theta_0: torch.Tensor = theta_0.sin()
        # Angle at timestep t
        theta_t: torch.Tensor = theta_0 * t
        sin_theta_t: torch.Tensor = theta_t.sin()
        # Finish the slerp algorithm
        s0: torch.Tensor = (theta_0 - theta_t).sin() / sin_theta_0
        s1: torch.Tensor = sin_theta_t / sin_theta_0
        slerped: torch.Tensor = s0 * v0 + s1 * v1

        out: torch.Tensor = slerped.where(can_slerp.unsqueeze(-1), out)

    return out


def generate_beta_tensor(size: int, alpha: float = 3, beta: float = 3) -> torch.Tensor:
    """
    Assume size as n
    Generates a PyTorch tensor of values [x0, x1, ..., xn-1] for the Beta distribution
    where each xi satisfies F(xi) = i/(n-1) for the CDF F of the Beta distribution.

    Args:
        size (int): The number of values to generate.
        alpha (float): The alpha parameter of the Beta distribution.
        beta (float): The beta parameter of the Beta distribution.

    Returns:
        torch.Tensor: A tensor of the inverse CDF values of the Beta distribution.
    """
    # Generating the inverse CDF values
    prob_values = [i / (size - 1) for i in range(size)]
    inverse_cdf_values = scipy.stats.beta.ppf(prob_values, alpha, beta)
    # Converting to a PyTorch tensor
    return torch.tensor(inverse_cdf_values, dtype=torch.float32)


def fourier_filter(x: torch.Tensor, y: torch.Tensor, threshold) -> torch.Tensor:
    r"""
    FFT
    """
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

    mask[..., crow - threshold[0] : crow + threshold[0], ccol - threshold[1] : ccol + threshold[1]] = 0
    x_freq = torch.where(mask == 0, x_freq, y_freq)

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(original_dtype)
