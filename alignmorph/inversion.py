"""Fixed-point inversion of the actual deterministic DDIM forward step."""

import math

import torch


def ddim_linear_coefficients(scheduler, timestep):
    """Return A, B such that x_previous = A*x_current + B*model_output.

    This contract requires eta=0 and disabled sample clipping/thresholding.
    The coefficients use the forward scheduler's prediction parameterization.
    """
    if scheduler.config.clip_sample or getattr(scheduler.config, "thresholding", False):
        raise ValueError(
            "Linear DDIM inversion requires unclipped/unthresholded samples"
        )
    t = int(timestep)
    if not 0 <= t < scheduler.config.num_train_timesteps:
        raise ValueError("timestep outside scheduler range")
    stride = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    previous = t - stride
    alpha = float(scheduler.alphas_cumprod[t])
    alpha_previous = float(
        scheduler.alphas_cumprod[previous]
        if previous >= 0
        else scheduler.final_alpha_cumprod
    )
    a, p = math.sqrt(alpha), math.sqrt(alpha_previous)
    b, q = math.sqrt(1 - alpha), math.sqrt(1 - alpha_previous)
    prediction = scheduler.config.prediction_type
    if prediction == "v_prediction":
        return p * a + q * b, q * a - p * b
    if prediction == "epsilon":
        return p / a, q - p * b / a
    if prediction == "sample":
        if b == 0:
            raise ValueError(
                "Sample prediction at zero noise is not invertible by this solver"
            )
        return q / b, p - q * a / b
    raise ValueError(f"Unsupported prediction type: {prediction}")


@torch.no_grad()
def invert_ddim_step(
    previous, predict, scheduler, timestep, *, iterations=5, relaxation=0.5
):
    """Solve x_current from x_previous; predict always receives x_current at t."""
    if iterations < 1 or not 0 < relaxation <= 1:
        raise ValueError("Positive iterations and relaxation in (0,1] are required")
    a, b = ddim_linear_coefficients(scheduler, timestep)
    if abs(a) < 1e-8:
        raise ValueError("DDIM step is singular under this parameterization")
    current = previous.clone()
    if b == 0:
        return previous / a
    for _ in range(iterations):
        prediction = predict(current, timestep)
        proposal = (previous.float() - b * prediction.float()) / a
        current = torch.lerp(current.float(), proposal, relaxation).to(previous.dtype)
        if not torch.isfinite(current).all():
            raise FloatingPointError("Non-finite fixed-point inversion state")
    return current
