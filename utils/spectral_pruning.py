"""Frequency-informed approximation using Speedy-Splat's pixelwise score kernel."""
import torch


def spectral_detail_weight(image, gamma):
    # Radial high-frequency emphasis; detach prevents gradients through the weight.
    luma = (image[:3] * image.new_tensor([0.299, 0.587, 0.114])[:, None, None]).sum(0)
    height, width = luma.shape
    fy = torch.fft.fftfreq(height, device=image.device)
    fx = torch.fft.rfftfreq(width, device=image.device)
    yy, xx = torch.meshgrid(fy, fx, indexing='ij')
    radius = torch.sqrt(xx.square() + yy.square())
    radius = radius / radius.max().clamp_min(1e-12)
    detail = torch.fft.irfft2(torch.fft.rfft2(luma, norm='ortho') * radius.pow(gamma / 2),
                             s=(height, width), norm='ortho').square()
    return detail / detail.mean().clamp_min(1e-12)


def fuse_scores(spatial, spectral, weight):
    def normalize(x):
        return x / torch.linalg.vector_norm(x).clamp_min(1e-12)
    return normalize(spatial) + weight * normalize(spectral)
