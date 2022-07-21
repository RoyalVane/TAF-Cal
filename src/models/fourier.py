import torch
import matplotlib.pyplot as plt
import numpy as np


def rec_amp_phase(img, device='cpu', phase_factor=1, amp_factor=1):
    phase, ampl = fft(img, device)
    phase = get_phase_img(phase, phase_factor)
    ampl = get_amp_img(ampl, amp_factor)

    return phase, ampl


def fft(img, device='cpu', eps=True, eps_val=1e-7):
    # get fft of the image
    fft_img = torch.rfft(img.clone(), signal_ndim=2, onesided=False).to(device)

    # extract phase and amplitude from the fft
    phase, ampl = extract_phase_amlp(fft_img.clone(), eps=eps, eps_val=eps_val)

    return phase, ampl


def extract_phase_amlp(fft_img, eps=True, eps_val=1e-7):
    # fft_img: size should be batch_size * 3 * h * w * 2
    ampl = fft_img[:, :, :, :, 0]**2 + fft_img[:, :, :, :, 1]**2
    if eps:
        ampl = torch.sqrt(ampl + eps)
        phase = torch.atan2(fft_img[:, :, :, :, 1] + eps_val,
                            fft_img[:, :, :, :, 0] + eps_val)
    else:
        ampl = torch.sqrt(ampl)
        phase = torch.atan2(fft_img[:, :, :, :, 1], fft_img[:, :, :, :, 0])
    return phase, ampl


def ifft(phase, ampl, device='cpu'):
    # recompse fft of image
    fft_img = torch.zeros(
        (phase.shape[0], phase.shape[1], phase.shape[2], phase.shape[3], 2),
        dtype=torch.float).to(device)
    fft_img[:, :, :, :, 0] = torch.cos(phase) * ampl
    fft_img[:, :, :, :, 1] = torch.sin(phase) * ampl

    # get the recomposed image
    _, _, imgH, imgW = phase.size()
    image = torch.irfft(fft_img,
                        signal_ndim=2,
                        onesided=False,
                        signal_sizes=[imgH, imgW])
    return image


def get_phase_img(phase, factor=1, device='cpu'):
    # recompse fft of image
    fft_img = torch.zeros(
        (phase.shape[0], phase.shape[1], phase.shape[2], phase.shape[3], 2),
        dtype=torch.float).to(device)
    fft_img[:, :, :, :, 0] = torch.cos(phase.clone()) * factor
    fft_img[:, :, :, :, 1] = torch.sin(phase.clone()) * factor

    # get the recomposed image
    _, _, imgH, imgW = phase.size()
    image = torch.irfft(fft_img,
                        signal_ndim=2,
                        onesided=False,
                        signal_sizes=[imgH, imgW])
    return image


def get_amp_img(ampl, factor=1, device='cpu'):
    # recompse fft of image
    fft_img = torch.zeros(
        (ampl.shape[0], ampl.shape[1], ampl.shape[2], ampl.shape[3], 2),
        dtype=torch.float).to(device)
    fft_img[:, :, :, :, 0] = ampl.clone() * factor
    fft_img[:, :, :, :, 1] = ampl.clone() * factor

    # get the recomposed image
    _, _, imgH, imgW = ampl.size()
    image = torch.irfft(fft_img,
                        signal_ndim=2,
                        onesided=False,
                        signal_sizes=[imgH, imgW])
    return image
