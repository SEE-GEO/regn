"""
======================
regn.data.augmentation
======================

This module provides functions to transform GMI images to simulate
the distortion in different parts of the swath.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

_A = 1.1501621502635013
_B = 1.2277700179533668

def atan(x):
    """
    Atan-fit which maps fractional pixel indices from [-1.0, 1.0]
    to fractional GMI cross-track distance.
    """
    return _A * np.arctan(_B * (x))

def tan(y):
    """
    Inverse of the tan function
    """
    return np.tan(y / _A) / _B

def scan_offset(i_f):
    """
    Quadratic fit to the pixel offset in along-track
    dimension.
    """
    return 30 * i_f**2

def get_pixel_coordinates(p_i, p_o):
    """
    Get pixel coordinates for given positions of the input and
    output windows.

    Args:
        p_i: The relative position of the  input window
             in a 221 x 221 pixel full-swath image, with
             -1 corresponding to the one extreme and 1 to
             the other extreme.
        p_o: The relative position of the 128 x 128 output window  with
             -1 corresponding to the last pixel and 1 to the first pixel.

    Returns:
        128 pixel indices of the with respect to the 221x221 input image.
    """
    i_c = 110

    d_p = 2.0 * 128 / 221
    o_p = 2.0 * (221 - 128) / 221
    o_l = -1.0 + 0.5 * (p_o + 1.0) * o_p
    o_r = o_l + d_p

    i_o = np.linspace(o_l, o_r, 128)
    y_o = atan(i_o)
    dy = (y_o[-1] - y_o[0])

    d_y_i = 0.5 * (p_i + 1.0) * (2.041 - dy) - 1.0205
    y_i = y_o - y_o[0] + d_y_i

    i_i = 110 + 110 * tan(y_i)

    return i_i

def get_scan_offsets(p_i, p_o):
    """
    Calculate the relative scan offsets across the swath.

    Args:
        p_i: The relative position of the  input window in a 221 x 221 pixel
             full-swath image, with -1 corresponding to the one extreme and 1
             to the other extreme.
        p_o: The relative position of the 128 x 128 output window  with
             -1 corresponding to the last pixel and 1 to the first pixel.

    Returns:
        128 scan offsets relative to start position of the central scan.
    """

    d_p = 2.0 * 128 / 221
    o_p = 2.0 * (221 - 128) / 221
    o_l = -1.0 + 0.5 * (p_o + 1.0) * o_p
    o_r = o_l + d_p

    i_l = -1.0 + 0.5 * (p_i + 1.0) * o_p
    i_r = i_l + d_p

    o_i = scan_offset(np.linspace(i_l, i_r, 128))
    o_o = scan_offset(np.linspace(o_l, o_r, 128))

    scan_coords = 110 - 64 + o_o - o_i
    return scan_coords

def extract_subscene(img_data, p_i, p_o):
    """
    Extract subscenene from a full-swath (221 x 221) of GMI observations.

    Args:
        p_i: The relative position of the  input window in a 221 x 221 pixel
             full-swath image, with -1 corresponding to the one extreme and 1
             to the other extreme.
        p_o: The relative position of the 128 x 128 output window  with
             -1 corresponding to the last pixel and 1 to the first pixel.

    Returns:
        A 128 x 128 data patch extracted from the input data which was
        transformed to mimic the perspective distortion that affects GMI
        retrievals.
    """
    indices = np.arange(221)
    interpolator = RegularGridInterpolator((indices, indices), img_data)

    s_offsets = get_scan_offsets(p_i, p_o)
    s_coords = s_offsets.reshape(1, -1) + np.arange(128).reshape(-1, 1)
    p_coords = get_pixel_coordinates(p_i, p_o).reshape(1, -1)
    p_coords = np.broadcast_to(p_coords, s_coords.shape)

    coords = np.concatenate([s_coords.reshape(-1, 1), p_coords.reshape(-1, 1)], axis=1)
    return interpolator(coords).reshape((128, 128) + img_data.shape[2:])

def mask_stripe(obs, p_o):
    """
    Masks high-frequency channels over or close to the ground truth to teach the
    network how to handle pixels where the high-frequency observations are missing.

    Args:
        obs: 15 x 128 x 128 GMI image.
        p_o: Fractional location of the output window.
    """

    d = p_o * 46
    c = 110
    i_l = int(c + d - 10)
    i_r = int(c + d + 10)

    obs[..., 9:, i_l:i_r] = np.nan



