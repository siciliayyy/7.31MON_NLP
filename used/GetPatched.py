"""
Patchify.py
"""
import numbers
from typing import Tuple, Union, cast

import numpy as np
from numpy.lib.stride_tricks import as_strided

Imsize = Union[Tuple[int, int], Tuple[int, int, int]]


def view_as_windows(arr_in, window_shape, step=1):
    # -- basic checks on arguments
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = (
                                (np.array(arr_in.shape) - np.array(window_shape)) // np.array(step)
                        ) + 1

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out


def patchify(image: np.ndarray, patch_size: Imsize, step: int = 1) -> np.ndarray:
    """
    Split a 2D or 3D image into small patches given the patch size.

    Parameters
    ----------
    image: the image to be split. It can be 2d (m, n) or 3d (k, m, n)
    patch_size: the size of a single patch
    step: the step size between patches

    Examples
    --------
     image = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
     patches = patchify(image, (2, 2), step=1)  # split image into 2*3 small 2*2 patches.
     assert patches.shape == (2, 3, 2, 2)
     reconstructed_image = unpatchify(patches, image.shape)
     assert (reconstructed_image == image).all()
    """
    return view_as_windows(image, patch_size, step)


def unpatchify(patches: np.ndarray, imsize: Imsize) -> np.ndarray:
    """
    Merge patches into the original image

    Parameters
    ----------
    patches: the patches to merge. It can be patches for a 2d image (k, l, m, n)
             or 3d volume (i, j, k, l, m, n)
    imsize: the size of the original image or volume

    Examples
    --------
     image = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
     patches = patchify(image, (2, 2), step=1)  # split image into 2*3 small 2*2 patches.
     assert patches.shape == (2, 3, 2, 2)
     reconstructed_image = unpatchify(patches, image.shape)
     assert (reconstructed_image == image).all()
    """

    assert len(patches.shape) / 2 == len(
        imsize
    ), "The patches dimension is not equal to the original image size"

    if len(patches.shape) == 4:
        return _unpatchify2d(patches, cast(Tuple[int, int], imsize))
    elif len(patches.shape) == 6:
        return _unpatchify3d(patches, cast(Tuple[int, int, int], imsize))
    else:
        raise NotImplementedError(
            "Unpatchify only supports a matrix of 2D patches (k, l, m, n)"
            f"or 3D volumes (i, j, k, l, m, n), but got: {patches.shape}"
        )


def _unpatchify2d(  # pylint: disable=too-many-locals
        patches: np.ndarray, imsize: Tuple[int, int]
) -> np.ndarray:
    assert len(patches.shape) == 4

    i_h, i_w = imsize
    image = np.zeros(imsize, dtype=patches.dtype)

    n_h, n_w, p_h, p_w = patches.shape

    s_w = 0 if n_w <= 1 else (i_w - p_w) / (n_w - 1)
    s_h = 0 if n_h <= 1 else (i_h - p_h) / (n_h - 1)

    # The step size should be same for all patches, otherwise the patches are unable
    # to reconstruct into a image
    if int(s_w) != s_w:
        raise NonUniformStepSizeError(i_w, n_w, p_w, s_w)
    if int(s_h) != s_h:
        raise NonUniformStepSizeError(i_h, n_h, p_h, s_h)
    s_w = int(s_w)
    s_h = int(s_h)

    i, j = 0, 0

    while True:
        i_o, j_o = i * s_h, j * s_w

        image[i_o: i_o + p_h, j_o: j_o + p_w] = patches[i, j]

        if j < n_w - 1:
            j = min((j_o + p_w) // s_w, n_w - 1)
        elif i < n_h - 1 and j >= n_w - 1:
            # Go to next row
            i = min((i_o + p_h) // s_h, n_h - 1)
            j = 0
        elif i >= n_h - 1 and j >= n_w - 1:
            # Finished
            break
        else:
            raise RuntimeError("Unreachable")

    return image


def _unpatchify3d(  # pylint: disable=too-many-locals
        patches: np.ndarray, imsize: Tuple[int, int, int]
) -> np.ndarray:
    assert len(patches.shape) == 6

    img_h, img_w, img_c = imsize
    image = np.zeros(imsize, dtype=patches.dtype)

    num_h, num_w, num_c, patch_h, patch_w, patch_c = patches.shape

    SLong_w = 0 if num_w <= 1 else (img_w - patch_w) / (num_w - 1)
    SLong_h = 0 if num_h <= 1 else (img_h - patch_h) / (num_h - 1)
    SLong_c = 0 if num_c <= 1 else (img_c - patch_c) / (num_c - 1)

    # The step size should be same for all patches, otherwise the patches are unable
    # to reconstruct into a image
    if int(SLong_w) != SLong_w:
        raise NonUniformStepSizeError(img_w, num_w, patch_w, SLong_w)
    if int(SLong_h) != SLong_h:
        raise NonUniformStepSizeError(img_h, num_h, patch_h, SLong_h)
    if int(SLong_c) != SLong_c:
        raise NonUniformStepSizeError(img_c, num_c, patch_c, SLong_c)

    SLong_w = int(SLong_w)
    SLong_h = int(SLong_h)
    SLong_c = int(SLong_c)

    H, W, C = 0, 0, 0

    while True:

        H_o, W_o, C_o = H * SLong_h, W * SLong_w, C * SLong_c

        image[H_o: H_o + patch_h, W_o: W_o + patch_w, C_o: C_o + patch_c] = patches[H, W, C]

        if C < num_c - 1:
            C = min((C_o + patch_c) // SLong_c, num_c - 1)
        elif W < num_w - 1 and C >= num_c - 1:
            W = min((W_o + patch_w) // SLong_w, num_w - 1)
            C = 0
        elif H < num_h - 1 and W >= num_w - 1 and C >= num_c - 1:
            H = min((H_o + patch_h) // SLong_h, num_h - 1)
            W = 0
            C = 0
        elif H >= num_h - 1 and W >= num_w - 1 and C >= num_c - 1:
            # Finished
            break
        else:
            raise RuntimeError("Unreachable")

    return image


class NonUniformStepSizeError(RuntimeError):
    def __init__(
            self, imsize: int, n_patches: int, patch_size: int, step_size: float
    ) -> None:
        super().__init__(imsize, n_patches, patch_size, step_size)
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.imsize = imsize
        self.step_size = step_size

    def __repr__(self) -> str:
        return f"Unpatchify only supports reconstructing image with a uniform step size for all patches. \
However, reconstructing {self.n_patches} x {self.patch_size}px patches to an {self.imsize} image requires {self.step_size} as step size, which is not an integer."

    def __str__(self) -> str:
        return self.__repr__()
