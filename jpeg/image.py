from pathlib import Path
from PIL import Image as pili
from scipy.fftpack import dct, idct
import numpy as np


def dct8_line(n):
    return np.fromfunction(lambda i: np.cos((i * 2 * n + n) * np.pi / 16), (8,))


class MacroBlock:
    q_mat = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]])

    d8 = 0.5 * np.array([np.full((8,), 1 / np.sqrt(2)),
                         dct8_line(1),
                         dct8_line(2),
                         dct8_line(3),
                         dct8_line(4),
                         dct8_line(5),
                         dct8_line(6),
                         dct8_line(7)])

    def __init__(self, block, q):
        self._array = block
        self._coefs = MacroBlock._zigzag(MacroBlock._quantize(MacroBlock._spectrum(self), q))
        self._ratio = 64 / len(self._coefs) if len(self._coefs) != 0 else np.inf

    @staticmethod
    def uncompress(mb, q):
        return MacroBlock._unspectrum(MacroBlock._unquantize(MacroBlock._unzigzag(mb.coefs, 8), q))

    @staticmethod
    def _spectrum(mb):
        return MacroBlock.d8 @ (np.array(mb._array, dtype='int') - 128) @ MacroBlock.d8.T

    @staticmethod
    def _unspectrum(spectrum):
        return np.array(np.round(MacroBlock.d8.T @ spectrum @ MacroBlock.d8) + 128, dtype='uint8')

    @staticmethod
    def _quantize(spectrum, q):
        alpha = 5000 / q if q < 50 else 200 - 2 * q
        return np.array(np.round(spectrum / ((MacroBlock.q_mat * alpha + 50) / 100)), dtype='int')

    @staticmethod
    def _unquantize(m, q):
        alpha = 5000 / q if q < 50 else 200 - 2 * q
        return m * ((MacroBlock.q_mat * alpha + 50) / 100)

    @staticmethod
    def _zigzag(m):
        # Ugly zigzag line
        z = np.concatenate([np.diagonal(m[::-1, :], i)[::(1 - 2 * (i % 2))] for i in range(1 - m.shape[0], m.shape[0])])
        # Get a filter, make it so we only lose the last zeros from zigzag.
        if z[-1] == 0:
            f = z != 0
            if True in f:
                f[:len(f) - f[::-1].tolist().index(True) - 1] = True
            return z[f]
        else:
            return z

    @staticmethod
    def _unzigzag(z, s):
        z = np.pad(z, (0, 64 - len(z)), constant_values=0)
        res = np.zeros((s, s), dtype='int')
        i = 0
        for j in range(1 - s, 1):
            diag_size = s - abs(j)
            np.fill_diagonal(res[::-1, :][abs(j):, :], z[i:i + diag_size][::(1 - 2 * (j % 2))])
            i += diag_size
        for j in range(1, s):
            diag_size = s - abs(j)
            np.fill_diagonal(res[::-1, :][:, j:], z[i:i + diag_size][::(1 - 2 * (j % 2))])
            i += diag_size
        return res

    @property
    def array(self):
        return self._array

    @property
    def coefs(self):
        return self._coefs

    @property
    def ratio(self):
        return self._ratio


class MyImage:
    def __init__(self, height=1, width=1, array=None, space='RGB', grayscale=False):
        self._repr = np.array(array.copy(), dtype='uint8') if array is not None else np.full((height, width, 3), 255,
                                                                                             dtype='uint8')
        self._space = space
        self._grayscale = grayscale

    @staticmethod
    def to_image(img) -> pili.Image:
        return pili.fromarray(img.array)

    @staticmethod
    def from_image(path):
        p = Path(path).expanduser()
        if not p.exists():
            raise Exception("Image does not exist")
        image = pili.open(p).convert("RGB")
        return MyImage(array=np.asarray(image))

    def __copy__(self):
        return MyImage(array=self._repr)

    @staticmethod
    def grayscale(img):
        res = img.array[..., :3] @ [0.2989, 0.5870, 0.1140]
        return MyImage(array=res, grayscale=True)

    @staticmethod
    def get_macro_arrays(img, q):
        arr = img.array

        height_pad = 8 - img.height % 8 if img.height % 8 != 0 else 0
        width_pad = 8 - img.width % 8 if img.width % 8 != 0 else 0
        # arr = np.pad(arr, [(0, height_pad), (0, width_pad), (0, 0)], mode='symmetric')
        arr = np.pad(arr, [(0, height_pad), (0, width_pad)], mode='symmetric')

        split_height = arr.shape[0] / 8
        split_width = arr.shape[1] / 8
        return np.array(
            [[MacroBlock(y, q) for y in np.split(x, split_width, axis=1)] for x in np.split(arr, split_height)])

    @staticmethod
    def grayscale_compress(img, q):
        img = MyImage.grayscale(img)
        return MyImage.get_macro_arrays(img, q)

    @staticmethod
    def grayscale_uncompress(macro_arrays, q):
        return MyImage(array=np.concatenate(
            np.concatenate(np.array([[MacroBlock.uncompress(mb2, q) for mb2 in mb1] for mb1 in macro_arrays]), axis=1),
            axis=1))

    @staticmethod
    def from_macro_arrays(macro_arrays):
        return MyImage(array=np.concatenate(np.concatenate(macro_arrays, axis=1), axis=1))

    @property
    def array(self):
        return self._repr

    @property
    def shape(self):
        return self._repr.shape

    @property
    def height(self):
        return self._repr.shape[0]

    @property
    def width(self):
        return self._repr.shape[1]

    @property
    def space(self):
        return self._space
