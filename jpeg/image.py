from pathlib import Path
from PIL import Image as pili
import numpy as np
from skimage import color


def dct8_line(n):
    return np.fromfunction(lambda i: np.cos((i * 2 * n + n) * np.pi / 16), (8,))


def build_dct8():
    return 0.5 * np.array(
        [
            np.full((8,), 1 / np.sqrt(2)),
            dct8_line(1),
            dct8_line(2),
            dct8_line(3),
            dct8_line(4),
            dct8_line(5),
            dct8_line(6),
            dct8_line(7),
        ]
    )


class MacroBlock:
    q_mat = np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ]
    )

    quv_mat = np.array(
        [
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
        ]
    )

    d8 = build_dct8()

    def __init__(self, block, q, space="RGB"):
        self._array = block
        self._coefs = MacroBlock._zigzag(
            MacroBlock._quantize(self._spectrum(), q, space)
        )
        self._ratio = len(self._coefs) / 64

    def uncompress(self, q, space="RGB"):
        return MacroBlock._unspectrum(
            MacroBlock._unquantize(MacroBlock._unzigzag(self.coefs, 8), q, space)
        )

    def _spectrum(self):
        return (
            MacroBlock.d8 @ (np.array(self._array, dtype="int") - 128) @ MacroBlock.d8.T
        )

    @staticmethod
    def _unspectrum(spectrum):
        return np.array(
            np.clip(np.round(MacroBlock.d8.T @ spectrum @ MacroBlock.d8) + 128, 0, 255),
            dtype="uint8",
        )

    @staticmethod
    def _quantize(spectrum, q, space):
        alpha = 5000 / q if q < 50 else 200 - 2 * q
        qmat = MacroBlock.quv_mat if space == "YUV" else MacroBlock.q_mat
        return np.round(spectrum / np.round((qmat * alpha + 50) / 100))

    @staticmethod
    def _unquantize(m, q, space):
        alpha = 5000 / q if q < 50 else 200 - 2 * q
        qmat = MacroBlock.quv_mat if space == "YUV" else MacroBlock.q_mat
        return m * np.round((qmat * alpha + 50) / 100)

    @staticmethod
    def _zigzag(m):
        # Ugly zigzag line
        z = np.concatenate(
            [
                np.diagonal(m[::-1, :], i)[:: (1 - 2 * (i % 2))]
                for i in range(1 - m.shape[0], m.shape[0])
            ]
        )
        # Get a filter, make it so we only lose the last zeros from zigzag.
        if z[-1] == 0:
            f = z != 0
            if True in f:
                f[: len(f) - f[::-1].tolist().index(True) - 1] = True
            return z[f]
        else:
            return z

    @staticmethod
    def _unzigzag(z, s):
        z = np.pad(z, (0, 64 - len(z)), constant_values=0)
        res = np.zeros((s, s), dtype="int")
        i = 0
        for j in range(1 - s, 1):
            diag_size = s - abs(j)
            np.fill_diagonal(
                res[::-1, :][abs(j) :, :], z[i : i + diag_size][:: (1 - 2 * (j % 2))]
            )
            i += diag_size
        for j in range(1, s):
            diag_size = s - abs(j)
            np.fill_diagonal(
                res[::-1, :][:, j:], z[i : i + diag_size][:: (1 - 2 * (j % 2))]
            )
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
    def __init__(self, array=None, height=1, width=1, space="RGB"):
        # space can be RGB, YUV or Grayscale
        self._array = (
            np.array(array.copy(), dtype="uint8")
            if array is not None
            else np.full((height, width, 3), 255, dtype="uint8")
        )
        self._space = space

        if space == "Grayscale" and len(self._array.shape) != 2:
            raise Exception("Grayscale images only accept one channel")
        elif (
            space != "Grayscale"
            and len(self._array.shape) != 3
            and self._array.shape[2] != 3
        ):
            raise Exception("Images shapes should be of the form (n, m, 3)")

    def channel0(self):
        return MyImage(self._array[:, :, 0], space="Grayscale")

    def channel1(self):
        return MyImage(self._array[:, :, 1], space="Grayscale")

    def channel2(self):
        return MyImage(self._array[:, :, 2], space="Grayscale")

    def to_image(self) -> pili.Image:
        return pili.fromarray(self.array)

    @staticmethod
    def from_image(path):
        p = Path(path).expanduser()
        if not p.exists():
            raise Exception("Image does not exist")
        image = pili.open(p).convert("RGB")
        return MyImage(np.asarray(image))

    def __copy__(self):
        return MyImage(self._array, space=self.space)

    def grayscale(self):
        res = self.array[..., :3] @ [0.2989, 0.5870, 0.1140]
        return MyImage(res, space="Grayscale")

    @staticmethod
    def RGB_to_YUV(img):
        if img.space != "RGB":
            raise Exception(f"Image space should be RGB instead of {img.space}")
        yuv = color.rgb2yuv(img.array)
        yuv[:, :, 0] *= 255
        yuv[:, :, 1] += 0.436
        yuv[:, :, 1] *= 255 / 0.872
        yuv[:, :, 2] += 0.615
        yuv[:, :, 2] *= 255 / 1.23
        return MyImage(np.clip(np.round(yuv), 0, 255), space="YUV")

    @staticmethod
    def YUV_to_RGB(img):
        if img.space != "YUV":
            raise Exception(f"Image space should be YUV instead of {img.space}")
        arr = np.array(img.array, dtype="float64")
        arr[:, :, 0] /= 255
        arr[:, :, 1] /= 255 / 0.872
        arr[:, :, 1] -= 0.436
        arr[:, :, 2] /= 255 / 1.23
        arr[:, :, 2] -= 0.615
        return MyImage(np.clip(np.round(color.yuv2rgb(arr) * 255), 0, 255))

    def get_macro_arrays(self, q, space="RGB"):
        arr = self.array

        height_pad = 8 - self.height % 8 if self.height % 8 != 0 else 0
        width_pad = 8 - self.width % 8 if self.width % 8 != 0 else 0
        arr = np.pad(arr, [(0, height_pad), (0, width_pad)], mode="symmetric")

        split_height = arr.shape[0] / 8
        split_width = arr.shape[1] / 8
        return (
            np.array(
                [
                    [MacroBlock(y, q, space) for y in np.split(x, split_width, axis=1)]
                    for x in np.split(arr, split_height)
                ]
            ),
            width_pad,
            height_pad,
        )

    @staticmethod
    def reassemble_macroblocks(macroblocks, q, mode="RGB"):
        return MyImage(
            np.concatenate(
                np.concatenate(
                    np.array(
                        [
                            [mb2.uncompress(q, mode) for mb2 in mb1]
                            for mb1 in macroblocks
                        ]
                    ),
                    axis=1,
                ),
                axis=1,
            ),
            space="Grayscale",
        )

    @staticmethod
    def grayscale_compress(img, q):
        if img.space != "Grayscale":
            raise Exception("Cannot perform this operation on non-grayscale image")
        return img.get_macro_arrays(q)

    @staticmethod
    def grayscale_uncompress(macro_arrays, q):
        res = MyImage.reassemble_macroblocks(macro_arrays[0], q)
        return res.trim(macro_arrays[1], macro_arrays[2])

    @staticmethod
    def rgb_compress(img, q):
        r = img.channel0()
        g = img.channel1()
        b = img.channel2()
        return [
            r.get_macro_arrays(q),
            g.get_macro_arrays(q),
            b.get_macro_arrays(q),
        ]

    @staticmethod
    def rgb_uncompress(macro_arrays, q):
        r = MyImage.reassemble_macroblocks(macro_arrays[0][0], q)
        g = MyImage.reassemble_macroblocks(macro_arrays[1][0], q)
        b = MyImage.reassemble_macroblocks(macro_arrays[2][0], q)
        r = r.trim(macro_arrays[0][1], macro_arrays[0][2])
        g = g.trim(macro_arrays[1][1], macro_arrays[1][2])
        b = b.trim(macro_arrays[2][1], macro_arrays[2][2])
        return MyImage(np.stack((r.array, g.array, b.array), axis=-1))

    @staticmethod
    def yuv_compress(img, q, downsampling="4:4:4"):
        yuv = MyImage.RGB_to_YUV(img)
        y = yuv.channel0()
        u = yuv.channel1()
        v = yuv.channel2()
        if downsampling == "4:2:2":
            u = u.downsampling(2, 1)
            v = v.downsampling(2, 1)
        elif downsampling == "4:2:0":
            u = u.downsampling(2, 2)
            v = v.downsampling(2, 2)
        return [
            y.get_macro_arrays(q, "YUV"),
            u.get_macro_arrays(q, "YUV"),
            v.get_macro_arrays(q, "YUV"),
        ]

    @staticmethod
    def yuv_uncompress(macro_arrays, q, downsampling="4:4:4"):
        y = MyImage.reassemble_macroblocks(macro_arrays[0][0], q, "YUV")
        u = MyImage.reassemble_macroblocks(macro_arrays[1][0], q, "YUV")
        v = MyImage.reassemble_macroblocks(macro_arrays[2][0], q, "YUV")
        y = y.trim(macro_arrays[0][1], macro_arrays[0][2])
        u = u.trim(macro_arrays[1][1], macro_arrays[1][2])
        v = v.trim(macro_arrays[2][1], macro_arrays[2][2])

        if downsampling == "4:2:2":
            u = u.upsampling(2, 1)
            v = v.upsampling(2, 1)
        elif downsampling == "4:2:0":
            u = u.upsampling(2, 2)
            v = v.upsampling(2, 2)
        u = u.trim(u.width - y.width, u.height - y.height)
        v = v.trim(v.width - y.width, v.height - y.height)

        stack = np.stack((y.array, u.array, v.array), axis=-1)
        return MyImage.YUV_to_RGB(MyImage(stack, space="YUV"))

    def trim(self, width, height):
        if width and height:
            return MyImage(self.array[:-height, :-width], space=self.space)
        elif width:
            return MyImage(self.array[:, :-width], space=self.space)
        elif height:
            return MyImage(self.array[:-height, :], space=self.space)
        else:
            return self

    def upsampling(self, sizex, sizey):
        f = lambda i, j: self._array[i // sizey, j // sizex]
        arr = np.fromfunction(f, (self.height * sizey, self.width * sizex), dtype=int)
        return MyImage(arr, space=self.space)

    def downsampling(self, sizex, sizey):
        arr = self.array[::sizey, ::sizex]
        return MyImage(arr, space=self.space)

    @property
    def array(self):
        return self._array

    @property
    def shape(self):
        return self._array.shape

    @property
    def height(self):
        return self._array.shape[0]

    @property
    def width(self):
        return self._array.shape[1]

    @property
    def space(self):
        return self._space
