from pathlib import Path
from PIL import Image as pili
import numpy as np


class MacroBlock:
    def __init__(self, block):
        self._block = block



class MyImage:
    def __init__(self, height=1, width=1, array=None, space='RGB', grayscale=False):
        self._repr = array.copy() if array is not None else np.full((height, width, 3), 255, dtype='uint8')
        self._space = space
        self._grayscale = grayscale

    def to_image(self) -> pili.Image:
        return pili.fromarray(self._repr)

    def __copy__(self):
        return MyImage(array=self._repr)

    @staticmethod
    def grayscale(img):
        res = img.array[..., :3] @ [0.2989, 0.5870, 0.1140]
        return MyImage(array=res, grayscale=True)

    @staticmethod
    def from_image(path):
        p = Path(path).expanduser()
        if not p.exists():
            raise Exception("Image does not exist")
        image = pili.open(p).convert("RGB")
        return MyImage(array=np.asarray(image))

    @staticmethod
    def get_macro_blocks(img):
        arr = img.array
        if img.height % 8 != 0:
            arr = np.concatenate((arr, np.zeros(shape=(8 - img.height % 8, img.width, 3), dtype='uint8')), axis=0)
        if img.width % 8 != 0:
            arr = np.concatenate((arr, np.zeros(shape=(arr.shape[0], 8 - img.width % 8, 3), dtype='uint8')), axis=1)

        split_height = arr.shape[0] / 8
        split_width = arr.shape[1] / 8
        return np.array([np.split(x, split_width, axis=1) for x in np.split(arr, split_height)])

    @staticmethod
    def from_macro_blocks(macro_blocks):
        return MyImage(array=np.concatenate(np.concatenate(macro_blocks, axis=1), axis=1))


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
        return self._repr.space
