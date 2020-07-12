from pathlib import Path
from PIL import Image as pili
import numpy as np


class MyImage:
    def __init__(self, height=1, width=1, array=None, space='RGB'):
        self._repr = array.copy() if array is not None else np.full(255, shape=(height, width, 3))
        self._space = space

    def to_image(self) -> pili.Image:
        return pili.fromarray(self._repr)

    def save(self, path):
        pili.fromarray(self._repr).save(path)

    def __copy__(self):
        return MyImage(array=self._repr)

    @staticmethod
    def grayscale(img):
        get_grey = lambda r,g,b: int(r * 0.2989 + g * 0.5870 + b * 0.1140)
        grayscaling = lambda x : np.array([get_grey(x[0], x[1], x[2]), get_grey(x[0], x[1], x[2]), get_grey(x[0], x[1], x[2])], dtype='uint8')
        res = np.apply_along_axis(grayscaling, 2, img.array)
        return MyImage(array=res)

    @staticmethod
    def from_image(path):
        p = Path(path).expanduser()
        if not p.exists():
            raise Exception("Image does not exist")
        image = pili.open(p).convert("RGB")
        return MyImage(array=np.asarray(image))

    @property
    def array(self):
        return self._repr

    @property
    def shape(self):
        return self._repr.shape

    @property
    def space(self):
        return self._repr.space
