import PIL.Image as pili
from pathlib import Path
import numpy as np


class Image:
    def __init__(self, imagepath):
        p = Path(imagepath).expanduser()
        if not p.exists():
            raise Exception("Image does not exist")
        image = pili.open(p)
        rgbimage = image.convert("RGB")
        self._repr = np.asarray(rgbimage)

    def toimage(self) -> pili.Image:
        return pili.fromarray(self._repr)

    @property
    def array(self):
        return self._repr
