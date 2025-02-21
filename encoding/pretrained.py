import numpy as np
from scipy.ndimage import zoom
from onnxruntime import InferenceSession
from typing import List, Dict
from time import perf_counter


DEFAULT_KEYS = [
    'Eyx1',
    'Varsyx1',
    'Covyx1',
    # 'Eyx2',
    # 'Varsyx2',
    # 'Covyx2',
    ]


def center_crop_image(
        image: np.ndarray,
        new_height: int = 256,
        new_width: int = 320,
        ):
    old_height, old_width, _  = image.shape
    # get vertical limits
    excess_height = old_height - new_height
    top = excess_height // 2
    bottom = old_height - excess_height + top
    # get horizontal limits
    excess_width = old_width - new_width
    left = excess_width // 2
    right = old_width - excess_width + left
    # crop
    return image[top:bottom, left:right, :]


def zoom_to_size(
        image: np.ndarray,
        new_height: int = 256,
        new_width: int = 320,
        ) -> np.ndarray:
    old_height, old_width, _  = image.shape
    factor_height = old_height // new_height
    factor_width = old_width // new_width
    factor = min(factor_height, factor_width)
    small_image = image[::factor, ::factor, :]
    return center_crop_image(small_image, new_height, new_width)


def shorten(name: str) -> str:
    """ Remove anything before and including first '/' from a string. """
    return name.split("/", maxsplit=1)[1] if "/" in name else name


def build_feed(image: np.ndarray) -> Dict[str, np.ndarray]:
    """ Make a stereo input frame out of an image. """
    image = image.astype(np.float32) / 255.
    resized = zoom_to_size(image)
    onebatch = resized[None,]
    return {"vision1": onebatch, "vision2": onebatch}


class OnnxModel(InferenceSession):
    
    def __init__(
            self,
            # path: str = "encoding/model.onnx",
            path: str = "encoding/optimized_model.onnx",
                downsampling_factor: int = 3,
            keys: List[str] = DEFAULT_KEYS,
            ):
        super().__init__(path, providers=['CPUExecutionProvider'])
        self.path = path
        self.keys = keys
        self.output_nodes = self.get_outputs()
        self.full_output_names = [n.name for n in self.output_nodes]
        self.short_output_names = [shorten(n) for n in self.full_output_names]
        self.downsampling_factor = downsampling_factor

    def __repr__(self) -> str:
        return ("OnnxModel(%r, downsampling_factor=%s, keys=%r)" %
                (self.path, self.downsampling_factor, self.keys))

    def compute_output_dict(self, image: np.ndarray) -> np.ndarray:
        output_list = self.run(self.full_output_names, build_feed(image))
        return {k: v[0] for k, v in zip(self.short_output_names, output_list)}

    def __call__(self, uint8_rgb_image: np.ndarray) -> np.ndarray:
        output_dict = self.compute_output_dict(uint8_rgb_image)
        all_features = np.concatenate([output_dict[k].flatten() for k in self.keys])
        return all_features[::self.downsampling_factor]


if __name__ == "__main__":

    model = OnnxModel()

    try:
        from scipy.misc import face  # scipy version v1.12 and older
    except:
        from scipy.datasets import face  # scipy version v1.10 and newer

    raccoon = face()
    image = raccoon[:256, :320, :]
    features = model(image)
    print(features.shape)  # (5120,)
