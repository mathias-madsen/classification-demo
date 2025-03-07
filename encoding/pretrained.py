import numpy as np
from scipy.ndimage import zoom
from onnxruntime import InferenceSession
from typing import List, Dict


DEFAULT_KEYS = [
    'Eyx1',
    'Varsyx1',
    'Covyx1',
    # 'Eyx2',
    # 'Varsyx2',
    # 'Covyx2',
    ]


def crop_to_aspect_ratio(
        image: np.ndarray,
        width_to_height: float = 320/256,
        ) -> np.ndarray:
    height, width, _ = image.shape
    if width/height > width_to_height:
        target_width = int(round(height * width_to_height))
        total_excess = width - target_width
        left = total_excess // 2
        right = total_excess - left
        return image[:, left:width - right, :]
    elif width/height < width_to_height:
        target_height = int(round(width / width_to_height))
        total_excess = height - target_height
        top = total_excess // 2
        bottom = total_excess - top
        return image[top:height - bottom, :, :]
    else:
        return image
    

def zoom_to_size(
        image: np.ndarray,
        new_height: int = 256,
        new_width: int = 320,
        ) -> np.ndarray:
    old_height, old_width, _  = image.shape
    if not np.isclose(old_width/old_height, new_width/new_height, atol=0.01):
        raise ValueError("Bad size: %sx%s" % (old_height, old_width))
    factor = new_height / old_height
    return zoom(image, [factor, factor, 1.0], order=0)


def shorten(name: str) -> str:
    """ Remove anything before and including first '/' from a string. """
    return name.split("/", maxsplit=1)[1] if "/" in name else name


def build_feed(image: np.ndarray) -> Dict[str, np.ndarray]:
    """ Make a stereo input frame out of an image. """
    image = image.astype(np.float32) / 255.
    cropped = crop_to_aspect_ratio(image)
    resized = zoom_to_size(cropped)
    onebatch = resized[None,]
    return {"vision1": onebatch, "vision2": onebatch}


class OnnxModel(InferenceSession):
    
    def __init__(
            self,
            path: str = "encoding/model.onnx",
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
