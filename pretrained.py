import numpy as np
from scipy.ndimage import zoom
from onnxruntime import InferenceSession


def crop_to_aspect_ratio(image, width_to_height=320/256):
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
    

def zoom_to_size(image, new_height=256, new_width=320):
    old_height, old_width, _  = image.shape
    if not np.isclose(old_width/old_height, new_width/new_height, atol=0.01):
        raise ValueError("Bad size: %sx%s" % (old_height, old_width))
    factor = new_height / old_height
    return zoom(image, [factor, factor, 1.0])


def shorten(name):
    """ Remove anything before and including first '/' from a string. """
    return name.split("/", maxsplit=1)[1] if "/" in name else name


def build_feed(image):
    """ Make a stereo input frame out of an image. """
    image = image.astype(np.float32) / 255.
    cropped = crop_to_aspect_ratio(image)
    resized = zoom_to_size(cropped)
    onebatch = resized[None,]
    return {"vision1": onebatch,
            "vision2": onebatch[:, ::-1, ::-1, :]}


class OnnxModel(InferenceSession):
    
    def __init__(self, path):
        super().__init__(path, providers=['CPUExecutionProvider'])
        self.output_nodes = self.get_outputs()
        self.full_output_names = [n.name for n in self.output_nodes]
        self.short_output_names = [shorten(n) for n in self.full_output_names]

    def compute_output_dict(self, image):
        output_list = self.run(self.full_output_names, build_feed(image))
        return {k: v[0] for k, v in zip(self.short_output_names, output_list)}

    def compute_feature_vector(self, image):
        output_dict = self.compute_output_dict(image)
        keys = [
            'Eyx1', 'Varsyx1', 'Covyx1',
            'Eyx2', 'Varsyx2', 'Covyx2',
            ]
        all_features = np.concatenate([output_dict[k].flatten() for k in keys])
        return all_features[::3]
        # return all_features


path = "model.onnx"
session = OnnxModel(path)


def encode(uint8_rgb_image):
    """ Compute a latent vector for a raw RGB image. """
    return session.compute_feature_vector(uint8_rgb_image)


if __name__ == "__main__":

    from scipy.misc import face
    raccoon = face()
    image = raccoon[:256, :320, :]
    features = encode(image)
    print(features.shape)  # (5120,)
