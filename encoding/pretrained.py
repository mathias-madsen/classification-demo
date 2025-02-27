import numpy as np
from scipy.ndimage import zoom
from onnxruntime import InferenceSession
from typing import List, Dict
from time import perf_counter

import tempfile
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_onnx_cache_directory() -> str:
    return str(tempfile.gettempdir())


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
            path: str = "encoding/optimized_model.onnx",
            downsampling_factor: int = 3,
            keys: List[str] = DEFAULT_KEYS,
            ):
        # providers=[
        #     ("OpenVINOExecutionProvider", {"cache_dir": _get_onnx_cache_directory()}),
        #     ("CUDAExecutionProvider", {}),
        #     ("CPUExecutionProvider", {}),
        # ]
        providers = [
            'OpenVINOExecutionProvider',
            'CPUExecutionProvider',
            ]
        super().__init__(path, providers=providers)
        self.path = path
        self.keys = keys
        self.output_nodes = self.get_outputs()
        self.full_output_names = [n.name for n in self.output_nodes]
        self.short_output_names = [shorten(n) for n in self.full_output_names]
        self.downsampling_factor = downsampling_factor
        self.build_times = []
        self.run_times = []
        self.squeeze_times = []
        self.concat_times = []
        self.downsample_times = []
    
    def reset_times(self):
        self.build_times = []
        self.run_times = []
        self.squeeze_times = []
        self.concat_times = []
        self.downsample_times = []

    def compute_time_stats(self):
        nframes = len(self.build_times)
        assert nframes > 0
        slists = {
            "build": self.build_times,
            "run": self.run_times,
            "squeeze": self.squeeze_times,
            "concat": self.concat_times,
            "downsample": self.downsample_times,
        }
        return [(k, np.mean(v), np.std(v)) for k, v in slists.items()]

    def __repr__(self) -> str:
        return ("OnnxModel(%r, downsampling_factor=%s, keys=%r)" %
                (self.path, self.downsampling_factor, self.keys))

    def compute_output_dict(self, image: np.ndarray) -> np.ndarray:
        output_list = self.run(self.full_output_names, build_feed(image))
        return {k: v[0] for k, v in zip(self.short_output_names, output_list)}

    def __call__(self, uint8_rgb_image: np.ndarray) -> np.ndarray:

        start = perf_counter()
        feed = build_feed(uint8_rgb_image)
        self.build_times.append(perf_counter() - start)

        start = perf_counter()
        output_list = self.run(self.full_output_names, feed)
        self.run_times.append(perf_counter() - start)

        start = perf_counter()
        output_dict = {k: v[0] for k, v in zip(self.short_output_names, output_list)}
        self.squeeze_times.append(perf_counter() - start)

        start = perf_counter()
        all_features = np.concatenate([output_dict[k].flatten() for k in self.keys])
        self.concat_times.append(perf_counter() - start)

        start = perf_counter()
        downsampled_features = all_features[::self.downsampling_factor]
        self.downsample_times.append(perf_counter() - start)

        return downsampled_features


if __name__ == "__main__":

    import os
    import shutil
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    from gui.opencv_camera import OpenCVCamera
    from classification.dataset import EncodingData
    from classification.discriminator import BiGaussianDiscriminator
    from gaussians.moments_tracker import MomentsTracker, combine

    model = OnnxModel(downsampling_factor=5)
    camera = OpenCVCamera(0)

    rootdir = "/tmp/delete_me/"
    if os.path.isdir(rootdir):
        shutil.rmtree(rootdir)
    
    rgb = camera.read_mirrored_rgb()
    dataset = EncodingData(model, rootdir)
    dim = dataset.compute_dimensions(rgb)  # and set tracker

    plt.ion()
    figure, axes = plt.subplots(figsize=(4, 3))
    plot = axes.imshow(rgb)
    figure.tight_layout()
    plt.pause(1e-4)
    plt.show()

    discriminator = BiGaussianDiscriminator()

    try:
        for eps_idx in range(6):
            if eps_idx >= 2:
                prior = MomentsTracker(np.zeros(dim), np.eye(dim), 10.0)
                stats1 = combine(dataset.class_episode_stats[1] + [prior])
                stats2 = combine(dataset.class_episode_stats[1] + [prior])
                discriminator.set_stats(stats1, stats2)
            print("Episode index:", eps_idx)
            class_number = 1 + (eps_idx % 2)  # classes are called 1 or 2
            dataset.initialize_recording(class_number)
            model.reset_times()
            for frame_idx in tqdm(range(100), leave=False):
                rgb = camera.read_mirrored_rgb()
                latent = dataset.record_frame(rgb)
                if eps_idx >= 2:
                    logprobs = discriminator(latent)
                plot.set_data(rgb)
                plt.pause(1e-4)
                if not plt.fignum_exists(figure.number):
                    break
            if not plt.fignum_exists(figure.number):
                break
            for n, m, s in model.compute_time_stats():
                print("%s: %.5f ms ± %.5f ms" % (n, 1000.0 * m, 1000.0 * s))
            print("")
    except KeyboardInterrupt:
        pass

    camera.release()
    shutil.rmtree(rootdir)