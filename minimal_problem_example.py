import numpy as np
from onnxruntime import InferenceSession
from time import perf_counter


dim = 512
random_matrix_square_root = np.random.normal(size=(dim, dim))
random_matrix = random_matrix_square_root.T @ random_matrix_square_root

session = InferenceSession(
    "encoding/optimized_model.onnx",
    providers = ["CPUExecutionProvider"],
    )

output_names = [node.name for node in session.get_outputs()]

if __name__ == "__main__":

    from tqdm import tqdm

    from gui.opencv_camera import OpenCVCamera

    # open the camera feed and acquire a test image:
    camera = OpenCVCamera(0)

    times = {True: [], False: []}
    alltimes = []

    for iteration in tqdm(range(400)):

        run_numpy = iteration % 40 < 20

        start = perf_counter()
        rgb = camera.read_mirrored_rgb()
        image = rgb.astype(np.float32) / 255.
        feed = {
            "vision1": image[None, :256, :320, :],
            "vision2": image[None, :256, :320, :],
            }
        session.run(output_names, feed)
        dur = 1000 * (perf_counter() - start)
        times[run_numpy].append(dur)
        alltimes.append(dur)

        if run_numpy:
            np.linalg.slogdet(random_matrix)

    for condition, durlist in times.items():
        mean = np.mean(durlist)
        std = np.std(durlist)
        print("%s: %.1f ± %.1f" % (condition, mean, std))
