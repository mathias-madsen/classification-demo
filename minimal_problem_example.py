import numpy as np
from onnxruntime import InferenceSession
from time import perf_counter
from scipy.misc import face
from tqdm import tqdm
from matplotlib import pyplot as plt


dim = 512
random_matrix_square_root = np.random.normal(size=(dim, dim))
random_matrix = random_matrix_square_root.T @ random_matrix_square_root

session = InferenceSession(
    "encoding/optimized_model.onnx",
    # "encoding/model.onnx",
    providers = ["CPUExecutionProvider"],
    )

output_names = [node.name for node in session.get_outputs()]

raccoon = face()
assert raccoon.dtype == np.uint8
raccoon = raccoon[:256, :320, :]
image = raccoon.astype(np.float32) / 255.
image_batch = image[None,]

if __name__ == "__main__":

    times = {True: [], False: []}
    alltimes = []

    for iteration in tqdm(range(400)):

        run_numpy = iteration % 40 < 20

        start = perf_counter()
        feed = {"vision1": image_batch, "vision2": image_batch}
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

    plt.figure(figsize=(12, 5))
    plt.plot(alltimes, ".-")
    plt.xlabel("frame number")
    plt.ylabel("duration (ms)")
    plt.show()
