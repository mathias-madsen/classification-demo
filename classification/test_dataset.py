import os
import numpy as np
from scipy.datasets import face
from tempfile import TemporaryDirectory

from classification.dataset import EncodingData
from encoding.pretrained import OnnxModel


def test_dataset():

    big_raccoon = face()
    small_raccoon = big_raccoon[:256, :320, :]

    encoder = OnnxModel()

    with TemporaryDirectory() as tempdir:

        dataset = EncodingData(encoder, tempdir)

        prep = lambda s: os.path.join(tempdir, s)

        assert os.path.isdir(prep("1"))
        assert os.path.isdir(prep("2"))
        assert os.path.isfile(prep("model_info.json"))
        assert not os.path.isfile(prep("class_indices_and_names.csv"))

        dataset.compute_dimensions(small_raccoon)
        
        dataset.class_names[1] = "LEFT"
        dataset.class_names[2] = "RIGHT"

        dataset.save_class_names_to_file()
        assert os.path.isfile(prep("class_indices_and_names.csv"))

        dataset.initialize_recording(class_index=1)
        for _ in range(3):
            dataset.record_frame(small_raccoon)
        dataset.save_recording()
        assert len(os.listdir(prep("1"))) == 3
        assert len(os.listdir(prep("2"))) == 0

        dataset.initialize_recording(class_index=1)
        for _ in range(3):
            dataset.record_frame(small_raccoon)
        dataset.save_recording()
        assert len(os.listdir(prep("1"))) == 6
        assert len(os.listdir(prep("2"))) == 0

        dataset.initialize_recording(class_index=2)
        for _ in range(3):
            dataset.record_frame(small_raccoon)
        dataset.save_recording()
        assert len(os.listdir(prep("1"))) == 6
        assert len(os.listdir(prep("2"))) == 3

        dataset.initialize_recording(class_index=2)
        for _ in range(3):
            dataset.record_frame(small_raccoon)
        dataset.discard_recording()
        assert len(os.listdir(prep("1"))) == 6
        assert len(os.listdir(prep("2"))) == 3


if __name__ == "__main__":

    test_dataset()