from matplotlib import pyplot as plt

from discriminator import BiGaussianDiscriminator
from image_encoding import encode
from collector import DataCollector


if __name__ == "__main__":

    plt.ion()
    collector = DataCollector(image_encoder=encode,
                              discriminator=BiGaussianDiscriminator())

