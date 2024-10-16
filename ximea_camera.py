import numpy as np
from ximea import xiapi


class XimeaCamera(xiapi.Camera):

    def __init__(self, downsampling_factor=None):
        xiapi.Camera.__init__(self)
        self.open_device()
        self.set_exposure(10000)
        self.set_imgdataformat("XI_RGB24")
        self.start_acquisition()
        self.image = xiapi.Image()
        self.downsampling_factor = downsampling_factor

    def read(self):
        self.get_image(self.image)
        data_raw = self.image.get_image_data_raw()
        data_array = np.frombuffer(data_raw, dtype=np.uint8)
        reshaped = data_array.reshape([self.image.height, self.image.width, 3])
        return True, reshaped
    
    def read_mirrored_rgb(self):
        success, bgr = self.read()
        assert success
        rgb = bgr[:, :, ::-1]
        rgb = rgb[:, ::-1, :]  # mirror left/right for visual sanity
        dsf = self.downsampling_factor
        return rgb[::dsf, ::dsf]  # downsample for speed
