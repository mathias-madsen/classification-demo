import numpy as np
import cv2 as cv
import time
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from collections import defaultdict


class DataCollector:

    # .add_axes takes (left, bottom, width, height)

    def __init__(self,
                 image_encoder,
                 discriminator,
                 display_fps=24,
                 save_fps=24,
                 trim_seconds_start=0.0,
                 trim_seconds_end=0.0,
                 ):
        if not plt.isinteractive():
            raise RuntimeError("Please call plt.ion() "
                               "before running this demo.")
        self.image_encoder = image_encoder
        self.discriminator = discriminator
        self.class_latent_episodes = defaultdict(list)
        self.display_fps = display_fps
        self.save_fps = save_fps
        self.trim_seconds_start = trim_seconds_start
        self.trim_seconds_end = trim_seconds_end
        self.time_of_last_image_capture = -float("inf")
        self.currently_selected_class = None
        self.recording_in_progress = False
        print("Setting up camera . . .")
        self.camera = cv.VideoCapture(0)
        self.test_image = self.read_rgb()
        height, width, _ = self.test_image.shape
        print("Done: resolution %sx%s.\n" % (height, width))
        self.figure = plt.figure(figsize=(12, 8))
        self.initialize_before_recording_screen({})

    def set_title(self, string, color="black"):
        print(string)
        self.figure.suptitle(string,
                             fontsize=24,
                             fontweight="bold",
                             color=color)
        plt.pause(0.001)

    def read_rgb(self):
        elapsed = time.perf_counter() - self.time_of_last_image_capture
        if elapsed < 1 / self.display_fps:
            time.sleep(1 / self.display_fps - elapsed)
        success, bgr = self.camera.read()
        assert success
        rgb = bgr[:, :, ::-1]
        rgb = rgb[:, ::-1, :]  # mirror left/right for visual sanity
        return rgb[::4, ::4]  # downsample for speed

    def initialize_before_recording_screen(self, mouse_event):

        if hasattr(self, "continue_button"):
            self.continue_button.set_active(False)

        self.figure.clf()

        self.set_title("Choose a class to start recording")
        num_classes = 2
        nrows = 6  # more rows ==> larger image and smaller buttons
        image_axes = plt.subplot2grid((nrows, num_classes),
                                      (0, 0),
                                      rowspan=nrows - 1,
                                      colspan=num_classes)
        image_axes.axis("off")
        rgb = self.read_rgb()
        window = image_axes.imshow(rgb)
        self.class_1_axes = plt.subplot2grid((nrows, 2), (nrows - 1, 0))
        self.class_2_axes = plt.subplot2grid((nrows, 2), (nrows - 1, 1))
        self.button_1 = Button(self.class_1_axes, "CLASS A", color="orange")
        self.button_2 = Button(self.class_2_axes, "CLASS B", color="magenta")
        self.button_1.on_clicked(self.start_recording)
        self.button_2.on_clicked(self.start_recording)
        self.figure.tight_layout()
        while self.currently_selected_class is None:
            frame = self.read_rgb()
            window.set_data(frame)
            plt.pause(0.001)
            if not plt.fignum_exists(self.figure.number):
                print("The figure was closed.")
                break  # the figure was closed manually

    def start_recording(self, mouse_event):
        if mouse_event.inaxes == self.class_1_axes:
            self.currently_selected_class = "A"
        elif mouse_event.inaxes == self.class_2_axes:
            self.currently_selected_class = "B"
        else:
            raise Exception("Unexpected axes: %r" % mouse_event.inaxes)
        self.button_1.set_active(False)
        self.button_2.set_active(False)
        self.figure.clf()
        self.initialize_during_recording_screen()

    def initialize_during_recording_screen(self):
        
        self.figure.clf()
        name = self.currently_selected_class
        self.set_title("Recording for class %r." % name,
                       color="orange" if name == "A" else "magenta")

        self.recording_in_progress = True
        nrows = 7  # more rows ==> larger image and smaller buttons
        image_axes = plt.subplot2grid((nrows, 2), (0, 0), rowspan=nrows - 1, colspan=2)
        image_axes.axis("off")
        rgb = self.read_rgb()
        window = image_axes.imshow(rgb)
        bar_axes = plt.subplot2grid((nrows, 2), (nrows - 1, 0))
        barplot, = bar_axes.plot([], [], lw=30)
        bar_axes.set_xlim(-30.0, 30.0)
        bar_axes.set_ylim(-1.0, +1.0)
        bar_axes.set_yticks([])
        button_axes = plt.subplot2grid((nrows, 2), (nrows - 1, 1))
        self.stop_button = Button(button_axes, "STOP")
        self.stop_button.on_clicked(self.stop_recording)
        self.figure.tight_layout()

        self.current_image_list = []
        self.current_encoding_list = []
        self.current_results_list = []
        while self.recording_in_progress:
            frame = self.read_rgb()
            window.set_data(frame)
            self.current_image_list.append(frame)
            assert self.current_encoding_list is not None
            
            latent_vector = self.image_encoder(frame)
            self.current_encoding_list.append(latent_vector)
            evidence = self.discriminator(latent_vector)
            assert np.shape(evidence) == (), np.shape(evidence)
            
            # we categorize decisions as confidently correct,
            # confidently incorrect, or not confident:
            if self.currently_selected_class == "A":
                self.current_results_list.append(evidence > 1e-5)
            elif self.currently_selected_class == "B":
                self.current_results_list.append(evidence < -1e-5)
            
            if evidence > 0.0:
                barplot.set_color("orange")
            else:
                barplot.set_color("magenta")
            barplot.set_data([0.0, -evidence], [0, 0])
            nsteps = len(self.current_encoding_list)
            name = self.currently_selected_class
            num_correct = sum(self.current_results_list)
            num_total = len(self.current_results_list)
            self.set_title("%s/%s correct labels as class %r" %
                           (num_correct, num_total, name),
                           color="orange" if name == "A" else "magenta")
            plt.pause(1 / self.display_fps)
            if not plt.fignum_exists(self.figure.number):
                print("The figure was closed.")
                break  # the figure was closed manually

    def stop_recording(self, mouse_event):
        
        print("Stopped recording.")
        self.recording_in_progress = False
        self.stop_button.set_active(False)
        self.initialize_after_recording_screen()

    def initialize_after_recording_screen(self):

        self.figure.clf()
        name = self.currently_selected_class
        self.set_title("Keep recording for class %r?" % name,
                       color="orange" if name == "A" else "magenta")

        nrows = 5
        ncols = 4

        # create the axes before so that there is something to show:
        axdict = {(r, c): plt.subplot2grid((nrows, ncols), (r, c))
                  for r in range(nrows - 1) for c in range(ncols)}
        
        for axes in axdict.values():
            axes.axis("off")
        
        keep_axes = plt.subplot2grid((nrows, ncols),
                                        (nrows - 1, ncols // 2),
                                        colspan=ncols // 2)

        discard_axes = plt.subplot2grid((nrows, ncols),
                                     (nrows - 1, 0),
                                     colspan=ncols // 2)
        
        self.discard_button = Button(discard_axes, "NO, DISCARD", color="lightsalmon")
        self.keep_button = Button(keep_axes, "YES, SAVE", color="palegreen")

        self.keep_button.on_clicked(self.save_recording)
        self.discard_button.on_clicked(self.discard_recording)

        self.figure.tight_layout()

        # and NOW fill the axes with example images:
        plt.pause(0.001)
        for row in range(nrows - 1):
            for col in range(ncols):
                axes = axdict[row, col]
                time = np.random.randint(len(self.current_image_list))
                axes.imshow(self.current_image_list[time])
        plt.pause(0.001)
    
    def num_examples_per_class(self):
        return {label: sum(len(e) for e in self.class_latent_episodes[label])
                for label in ["A", "B"]}

    def save_recording(self, mouse_event):
        self.figure.clf()
        self.set_title("Saving recording . . .")

        self.current_image_list = np.stack(self.current_image_list, axis=0)
        print("Video shape: %r." % (self.current_image_list.shape,))

        self.current_encoding_list = np.stack(self.current_encoding_list, axis=0)
        print("Latent vectors shape: %r." % (self.current_encoding_list.shape,))

        # optionally trim the ends of the episode:
        full_length = len(self.current_encoding_list)
        start = int(round(self.trim_seconds_start / self.display_fps))
        end = full_length - int(round(self.trim_seconds_end / self.display_fps))
        print(start, ":", end)
        truncated_episode = self.current_encoding_list[start:end]
        print("Trimmed shape: %r." % (truncated_episode.shape,))

        # optionally throw away from intermediate (redundant) frames:
        downsampling_factor = self.display_fps // self.save_fps
        downsampled_episode = truncated_episode[::downsampling_factor]
        print("Downsampled shape: %r." % (downsampled_episode.shape,))

        label = self.currently_selected_class
        self.class_latent_episodes[label].append(downsampled_episode)

        self.set_title("Saved recording.")
        self.keep_button.set_active(False)
        self.discard_button.set_active(False)
        self.currently_selected_class = None

        sizes = self.num_examples_per_class()
        print("CLASS SIZES:", sizes)
        minlength = downsampled_episode.shape[1] + 1
        assert minlength >= 1
        if all(n >= 1 for n in sizes.values()):
        # if True:
            pos = np.concatenate(self.class_latent_episodes["A"], axis=0)
            neg = np.concatenate(self.class_latent_episodes["B"], axis=0)
            self.discriminator.fit(pos, neg)
            self.show_fit_results_screen()
        else:
            self.initialize_before_recording_screen({})

    def show_fit_results_screen(self):
        self.figure.clf()
        pos_vectors = np.concatenate(self.class_latent_episodes["A"], axis=0)
        neg_vectors = np.concatenate(self.class_latent_episodes["B"], axis=0)
        pos_scores = self.discriminator(pos_vectors)
        neg_scores = self.discriminator(neg_vectors)

        nrows = 6  # more rows ==> larger image and smaller buttons
        hist_axes = plt.subplot2grid((nrows, 1), (0, 0), rowspan=nrows - 1)
        button_axes = plt.subplot2grid((nrows, 1), (nrows - 1, 0))

        self.continue_button = Button(button_axes, "CONTINUE")
        self.continue_button.on_clicked(self.initialize_before_recording_screen)

        counts = self.num_examples_per_class()
        hist_axes.hist(-pos_scores, bins=30, color="orange", alpha=0.5,
                       density=True, label="CLASS A (N=%s)" % counts["A"])
        hist_axes.hist(-neg_scores, bins=30, color="magenta", alpha=0.5,
                       density=True, label="CLASS B (N=%s)" % counts["B"])
        hist_axes.legend()
        hist_axes.set_title("Evidence for class B according to the model AFTER FIT")
        self.figure.tight_layout()
        plt.pause(0.001)

    def discard_recording(self, mouse_event):
        self.figure.clf()
        self.set_title("Discarded recording.")
        self.keep_button.set_active(False)
        self.discard_button.set_active(False)
        self.current_episode = None
        self.currently_selected_class = None
        self.initialize_before_recording_screen({})
