import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.widgets import Button


LEFT = 0
RIGHT = 1

LEFT_COLOR = "orange"
RIGHT_COLOR = "magenta"


class DataCollector:

    def __init__(self, image_encoder, discriminator):
        if not plt.isinteractive():
            raise RuntimeError("Please call plt.ion() "
                               "before running this demo.")
        self.image_encoder = image_encoder
        self.discriminator = discriminator
        self.class_latent_episodes = {LEFT: [], RIGHT: []}
        self.time_of_last_image_capture = -float("inf")
        self.currently_selected_class = None
        self.recording_in_progress = False
        print("Setting up camera . . .")
        self.camera = cv.VideoCapture(0)
        self.test_image = self.read_rgb()
        height, width, _ = self.test_image.shape
        print("Done: resolution %sx%s.\n" % (height, width))
        self.figure = plt.figure(figsize=(12, 8))
        self.figure.canvas.mpl_connect("close_event", self.on_close)
        self.show_ready_to_record({})
    
    def on_close(self, event):
        print("The figure was closed.\n\n")

    def set_title(self, string, color="black"):
        print(string)
        self.figure.suptitle(string,
                             fontsize=24,
                             fontweight="bold",
                             color=color)
        plt.pause(0.001)

    def read_rgb(self):
        success, bgr = self.camera.read()
        assert success
        rgb = bgr[:, :, ::-1]
        rgb = rgb[:, ::-1, :]  # mirror left/right for visual sanity
        return rgb[::4, ::4]  # downsample for speed

    def show_ready_to_record(self, mouse_event):

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

        self.left_axes = plt.subplot2grid((nrows, 2), (nrows - 1, 0))
        self.right_axes = plt.subplot2grid((nrows, 2), (nrows - 1, 1))

        self.left_button = Button(self.left_axes,
                               "CLASS %s" % LEFT,
                               color=LEFT_COLOR)

        self.right_button = Button(self.right_axes,
                               "CLASS %s" % RIGHT,
                               color=RIGHT_COLOR)

        self.left_button.on_clicked(self.start_recording)
        self.right_button.on_clicked(self.start_recording)
        self.figure.tight_layout()
        while self.currently_selected_class is None:
            frame = self.read_rgb()
            window.set_data(frame)
            plt.pause(0.001)
            if not plt.fignum_exists(self.figure.number):
                break  # the figure was closed manually

    def start_recording(self, mouse_event):
        if mouse_event.inaxes == self.left_axes:
            self.currently_selected_class = LEFT
        elif mouse_event.inaxes == self.right_axes:
            self.currently_selected_class = RIGHT
        else:
            raise Exception("Unexpected axes: %r" % mouse_event.inaxes)
        self.left_button.set_active(False)
        self.right_button.set_active(False)
        self.figure.clf()
        self.show_recording_in_progress()

    def show_recording_in_progress(self):

        self.figure.clf()
        name = self.currently_selected_class
        self.set_title("Recording for class %r." % name,
                       color=LEFT_COLOR if name == LEFT else RIGHT_COLOR)

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
            if self.currently_selected_class == RIGHT:
                self.current_results_list.append(evidence > 1e-5)
            elif self.currently_selected_class == LEFT:
                self.current_results_list.append(evidence < -1e-5)

            if evidence > 0.0:
                barplot.set_color(RIGHT_COLOR)
            else:
                barplot.set_color(LEFT_COLOR)
            barplot.set_data([0.0, evidence], [0, 0])
            name = self.currently_selected_class
            num_correct = sum(self.current_results_list)
            num_total = len(self.current_results_list)
            self.set_title("%s/%s correct labels as class %r" %
                           (num_correct, num_total, name),
                           color=LEFT_COLOR if name == LEFT else RIGHT_COLOR)
            plt.pause(0.001)
            if not plt.fignum_exists(self.figure.number):
                break  # the figure was closed manually

    def stop_recording(self, mouse_event):

        print("Stopped recording.")
        self.recording_in_progress = False
        self.stop_button.set_active(False)
        self.show_rate_recording_screen()

    def show_rate_recording_screen(self):

        self.figure.clf()
        name = self.currently_selected_class
        self.set_title("Keep recording for class %r?" % name,
                       color=LEFT_COLOR if name == LEFT else RIGHT_COLOR)

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

        self.discard_button = Button(discard_axes,
                                     "NO, DISCARD",
                                     color="lightsalmon")
        
        self.keep_button = Button(keep_axes,
                                  "YES, SAVE",
                                  color="palegreen")

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
        return {label: sum(len(e) for e in collection) for
                label, collection in self.class_latent_episodes.items()}

    def save_recording(self, mouse_event):
        self.figure.clf()
        self.set_title("Saving recording . . .")

        self.current_image_list = np.stack(self.current_image_list, axis=0)
        print("Video shape: %r." % (self.current_image_list.shape,))

        self.current_encoding_list = np.stack(self.current_encoding_list, axis=0)
        print("Latent vectors shape: %r." % (self.current_encoding_list.shape,))

        label = self.currently_selected_class
        self.class_latent_episodes[label].append(self.current_encoding_list)

        self.set_title("Saved recording.")
        self.keep_button.set_active(False)
        self.discard_button.set_active(False)
        self.currently_selected_class = None

        sizes = self.num_examples_per_class()
        print("CLASS SIZES:", sizes)
        minlength = self.current_encoding_list.shape[1] + 1
        assert minlength >= 1
        if all(n >= 1 for n in sizes.values()):
            self.show_fit_results_screen()
        else:
            self.show_ready_to_record({})

    def show_fit_results_screen(self):

        self.figure.clf()
        pos_vectors = np.concatenate(self.class_latent_episodes[RIGHT], axis=0)
        neg_vectors = np.concatenate(self.class_latent_episodes[LEFT], axis=0)

        pos_scores_before = self.discriminator(pos_vectors)
        neg_scores_before = self.discriminator(neg_vectors)

        self.discriminator.fit(pos_vectors, neg_vectors)

        pos_scores_after = self.discriminator(pos_vectors)
        neg_scores_after = self.discriminator(neg_vectors)

        nrows = 6  # more rows ==> larger image and smaller buttons
        rowspan = (nrows - 1) // 2
        hist_axes_top = plt.subplot2grid((nrows, 1), (0, 0), rowspan=rowspan)
        hist_axes_bot = plt.subplot2grid((nrows, 1), (rowspan, 0), rowspan=rowspan)
        button_axes = plt.subplot2grid((nrows, 1), (2*rowspan, 0))

        self.continue_button = Button(button_axes, "CONTINUE")
        self.continue_button.on_clicked(self.show_ready_to_record)

        counts = self.num_examples_per_class()

        hist_axes_top.hist(pos_scores_before,
                           bins=25,
                           color=RIGHT_COLOR,
                           alpha=0.5,
                           density=True,
                           label="CLASS %s (N=%s)" % (RIGHT, counts[RIGHT]))
        
        hist_axes_top.hist(neg_scores_before,
                           bins=25,
                           color=LEFT_COLOR,
                           alpha=0.5,
                           density=True,
                           label="CLASS %s (N=%s)" % (LEFT, counts[LEFT]))
        
        hist_axes_top.legend()
        hist_axes_top.set_title("Evidence in favor of %r BEFORE FIT" % (RIGHT,))

        hist_axes_bot.hist(pos_scores_after,
                           bins=25,
                           color=RIGHT_COLOR,
                           alpha=0.5,
                           density=True,
                           label="CLASS %s (N=%s)" % (RIGHT, counts[RIGHT]))
        
        hist_axes_bot.hist(neg_scores_after,
                           bins=25,
                           color=LEFT_COLOR,
                           alpha=0.5,
                           density=True,
                           label="CLASS %s (N=%s)" % (LEFT, counts[LEFT]))
        
        hist_axes_bot.legend()
        hist_axes_bot.set_title("Evidence in favor of %r AFTER FIT" % (RIGHT,))

        self.figure.tight_layout()
        plt.pause(0.001)

    def discard_recording(self, mouse_event):
        self.figure.clf()
        self.set_title("Discarded recording.")
        self.keep_button.set_active(False)
        self.discard_button.set_active(False)
        self.current_episode = None
        self.currently_selected_class = None
        self.show_ready_to_record({})
