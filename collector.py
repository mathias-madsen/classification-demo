import numpy as np
import cv2 as cv
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, TextBox


LEFT = 0
RIGHT = 1

LEFT_COLOR = "orange"
RIGHT_COLOR = "magenta"


class EvidenceBar(Rectangle):

    def __init__(self, axes, maxval=350.0):
        Rectangle.__init__(self, xy=(0.0, 0.1), width=0, height=0.8)
        axes.add_patch(self)
        axes.set_ylim(0, 1)
        axes.set_yticks([])
        axes.set_xlim(-maxval, +maxval)
    
    def set_value(self, value):
        self.set_width(value)
        self.set_color(LEFT_COLOR if value < 0 else RIGHT_COLOR)


class DataCollector:

    def __init__(self, image_encoder, discriminator):
        self.maxval = 100.0
        if not plt.isinteractive():
            raise RuntimeError("Please call plt.ion() "
                               "before running this demo.")
        self.image_encoder = image_encoder
        self.discriminator = discriminator
        self.class_latent_episodes = {LEFT: [], RIGHT: []}
        self.class_names = {LEFT: "", RIGHT: ""}
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
        self.show_provide_names({})

    def on_close(self, event):
        print("The figure was closed.\n\n")

    def show_provide_names(self, event):

        self.figure.clf()

        self.top_axes = plt.subplot2grid((2, 1), (0, 0))
        self.bot_axes = plt.subplot2grid((2, 1), (1, 0))

        self.top_box = TextBox(self.top_axes, label="A:")
        self.bot_box = TextBox(self.bot_axes, label="B:")

        self.top_box.label.set_fontsize(28)
        self.bot_box.label.set_fontsize(28)
        self.top_box.label.set_fontweight("bold")
        self.bot_box.label.set_fontweight("bold")
        self.top_box.label.set_color("green")
        self.bot_box.label.set_color("blue")

        self.top_box.text_disp.set_fontsize(28)
        self.bot_box.text_disp.set_fontsize(28)

        plt.pause(0.001)

        self.top_box.on_submit(self.store_name_neg)
        self.bot_box.on_submit(self.store_name_pos_and_continue)

        self.figure.suptitle("NAME YOUR CLASSES")

    def store_name_neg(self, text):
        self.class_names[LEFT] = text.upper()

    def store_name_pos_and_continue(self, text):
        self.class_names[RIGHT] = text.upper()
        if all(self.class_names.values()):
            self.top_box.set_active(False)
            self.bot_box.set_active(False)
            self.show_ready_to_record({})

    def set_title(self, string, color="black", print_too=True):
        if print_too:
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
        nrows = 6  # more rows ==> larger image and smaller buttons

        image_axes = plt.subplot2grid((nrows, 2), (0, 0),
                                      rowspan=nrows - 2, colspan=2)
        image_axes.axis("off")
        rgb = self.read_rgb()
        window = image_axes.imshow(rgb)

        bar_axes = plt.subplot2grid((nrows, 2), (nrows - 2, 0), colspan=2)
        barplot = EvidenceBar(bar_axes, maxval=self.maxval)

        self.left_axes = plt.subplot2grid((nrows, 2), (nrows - 1, 0))
        self.right_axes = plt.subplot2grid((nrows, 2), (nrows - 1, 1))

        self.left_button = Button(self.left_axes,
                               "CLASS %r" % self.class_names[LEFT],
                               color=LEFT_COLOR)

        self.right_button = Button(self.right_axes,
                               "CLASS %r" % self.class_names[RIGHT],
                               color=RIGHT_COLOR)

        self.left_button.on_clicked(self.start_recording)
        self.right_button.on_clicked(self.start_recording)
        self.figure.tight_layout()
        while self.currently_selected_class is None:
            frame = self.read_rgb()
            evidence = self.discriminator(self.image_encoder(frame))
            barplot.set_value(evidence)
            window.set_data(frame)
            del frame
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
        idx = self.currently_selected_class
        name = self.class_names[idx]
        self.set_title("Recording for class %r." % name,
                       color=LEFT_COLOR if idx == LEFT else RIGHT_COLOR)

        self.recording_in_progress = True
        nrows = 7  # more rows ==> larger image and smaller buttons
        image_axes = plt.subplot2grid((nrows, 2), (0, 0), rowspan=nrows - 1, colspan=2)
        image_axes.axis("off")
        rgb = self.read_rgb()
        window = image_axes.imshow(rgb)
        
        bar_axes = plt.subplot2grid((nrows, 2), (nrows - 1, 0))
        barplot = EvidenceBar(bar_axes, maxval=self.maxval)
        
        button_axes = plt.subplot2grid((nrows, 2), (nrows - 1, 1))
        self.stop_button = Button(button_axes, "STOP")
        self.stop_button.on_clicked(self.stop_recording)
        self.figure.tight_layout()

        self.current_image_list = []
        self.current_encoding_list = []
        self.current_results_list = []

        while self.recording_in_progress:

            frame = self.read_rgb()
            self.current_image_list.append(frame)
            window.set_data(frame)

            latent_vector = self.image_encoder(frame)
            self.current_encoding_list.append(latent_vector)
            evidence = self.discriminator(latent_vector)
            barplot.set_value(evidence)

            # we categorize decisions as confidently correct,
            # confidently incorrect, or not confident:
            if self.currently_selected_class == RIGHT:
                self.current_results_list.append(evidence > 1e-5)
            elif self.currently_selected_class == LEFT:
                self.current_results_list.append(evidence < -1e-5)
            num_correct = sum(self.current_results_list)
            num_total = len(self.current_results_list)

            name = self.class_names[self.currently_selected_class]
            color = LEFT_COLOR if idx == LEFT else RIGHT_COLOR
            self.set_title("%s/%s correct labels as class %r" %
                           (num_correct, num_total, name),
                           color=color, print_too=False)

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
        idx = self.currently_selected_class
        name = self.class_names[idx]
        self.set_title("Save video for class %r?" % name,
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
        return {self.class_names[idx]: sum(len(e) for e in collection)
                for idx, collection in self.class_latent_episodes.items()}

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

        del self.current_image_list

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
        lname = self.class_names[LEFT]
        rname = self.class_names[RIGHT]

        if not (np.allclose(pos_scores_before, 0) and
                np.allclose(neg_scores_before, 0)):

            hist_axes_top.hist(pos_scores_before,
                            bins=25,
                            color=RIGHT_COLOR,
                            alpha=0.5,
                            density=True,
                            label="CLASS %r (N=%s)" % (rname, counts[rname]))
            
            hist_axes_top.hist(neg_scores_before,
                            bins=25,
                            color=LEFT_COLOR,
                            alpha=0.5,
                            density=True,
                            label="CLASS %r (N=%s)" % (lname, counts[lname]))
            
            hist_axes_top.legend()
            hist_axes_top.set_title("BEFORE model fitting")

        hist_axes_bot.hist(pos_scores_after,
                           bins=25,
                           color=RIGHT_COLOR,
                           alpha=0.5,
                           density=True,
                           label="CLASS %r (N=%s)" % (rname, counts[rname]))
        
        hist_axes_bot.hist(neg_scores_after,
                           bins=25,
                           color=LEFT_COLOR,
                           alpha=0.5,
                           density=True,
                           label="CLASS %r (N=%s)" % (lname, counts[lname]))
        
        hist_axes_bot.legend()
        hist_axes_bot.set_title("AFTER model fitting")

        all_scores = np.concatenate([pos_scores_after, neg_scores_after])
        self.maxval = 1.5 * np.max(np.abs(all_scores))

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
