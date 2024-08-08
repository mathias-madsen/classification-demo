import numpy as np
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, TextBox

from gaussians.moments_tracker import MomentsTracker, combine


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

    frames_per_update = 5

    def __init__(self, image_encoder, discriminator, source=0):
        
        if not plt.isinteractive():
            raise RuntimeError("Please call plt.ion() "
                               "before running this demo.")
        
        self.maxval = 100.0
        self.image_encoder = image_encoder
        self.discriminator = discriminator
        self.class_latent_episodes = {LEFT: [], RIGHT: []}
        self.class_names = {LEFT: "", RIGHT: ""}
        self.time_of_last_image_capture = -float("inf")
        self.currently_selected_class = None
        self.recording_in_progress = False

        print("Setting up camera . . .")
        if type(source) == int:
            import cv2 as cv
            self.camera = cv.VideoCapture(source)
        else:
            from ximea_camera import XimeaCamera
            self.camera = XimeaCamera()
        test_image = self.read_rgb()
        height, width, _ = test_image.shape
        print("Done: resolution %sx%s.\n" % (height, width))

        print("Testing image encoder . . .")
        test_latent = self.image_encoder(test_image)
        self.latent_dim, = dim, = test_latent.shape
        print("Latent space dimensionality: %s.\n" % (self.latent_dim,))

        self.current_tracker = MomentsTracker(np.zeros(dim), np.eye(dim), 0)
        self.class_eps_stats = {LEFT: [], RIGHT: []}

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
        self.top_box.label.set_color(LEFT_COLOR)
        self.bot_box.label.set_color(RIGHT_COLOR)

        self.top_box.text_disp.set_fontsize(28)
        self.bot_box.text_disp.set_fontsize(28)

        plt.pause(0.001)

        self.top_box.on_submit(self.store_name_neg)
        self.bot_box.on_submit(self.store_name_pos_and_continue)

        self.figure.suptitle("NAME YOUR CLASSES",
                             fontsize=24, fontweight="bold")

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
        self.figure.suptitle(string.strip(),
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

        self.set_title("Choose a class to start recording\n")
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

        self.left_button.label.set_fontsize(18)
        self.right_button.label.set_fontsize(18)

        self.figure.tight_layout()
        
        recent_latents = []
        while self.currently_selected_class is None:
            frame = self.read_rgb()
            recent_latents.append(self.image_encoder(frame))
            window.set_data(frame)
            if (len(recent_latents) >= self.frames_per_update
                and self.all_classes_nonempty()):
                evidence = self.discriminator(recent_latents).mean()
                barplot.set_value(evidence)
                recent_latents.clear()
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
        self.stop_button.label.set_fontsize(18)
        self.figure.tight_layout()

        self.current_image_list = []
        self.current_encoding_list = []
        recent_latents = []
        self.current_tracker.reset()

        while self.recording_in_progress:

            frame = self.read_rgb()
            self.current_image_list.append(frame)
            window.set_data(frame)

            latent_vector = self.image_encoder(frame)
            self.current_tracker.update_with_single(latent_vector)
            self.current_encoding_list.append(latent_vector)
            recent_latents.append(latent_vector)

            if (len(recent_latents) >= self.frames_per_update
                and self.all_classes_nonempty()):
                evidence = self.discriminator(recent_latents).mean()
                recent_latents.clear()
                barplot.set_value(evidence)

            name = self.class_names[self.currently_selected_class]
            color = LEFT_COLOR if idx == LEFT else RIGHT_COLOR
            nframes = len(self.current_encoding_list)
            title = "Recorded %s examples of class %r" % (nframes, name)
            self.set_title(title, color=color, print_too=False)

            plt.pause(0.001)

            if not plt.fignum_exists(self.figure.number):
                break  # the figure was closed manually

    def stop_recording(self, mouse_event):

        print("Stopped recording.\n")
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
        self.discard_button.label.set_fontsize(18)
        self.keep_button.label.set_fontsize(18)

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

    def all_classes_nonempty(self):
        return all(n >= 1 for n in self.num_examples_per_class().values())

    def save_recording(self, mouse_event):
        self.figure.clf()
        self.set_title("Saving recording . . .")

        # self.current_image_list = np.stack(self.current_image_list, axis=0)
        # print("Video shape: %r." % (self.current_image_list.shape,))

        label = self.currently_selected_class  # an index, 0 or 1
        name = self.class_names[label]

        episode_array = np.stack(self.current_encoding_list, axis=0)
        print("Adding %s frames to class %s." % (len(episode_array), name))

        self.class_latent_episodes[label].append(episode_array)

        self.class_eps_stats[label].append(self.current_tracker.copy())
        self.current_tracker.reset()

        self.set_title("Saved recording.\n")

        self.keep_button.set_active(False)
        self.discard_button.set_active(False)
        self.currently_selected_class = None

        del self.current_image_list

        print("CLASS SIZES:", self.num_examples_per_class())
        if self.all_classes_nonempty():
            self.show_fit_results_screen()
        else:
            self.show_ready_to_record({})

    def show_fit_results_screen(self):

        self.figure.clf()

        right_latent = self.class_latent_episodes[RIGHT]
        right_stats = self.class_eps_stats[RIGHT]
        assert len(right_latent) == len(right_stats)
        assert (sum(len(vs) for vs in right_latent) ==
                sum(m.count for m in right_stats))

        left_latent = self.class_latent_episodes[LEFT]
        left_stats = self.class_eps_stats[LEFT]
        assert len(left_latent) == len(left_stats)
        assert (sum(len(vs) for vs in left_latent) ==
                sum(m.count for m in left_stats))

        right_crossval_accuracy = None
        left_crossval_accuracy = None

        if len(right_stats) >= 2:
            right_accuracies = []
            for idx in range(len(right_stats)):
                train_left = combine(left_stats)  # all left
                train_right = combine([m for i, m in enumerate(right_stats)
                                       if i != idx])  # only selected right
                self.discriminator.fit_with_moments(train_right, train_left)
                test_right = right_latent[idx]
                logits = self.discriminator(test_right)
                right_accuracies.append(logits > 1e-5)
            right_crossval_accuracy = np.mean(np.concatenate(right_accuracies))
            print("RIGHT CROSSVAL ACCURACY:", right_crossval_accuracy)
        if len(left_stats) >= 2:
            left_accuracies = []
            for idx in range(len(left_stats)):
                train_left = combine([m for i, m in enumerate(left_stats)
                                      if i != idx])  # only selected left
                train_right = combine(right_stats)  # all right
                self.discriminator.fit_with_moments(train_right, train_left)
                test_left = left_latent[idx]
                logits = self.discriminator(test_left)
                left_accuracies.append(logits < -1e-5)
            left_crossval_accuracy = np.mean(np.concatenate(left_accuracies))
            print("LEFT CROSSVAL ACCURACY:", left_crossval_accuracy)

        train_right = combine(right_stats)
        train_left = combine(left_stats)
        self.discriminator.fit_with_moments(train_right, train_left)        

        # FYI, the training accuracy:
        test_right = np.concatenate(self.class_latent_episodes[RIGHT], axis=0)
        test_left = np.concatenate(self.class_latent_episodes[LEFT], axis=0)
        accuracy_right = np.mean(self.discriminator(test_right) > 1e-5)
        print("Training accuracy right:", accuracy_right)
        accuracy_left = np.mean(self.discriminator(test_left) < -1e-5)
        print("Training accuracy right:", accuracy_left)
        print()

        left_neps = len(self.class_eps_stats[LEFT])
        right_neps = len(self.class_eps_stats[RIGHT])
        left_nframes = sum(m.count for m in self.class_eps_stats[LEFT])
        right_nframes = sum(m.count for m in self.class_eps_stats[RIGHT])

        nrows = 6  # more rows ==> relatively smaller button
        text_axes = plt.subplot2grid((nrows, 1), (0, 0), rowspan=nrows - 1)
        button_axes = plt.subplot2grid((nrows, 1), (nrows - 1, 0))

        self.continue_button = Button(button_axes, "CONTINUE")
        self.continue_button.on_clicked(self.show_ready_to_record)
        self.continue_button.label.set_fontsize(18)

        text_axes.text(x=0, y=4, s=self.class_names[RIGHT],
                       color=RIGHT_COLOR, fontweight="bold", fontsize=36)

        if right_crossval_accuracy is not None:
            title = ("Cross-validated accuracy: %.1f pct" %
                    (100 * right_crossval_accuracy,))
            text_axes.text(x=0, y=3.5, s=title, fontsize=24,
                           fontweight="bold", color=RIGHT_COLOR)
        else:
            title = ("Cross-validated accuracy: N/A")
            text_axes.text(x=0, y=3.5, s=title, fontsize=24,
                           fontweight="bold", color="gray")

        size_info = "%s episodes (%s frames)" % (right_neps, right_nframes)
        text_axes.text(x=0, y=3.0, s=size_info, fontsize=18)


        text_axes.text(x=0, y=2.0, s=self.class_names[LEFT],
                       color=LEFT_COLOR, fontweight="bold", fontsize=24)

        if left_crossval_accuracy is not None:
            title = ("Cross-validated accuracy: %.1f pct" %
                    (100 * left_crossval_accuracy,))
            text_axes.text(x=0, y=1.5, s=title, fontsize=24,
                           fontweight="bold", color=LEFT_COLOR)
        else:
            title = ("Cross-validated accuracy: N/A")
            text_axes.text(x=0, y=1.5, s=title, fontsize=24,
                           fontweight="bold", color="gray")

        size_info = "%s episodes (%s frames)" % (left_neps, left_nframes)
        text_axes.text(x=0, y=1.0, s=size_info, fontsize=18)

        information = (
            "The cross-validation accuracy is computed by leaving out parts "
            "of the data for testing while training on the rest.\n"
            "It can be computed for a class when there are at least two "
            "episodes of data for that class.\n"
            "\n"
            "Note that it is normal for the cross-validation loss to oscillate"
            "early on; it stabilizes as more data is added."
            )
        text_axes.text(x=0, y=-1, s=information, fontsize=12, color="gray")


        text_axes.set_ylim(-2, 5)
        text_axes.axis("off")

        plt.pause(0.001)

    def discard_recording(self, mouse_event):
        self.figure.clf()
        self.set_title("Discarded recording.")
        self.keep_button.set_active(False)
        self.discard_button.set_active(False)
        self.current_episode = None
        self.currently_selected_class = None
        self.show_ready_to_record({})
