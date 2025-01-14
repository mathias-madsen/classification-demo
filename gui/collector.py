import os
import numpy as np
from time import perf_counter
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, TextBox

from classification.dataset import EncodingData
from classification.discriminator import BiGaussianDiscriminator
from gaussians.moments_tracker import combine
from gaussians.marginal_log_likelihoods import pick_best_combination



LEFT = 0
RIGHT = 1

LEFT_COLOR = "orange"
RIGHT_COLOR = "magenta"


class ProbabilityBars:

    def __init__(self, axes, names):
        self.left_plot, = axes.plot([], [], "-", lw=10, color=LEFT_COLOR)
        self.right_plot, = axes.plot([], [], "-", lw=10, color=RIGHT_COLOR)
        self.neither_plot, = axes.plot([], [], "-", lw=10, color="gray")
        axes.set_xlim(-0.1, 1.1)
        axes.set_ylim(-0.5, 2.5)
        axes.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
        left_name, right_name = names
        axes.set_yticks([2, 1, 0], [left_name, right_name, "NEITHER"])
    
    def set_value(self, logprobs):
        p_left, p_right, p_neither = np.exp(logprobs)
        self.left_plot.set_data([0, p_left], [2, 2])
        self.right_plot.set_data([0, p_right], [1, 1])
        self.neither_plot.set_data([0, p_neither], [0, 0])



class LoadBar:

    def __init__(self, figure, total):
        self.total = total
        self.num_done = 0
        self.axes = figure.add_axes(111)
        self.axes.axis("off")
        self.ticker = self.axes.text(
            x=0.5,
            y=0.5,
            s=(" " * total),
            family="Courier New",
            ha="center",
            fontweight="bold",
            fontsize=36,
            )
        plt.pause(0.001)

    def update(self):
        self.num_done += 1
        bar = "█" * self.num_done
        fill = "░" * (self.total - self.num_done)
        self.ticker.set_text(bar + fill)
        plt.pause(0.001)


class DataCollector:

    frames_per_update = 5

    def __init__(self, image_encoder, discriminator, camera, rootdir=None):
        
        if not plt.isinteractive():
            raise RuntimeError("Please call plt.ion() "
                               "before running this demo.")
        
        self.maxval = 100.0

        self.time_of_last_image_capture = -float("inf")

        self.discriminator: BiGaussianDiscriminator = discriminator

        self.dataset = EncodingData(image_encoder, rootdir)
        self.dataset.save_model_information()

        self.camera = camera
        test_image = self.camera.read_mirrored_rgb()
        self.dataset.compute_dimensions(test_image)

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

        self.figure.suptitle("NAME YOUR CLASSES",
                             fontsize=24, fontweight="bold")

        self.bot_axes.set_xlabel(" ", fontsize=18, color="gray")

        plt.pause(0.001)

        self.top_box.on_text_change(self.inspect_user_provided_names)
        self.bot_box.on_text_change(self.inspect_user_provided_names)
        self.top_box.on_submit(self.inspect_user_provided_names)
        self.bot_box.on_submit(self.inspect_user_provided_names)
        self.figure.canvas.mpl_connect(
            "key_press_event",
            self.store_names_if_enter_was_clicked_and_ready,
            )

    def process_current_text_box_contents(self):
        top_name = self.top_box.text.replace("\n", " ").strip().upper()
        bot_name = self.bot_box.text.replace("\n", " ").strip().upper()
        return top_name, bot_name
    
    def inspect_user_provided_names(self, event):
        plt.pause(0.01)
        name1, name2 = self.process_current_text_box_contents()
        if name1 and name2 and name1 != name2:
            msg = "Press Enter to continue"
            self.bot_axes.set_xlabel(msg, fontsize=18, color="gray")
        else:
            self.bot_axes.set_xlabel("", fontsize=18, color="gray")

    def store_names_if_enter_was_clicked_and_ready(self, event):
        name1, name2 = self.process_current_text_box_contents()
        if event.key == "enter":
            if name1 and name2 and name1 != name2:
                self.dataset.class_names[LEFT] = name1
                self.dataset.class_names[RIGHT] = name2
                self.top_box.set_active(False)
                self.bot_box.set_active(False)
                self.dataset.save_class_names_to_file()
                self.show_ready_to_record({})
        else:
            self.inspect_user_provided_names(None)

    def set_title(self, string, color="black", print_too=True):
        if print_too:
            print(string)
        self.figure.suptitle(string.strip(),
                             fontsize=24,
                             fontweight="bold",
                             color=color)
        plt.pause(0.001)

    def reset_stopwatch(self):
        self.time_of_last_pause = perf_counter()
    
    def pause_till_complete(self, fps=10.0):
        seconds_already_taken = perf_counter() - self.time_of_last_pause
        target_duration_in_seconds = 1 / fps
        remaining_seconds = target_duration_in_seconds - seconds_already_taken
        plt.pause(max(remaining_seconds, 0.001))

    def show_ready_to_record(self, mouse_event):

        if hasattr(self, "continue_button"):
            self.continue_button.set_active(False)

        self.figure.clf()

        self.set_title("Choose a class to start recording\n")
        nrows = 6  # more rows ==> larger image and smaller buttons

        image_axes = plt.subplot2grid((nrows, 2), (0, 0),
                                      rowspan=nrows - 2, colspan=2)
        image_axes.axis("off")
        rgb = self.camera.read_mirrored_rgb()
        window = image_axes.imshow(rgb)

        bar_axes = plt.subplot2grid((nrows, 2), (nrows - 2, 0), colspan=2)
        barplot = ProbabilityBars(bar_axes, self.dataset.class_names.values())

        self.left_axes = plt.subplot2grid((nrows, 2), (nrows - 1, 0))
        self.right_axes = plt.subplot2grid((nrows, 2), (nrows - 1, 1))

        self.left_button = Button(
            self.left_axes,
            "CLASS %r" % self.dataset.class_names[LEFT],
            color=LEFT_COLOR,
            )

        self.right_button = Button(
            self.right_axes,
            "CLASS %r" % self.dataset.class_names[RIGHT],
            color=RIGHT_COLOR,
            )

        self.left_button.on_clicked(self.start_recording)
        self.right_button.on_clicked(self.start_recording)

        self.left_button.label.set_fontsize(18)
        self.right_button.label.set_fontsize(18)

        self.figure.tight_layout()

        nshown = 0
        nonempty = self.dataset.all_classes_nonempty()
        self.reset_stopwatch()
        while self.dataset.currently_selected_class is None:
            frame = self.camera.read_mirrored_rgb()
            window.set_data(frame)
            if nonempty and nshown % 3 == 0:
                latent = self.dataset.image_encoder(frame)
                logprobs = self.discriminator(latent)
                barplot.set_value(logprobs)
            del frame
            nshown += 1
            if not plt.fignum_exists(self.figure.number):
                break  # the figure was closed manually
            self.pause_till_complete()

    def start_recording(self, mouse_event):
        if mouse_event.inaxes == self.left_axes:
            self.dataset.initialize_recording(LEFT)
        elif mouse_event.inaxes == self.right_axes:
            self.dataset.initialize_recording(RIGHT)
        else:
            raise Exception("Unexpected axes: %r" % mouse_event.inaxes)        
        self.left_button.set_active(False)
        self.right_button.set_active(False)
        self.figure.clf()
        self.show_recording_in_progress()

    def show_recording_in_progress(self):

        self.figure.clf()
        name = self.dataset.get_current_class_name()
        if self.dataset.currently_selected_class == LEFT:
            color = LEFT_COLOR
        elif self.dataset.currently_selected_class == RIGHT:
            color = RIGHT_COLOR
        else:
            raise Exception("Unexpected class index %r" %
                            self.dataset.currently_selected_class)

        self.recording_in_progress = True
        nrows = 7  # more rows ==> larger image and smaller buttons
        image_axes = plt.subplot2grid((nrows, 2), (0, 0), rowspan=nrows - 1, colspan=2)
        image_axes.axis("off")
        rgb = self.camera.read_mirrored_rgb()
        window = image_axes.imshow(rgb)

        title = self.figure.suptitle(
            "Recoding for class %r" % name,
            color=color,
            fontsize=28,
            fontweight="bold",
            )

        bar_axes = plt.subplot2grid((nrows, 2), (nrows - 1, 0))
        barplot = ProbabilityBars(bar_axes, self.dataset.class_names.values())
        
        button_axes = plt.subplot2grid((nrows, 2), (nrows - 1, 1))
        self.stop_button = Button(button_axes, "STOP")
        self.stop_button.on_clicked(self.stop_recording)
        self.stop_button.label.set_fontsize(18)
        self.figure.tight_layout()

        nframes = 0
        nonempty = self.dataset.all_classes_nonempty()
        self.reset_stopwatch()
        while self.recording_in_progress:

            frame = self.camera.read_mirrored_rgb()
            self.dataset.record_frame(frame)
            window.set_data(frame)
            nframes += 1

            if (nframes % self.frames_per_update == 0 and nonempty):
                all_latents = self.dataset.current_encoding_list
                logprobs = self.discriminator(all_latents[-1])
                barplot.set_value(logprobs)
            
            name = self.dataset.get_current_class_name()            
            header = "Recorded %s examples of class %r" % (nframes, name)
            title.set_text(header)
            
            self.pause_till_complete()

            if not plt.fignum_exists(self.figure.number):
                break  # the figure was closed manually

    def stop_recording(self, mouse_event):

        print("Stopped recording.\n")
        self.recording_in_progress = False
        self.stop_button.set_active(False)
        self.show_rate_recording_screen()

    def show_rate_recording_screen(self):

        self.figure.clf()
        idx = self.dataset.currently_selected_class
        name = self.dataset.get_current_class_name()
        self.set_title("Save video for class %r?" % name,
                       color=LEFT_COLOR if idx == LEFT else RIGHT_COLOR)

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
                time = np.random.randint(len(self.dataset.current_image_list))
                axes.imshow(self.dataset.current_image_list[time])
        plt.pause(0.001)

    def save_recording(self, mouse_event):

        self.figure.clf()

        self.set_title("Saving recording . . .")
        self.dataset.save_recording()
        self.set_title("Saved recording.\n")

        self.keep_button.set_active(False)
        self.discard_button.set_active(False)

        if self.dataset.all_classes_nonempty():
            self.show_fit_results_screen()
        else:
            self.show_ready_to_record({})

    def show_fit_results_screen(self):

        self.figure.clf()
        plt.pause(0.001)

        num_steps = self.dataset.compute_num_cross_val_steps()
        load_bar = LoadBar(self.figure, num_steps)

        k0, n0 = 0, 0  # totals for the 'left' class
        for k, n in self.dataset.crossval_accuracy(0):
            k0 += k
            n0 += n
            load_bar.update()

        if n0 > 0:
            print("LEFT CROSSVAL ACCURACY: %s/%s = %.1f pct" %
                  (k0, n0, 100.0 * k0 / n0))

        k1, n1 = 0, 0  # totals for the 'right' class
        for k, n in self.dataset.crossval_accuracy(1):
            k1 += k
            n1 += n
            load_bar.update()

        if n1 > 0:
            print("RIGHT CROSSVAL ACCURACY: %s/%s = %.1f pct" %
                  (k1, n1, 100.0 * k1 / n1))

        if n0 > 0 or n1 > 0:
            print("")  # blank line below the printed accuracies

        neg = combine(self.dataset.class_episode_stats[LEFT])
        pos = combine(self.dataset.class_episode_stats[RIGHT])
        neg, pos = pick_best_combination(neg, pos, verbose=True)
        self.discriminator.set_stats(neg, pos)
        outpath = os.path.join(self.dataset.rootdir, "discriminator.npz")
        self.discriminator.save(outpath)

        # FYI, the training accuracy:
        for idx in [LEFT, RIGHT]:
            name = self.dataset.class_names[idx]
            eps_codes = self.dataset.class_episode_codes[idx]
            concat_eps_codes = np.concatenate(eps_codes, axis=0)
            corrects = self.discriminator.classify(concat_eps_codes) == idx
            k = sum(corrects)
            n = len(corrects)
            print("Training accuracy %r: %s/%s = %.1f pct" %
                (name, k, n, 100.0 * k / n))
        print()

        # clear the crossval bar and make space for the fit results screen:
        self.figure.clf()
        plt.pause(0.001)

        left_neps = len(self.dataset.class_episode_stats[LEFT])
        right_neps = len(self.dataset.class_episode_stats[RIGHT])
        left_nframes = sum(m.count for m in self.dataset.class_episode_stats[LEFT])
        right_nframes = sum(m.count for m in self.dataset.class_episode_stats[RIGHT])

        nrows = 6  # more rows ==> relatively smaller button
        text_axes = plt.subplot2grid((nrows, 1), (0, 0), rowspan=nrows - 1)
        button_axes = plt.subplot2grid((nrows, 1), (nrows - 1, 0))

        self.continue_button = Button(button_axes, "CONTINUE")
        self.continue_button.on_clicked(self.show_ready_to_record)
        self.continue_button.label.set_fontsize(18)

        # LEFT == 0

        text_axes.text(x=0, y=4.0, s=self.dataset.class_names[LEFT],
                       color=LEFT_COLOR, fontweight="bold", fontsize=36)

        if n0 > 0:
            title = ("Cross-validated accuracy: %s/%s = %.1f pct" %
                    (k0, n0, 100 * k0 / n0))
            text_axes.text(x=0, y=3.5, s=title, fontsize=24,
                           fontweight="bold", color=LEFT_COLOR)
        else:
            title = ("Cross-validated accuracy: N/A")
            text_axes.text(x=0, y=3.5, s=title, fontsize=24,
                           fontweight="bold", color="gray")

        size_info = "%s episodes (%s frames)" % (left_neps, left_nframes)
        text_axes.text(x=0, y=3.0, s=size_info, fontsize=18)

        # RIGHT == 1

        text_axes.text(x=0, y=2.0, s=self.dataset.class_names[RIGHT],
                       color=RIGHT_COLOR, fontweight="bold", fontsize=36)

        if n1 > 0:
            title = ("Cross-validated accuracy: %s/%s = %.1f pct" %
                    (k1, n1, 100 * k1 / n1))
            text_axes.text(x=0, y=1.5, s=title, fontsize=24,
                           fontweight="bold", color=RIGHT_COLOR)
        else:
            title = ("Cross-validated accuracy: N/A")
            text_axes.text(x=0, y=1.5, s=title, fontsize=24,
                           fontweight="bold", color="gray")

        size_info = "%s episodes (%s frames)" % (right_neps, right_nframes)
        text_axes.text(x=0, y=1.0, s=size_info, fontsize=18)

        information = (
            "The cross-validation accuracy is computed by leaving out parts "
            "of the data for testing while training on the rest.\n"
            "It can be computed for a class when there are at least two "
            "episodes of data for that class.\n"
            "\n"
            "Note that the cross-validation loss may vary when there are "
            "few episodes; it stabilizes as more data is added."
            )
        text_axes.text(x=0, y=-1, s=information, fontsize=12, color="gray")

        text_axes.set_ylim(-2, 5)
        text_axes.axis("off")

        outpath = os.path.join(self.dataset.rootdir, "params.npz")
        self.discriminator.save(outpath)

        self.figure.tight_layout()
        plt.pause(0.001)

    def discard_recording(self, mouse_event):
        self.figure.clf()
        self.dataset.discard_recording()
        self.set_title("Discarded recording.")
        self.keep_button.set_active(False)
        self.discard_button.set_active(False)
        self.show_ready_to_record({})
