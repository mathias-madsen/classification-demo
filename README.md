# Instant Classifier Training

This repository contains a demo that allows you to teach a neural network to classify an image in an online fashion.

![animation](https://github.com/mathias-madsen/classification-demo/assets/16747080/f84c37e6-a338-4f15-a14c-41961d1832a3)

You train this classifier by recording videos of the two objects you want distinguish.

The recording tool shows you how well the classifier performs on novel images as you go along. This all comes together in the following loop:

![how_it_works](https://github.com/mathias-madsen/classification-demo/assets/16747080/7a1466a0-f6db-4f4d-871d-a3e99b0a538c)


# Installation and Use

To install the dependencies, make sure you have [Python 3](https://www.python.org/downloads/) 
installed and run
```
pip install -r requirements.txt
```
ideally [in a virtual environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments). You can then test whether Python has
access to your camera by running
```
python3 test_camera.py
```
If that script opens a window with a live video feed from from your camera,
you can use
```
python3 run_classification_demo.py
```
to try the demo.
