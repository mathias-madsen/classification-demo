This repository contains a demo that allows you to teach a neural network
to classify an image in an online fashion. The demo will prompt you to
record videos of the two image classes you wish to distinguish, and show
you how well it performs at categorizing novel images.

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