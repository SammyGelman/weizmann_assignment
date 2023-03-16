Weizmann home assignment 

Binary classification and Denoising

Binary classification:

Create synthetic data set of circles and rectangles - train a simple binary classifier to distinguish between the two image types.

Code:
generate_circles.py:
    This script generates the data for the circles, giving them variable size and location - the files are saved in the data dir and in a non-compressed npz file format.

generate_rectangles.py:
    This script generates the data for the rectangles, giving them variable size and location - the files are saved in the data dir and in a non-compressed npz file format.

generate_datasets:
    This script is contains a function to load the npz files, combine and shuffle them in addition to attaching appropriate labels.

binary classifier:
    This script runs and trains a model to classify circles and rectangles. It plots results at the end. Plotting scripts were taken from the tesnorflow online documentation. 


Denoising:

An image was provided with a noise overlay - the object of this project is to remove the noise.

For this the python library opencv was used.

A small gaussin blur was used and then a color transform filter was used. 

denoise.py:
    This script will denoise and apply the BGR2RGB filter and plot the transformed image.

