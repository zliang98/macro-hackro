import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure, morphology

# Load and binarize the image
images = io.imread('generate_rotating_ellipsoids result - 5degreesperframe.tif')  # Or .jpg if already converted

labeled_image = measure.label(images)
regions = measure.regionprops(labeled_image)