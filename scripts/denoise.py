import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt 

img = cv.imread('../data/img.png')

median = cv.medianBlur(img,3)
rgb = cv.cvtColor(median, cv.COLOR_BGR2RGB)

plt.imshow(rgb)
