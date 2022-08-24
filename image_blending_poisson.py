# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
# from google.colab.patches import cv2_imshow

def PoissonImageBlending(source, destination):
  # create an all "White" mask: 255, if black mask is 0
  mask = 255 * np.ones(destination.shape, destination.dtype) 
  # navigate the source img location
  width, height, channels = source.shape
  center = (height//2, width//2)

  # using built-in funtion `cv2.seamlessClone` to achieve Poisson Image goal
  blended = cv2.seamlessClone(destination, source, mask, center, 2) # cv::MIXED_CLONE = 2
  output = blended
  imageio.imsave("output.png", output)
  img = cv2.imread("output.png")
  cv2.imshow('blend',img)

# run it!!
source = imageio.imread("bXlI1jWQdwZZMwWsS8ILyiZB0GxBpYAz.jpg")
destination = imageio.imread("x6YLfGvlzfa5iYCavY5ZxZ40CLUxSzf7.jpg")
PoissonImageBlending(source, destination)