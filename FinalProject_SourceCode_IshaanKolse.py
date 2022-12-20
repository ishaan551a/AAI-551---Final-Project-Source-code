# Opencv is the key library for this project, particularly the cv2 module
import cv2 as cv
import numpy as np

# Lets load in our test image
image = cv.imread('1Cones1.jpg')

# Display the test image, simply press enter to dismiss the display window
cv.imshow('Test Image', image)
cv.waitKey(0)
cv.destroyAllWindows()

# converting to HSV format which is best suited for machine vision tasks
hsv1 = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# HSV channel values for isolating the cone
lower_bound = (10, 50, 50)
upper_bound = (20, 255, 255)

# Create the initial mask based on the desired HSV range for the orange color to isolate the cone
mask1 = cv.inRange(hsv1, lower_bound, upper_bound)
isolated_image = cv.bitwise_and(image, image, mask=mask1)
cv.imshow('Isolated Image', isolated_image)
cv.waitKey(0)
cv.destroyAllWindows()


# Once again converting to HSV format, but this time we use the isolated image
hsv2 = cv.cvtColor(isolated_image, cv.COLOR_BGR2HSV)
# Actual thresholding
lower_bound2 = (15, 0, 0)
upper_bound2 = (40, 255, 255)

# Perform color thresholding
mask2 = cv.inRange(hsv2, lower_bound2, upper_bound2)

# Display the resulting image
cv.imshow('Masked Image', mask2)
cv.waitKey(0)
cv.destroyAllWindows()

moments = cv.moments(mask2)

# Compute the centroid of the masked image
centroid_x = int(moments['m10'] / moments['m00'])
centroid_y = int(moments['m01'] / moments['m00'])

# Draw a circle at the centroid location
centroidm = cv.circle(mask2, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
centroidi = cv.circle(image, (centroid_x, centroid_y), 5, (0, 0, 255), -1)


# Display the resulting image
cv.imshow('Centroid for masked image',centroidm )
cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow('Centroid for initial image',centroidi )
cv.waitKey(0)
cv.destroyAllWindows()





