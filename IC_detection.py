import cv2
import numpy as np
import argparse

# Using argparse to parse the image path
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# Function for auto canny edge detection
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


# Load image into origImg
origImg = cv2.imread(args["image"])

# OPTIONAL (Depends upon orginal size of image)
# Calculating aspect ratio to resize the image
# r = 800.0 / origImg.shape[1]
# # dimensions of resized image
# dim = (800, int(origImg.shape[0] * r))
# origImg = cv2.resize(origImg, dim, interpolation=cv2.INTER_AREA)


# Applying Gaussian Blur to reduce noise
image = cv2.GaussianBlur(origImg, (21, 21), 5)

# Changing color space to HSV from RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# spliting all three channels and storing H channel in 'h'
h, s, v = cv2.split(image)

# Thresholding image
_, thresh = cv2.threshold(h, 100, 255, cv2.THRESH_BINARY)

# Morphological operations (dilation and erosion)
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=1)
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(dilated, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Applying auto canny edge detection
canny = auto_canny(eroded)

# Finding contours
_, cnts, _ = cv2.findContours(
    canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculating mean area of contours
areas = []
for c in cnts:
    areas.append(cv2.contourArea(c))
meanArea = np.mean(areas)

IC_No = 1
for (i, c) in enumerate(cnts):
    if cv2.contourArea(c) > meanArea:
        x, y, w, h = cv2.boundingRect(c)
        IC = origImg[y:y + h, x:x + w]
        cv2.imshow("IC " + str(IC_No), IC)
        # Saving the extracted ICs onto the disk
        cv2.imwrite("IC/" +
        "/IC-" + str(IC_No) + ".jpg", IC)
        IC_No += 1
        cv2.waitKey(0)


cv2.waitKey(0)
cv2.destroyAllWindows()
