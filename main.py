import cv2
import numpy as np

# Load the main image and the template image
card_img = cv2.imread('cards/copag/bases/hearts/2.jpg')
rank_img = cv2.imread('cards/copag/templates/black/2.jpg')
rank_img_f = cv2.rotate(rank_img,cv2.ROTATE_180)

# Convert the images to grayscale
card_img_g = cv2.cvtColor(card_img,cv2.COLOR_BGR2GRAY)
rank_img_g = cv2.cvtColor(rank_img,cv2.COLOR_BGR2GRAY)
rank_img_f_g = cv2.cvtColor(rank_img_f,cv2.COLOR_BGR2GRAY)

# Perform template matching
rightsideup = cv2.matchTemplate(card_img_g,rank_img_g,cv2.TM_CCOEFF_NORMED)
upsidedown = cv2.matchTemplate(card_img_g,rank_img_f_g,cv2.TM_CCOEFF_NORMED)

# Set a threshold for the match
threshold = 0.8
locations = np.where((rightsideup >= threshold) | (upsidedown >= threshold))

# Draw rectangles around matched locations
for location in zip(*locations[::-1]):
    cv2.rectangle(card_img,location,(location[0] + rank_img.shape[1],location[1] + rank_img.shape[0]),(0, 255, 0),2)

# Display the result
cv2.namedWindow("Result",cv2.WINDOW_NORMAL)
boxed_img = cv2.resize(card_img,(2256,1504))
cv2.imshow("Result",boxed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()