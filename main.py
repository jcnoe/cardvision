import cv2
import numpy as np

# Load card, rank, and suit images
card_img = cv2.imread('cards/copag/bases/hearts/2.jpg')
rank_img = cv2.imread('cards/copag/templates/black/2.jpg')
rank_img_f = cv2.rotate(rank_img,cv2.ROTATE_180)
suit_img = cv2.imread('cards/copag/templates/red/h.jpg')
suit_img_f = cv2.rotate(suit_img,cv2.ROTATE_180)

# Convert to grayscale
card_img_g = cv2.cvtColor(card_img,cv2.COLOR_BGR2GRAY)
rank_img_g = cv2.cvtColor(rank_img,cv2.COLOR_BGR2GRAY)
rank_img_f_g = cv2.cvtColor(rank_img_f,cv2.COLOR_BGR2GRAY)
suit_img_g = cv2.cvtColor(suit_img,cv2.COLOR_BGR2GRAY)
suit_img_f_g = cv2.cvtColor(suit_img_f,cv2.COLOR_BGR2GRAY)

# Template matching in two directions
rank_0 = cv2.matchTemplate(card_img_g,rank_img_g,cv2.TM_CCOEFF_NORMED)
rank_180 = cv2.matchTemplate(card_img_g,rank_img_f_g,cv2.TM_CCOEFF_NORMED)
suit_0 = cv2.matchTemplate(card_img_g,suit_img_g,cv2.TM_CCOEFF_NORMED)
suit_180 = cv2.matchTemplate(card_img_g,suit_img_f_g,cv2.TM_CCOEFF_NORMED)

# Threshold the matches
threshold = 0.8
rank_locs = np.where((rank_0 >= threshold) | (rank_180 >= threshold))
suit_locs = np.where((suit_0 >= threshold) | (suit_180 >= threshold))

# Draw rectangles around matches
for locs in zip(*rank_locs[::-1]):
    cv2.rectangle(card_img,locs,(locs[0] + rank_img.shape[1],locs[1] + rank_img.shape[0]),(0, 255, 0),2)

for locs in zip(*suit_locs[::-1]):
    cv2.rectangle(card_img,locs,(locs[0] + suit_img.shape[1],locs[1] + suit_img.shape[0]),(0, 255, 0),2)

# Display boxed image
cv2.namedWindow("Result",cv2.WINDOW_NORMAL)
boxed_img = cv2.resize(card_img,(2256,1504))
cv2.imshow("Result",boxed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()