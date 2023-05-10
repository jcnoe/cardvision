import cv2
import numpy as np

def detectSuit(card_img,manufacturer):

    threshold = 0.95

    # Convert card image to greyscale
    card_img_g = cv2.cvtColor(card_img,cv2.COLOR_BGR2GRAY)

    # Create paths
    base_path = "cards/" + manufacturer + "/templates/"
    club_path = base_path + "black/c.jpg"
    spade_path = base_path + "black/s.jpg"
    diamond_path = base_path + "red/d.jpg"
    heart_path = base_path + "red/h.jpg"

    # Look for clubs
    club_img_g = cv2.cvtColor(cv2.imread(club_path),cv2.COLOR_BGR2GRAY)
    club_img_f_g = cv2.rotate(club_img_g,cv2.ROTATE_180)
    club_0 = cv2.matchTemplate(card_img_g,club_img_g,cv2.TM_CCOEFF_NORMED)
    club_180 = cv2.matchTemplate(card_img_g,club_img_f_g,cv2.TM_CCOEFF_NORMED)
    club_locs = np.where((club_0 >= threshold) | (club_180 >= threshold))
    if (len(club_locs[0]) != 0):
        print("Card is a club")
        return "c"
    
    # Look for spades
    spade_img_g = cv2.cvtColor(cv2.imread(spade_path),cv2.COLOR_BGR2GRAY)
    spade_img_f_g = cv2.rotate(spade_img_g,cv2.ROTATE_180)
    spade_0 = cv2.matchTemplate(card_img_g,spade_img_g,cv2.TM_CCOEFF_NORMED)
    spade_180 = cv2.matchTemplate(card_img_g,spade_img_f_g,cv2.TM_CCOEFF_NORMED)
    spade_locs = np.where((spade_0 >= threshold) | (spade_180 >= threshold))
    if (len(spade_locs[0]) != 0):
        print("Card is a spade")
        return "s"
    
    # Look for diamonds
    diamond_img_g = cv2.cvtColor(cv2.imread(diamond_path),cv2.COLOR_BGR2GRAY)
    diamond_img_f_g = cv2.rotate(diamond_img_g,cv2.ROTATE_180)
    diamond_0 = cv2.matchTemplate(card_img_g,diamond_img_g,cv2.TM_CCOEFF_NORMED)
    diamond_180 = cv2.matchTemplate(card_img_g,diamond_img_f_g,cv2.TM_CCOEFF_NORMED)
    diamond_locs = np.where((diamond_0 >= threshold) | (diamond_180 >= threshold))
    if (len(diamond_locs[0]) != 0):
        print("Card is a diamond")
        return "d"
    
    # Look for hearts
    heart_img_g = cv2.cvtColor(cv2.imread(heart_path),cv2.COLOR_BGR2GRAY)
    heart_img_f_g = cv2.rotate(heart_img_g,cv2.ROTATE_180)
    heart_0 = cv2.matchTemplate(card_img_g,heart_img_g,cv2.TM_CCOEFF_NORMED)
    heart_180 = cv2.matchTemplate(card_img_g,heart_img_f_g,cv2.TM_CCOEFF_NORMED)
    heart_locs = np.where((heart_0 >= threshold) | (heart_180 >= threshold))
    if (len(heart_locs[0]) != 0):
        print("Card is a heart")
        return "h"
    
    print("Card suit is unknown")
    return "u"

# Load card, rank, and suit images
card_img = cv2.imread('cards/copag/bases/diamonds/2.jpg')
rank_img = cv2.imread('cards/copag/templates/red/10.jpg')
rank_img_f = cv2.rotate(rank_img,cv2.ROTATE_180)

# Convert to grayscale
card_img_g = cv2.cvtColor(card_img,cv2.COLOR_BGR2GRAY)
rank_img_g = cv2.cvtColor(rank_img,cv2.COLOR_BGR2GRAY)
rank_img_f_g = cv2.cvtColor(rank_img_f,cv2.COLOR_BGR2GRAY)

# Template matching in two directions
rank_0 = cv2.matchTemplate(card_img_g,rank_img_g,cv2.TM_CCOEFF_NORMED)
rank_180 = cv2.matchTemplate(card_img_g,rank_img_f_g,cv2.TM_CCOEFF_NORMED)

# Threshold the matches
threshold = 0.8
rank_locs = np.where((rank_0 >= threshold) | (rank_180 >= threshold))

# Draw rectangles around matches
for locs in zip(*rank_locs[::-1]):
    cv2.rectangle(card_img,locs,(locs[0] + rank_img.shape[1],locs[1] + rank_img.shape[0]),(0, 255, 0),2)

suit = detectSuit(card_img,"copag")

# Display boxed image
#cv2.namedWindow("Result",cv2.WINDOW_NORMAL)
#boxed_img = cv2.resize(card_img,(2256,1504))
#cv2.imshow("Result",boxed_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()