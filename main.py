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

def detectRank(card_img,manufacturer,suit):

    threshold = 0.95
    rank_list = ["2","3","4","5","6","7","8","9","10","j","q","k","a"]
    
    # Convert card image to greyscale
    card_img_g = cv2.cvtColor(card_img,cv2.COLOR_BGR2GRAY)

    # Generate path variable
    if ((suit == "c") | (suit == "s")):
        base_path = "cards/" + manufacturer + "/templates/black/"
    else:
        base_path = "cards/" + manufacturer + "/templates/red/"

    # For all possible ranks, perform template matching
    for rank in rank_list:
        path = base_path + rank + ".jpg"
        rank_img_g = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2GRAY)
        rank_img_g_f = cv2.rotate(rank_img_g,cv2.ROTATE_180)
        rank_0 = cv2.matchTemplate(card_img_g,rank_img_g,cv2.TM_CCOEFF_NORMED)
        rank_180 = cv2.matchTemplate(card_img_g,rank_img_g_f,cv2.TM_CCOEFF_NORMED)
        rank_locs = np.where((rank_0 >= threshold) | (rank_180 >= threshold))
        # Exit early if rank is found
        if (len(rank_locs[0]) != 0):
            print("Card is a " + rank)
            return rank

# Load card, rank, and suit images
card_img = cv2.imread('cards/copag/bases/diamonds/2.jpg')

suit = detectSuit(card_img,"copag")
rank = detectRank(card_img,"copag",suit)

# Display boxed image
#cv2.namedWindow("Result",cv2.WINDOW_NORMAL)
#boxed_img = cv2.resize(card_img,(2256,1504))
#cv2.imshow("Result",boxed_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()