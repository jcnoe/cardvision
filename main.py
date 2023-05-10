import cv2
import numpy as np

def detectSuit(card_img,manufacturer):

    threshold = 0.95
    suit_list = ["c","s","d","h"]

    # Convert card image to greyscale
    card_img_g = cv2.cvtColor(card_img,cv2.COLOR_BGR2GRAY)

    # For all possible ranks, perform template matching
    for suit in suit_list:
        # Generate path variable
        if ((suit == "c") | (suit == "s")):
            base_path = "cards/" + manufacturer + "/templates/black/"
        else:
            base_path = "cards/" + manufacturer + "/templates/red/"
        path = base_path + suit + ".jpg"
        suit_img_g = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2GRAY)
        suit_img_g_f = cv2.rotate(suit_img_g,cv2.ROTATE_180)
        suit_0 = cv2.matchTemplate(card_img_g,suit_img_g,cv2.TM_CCOEFF_NORMED)
        suit_180 = cv2.matchTemplate(card_img_g,suit_img_g_f,cv2.TM_CCOEFF_NORMED)
        suit_locs = np.where((suit_0 >= threshold) | (suit_180 >= threshold))
        # Exit early if rank is found
        if (len(suit_locs[0]) != 0):
            return suit
    
    print("Card suit cannot be determined")
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
            return rank
        
    print("Card rank cannot be determined")
    return "u"

# Load card, rank, and suit images
card_img = cv2.imread('cards/copag/bases/hearts/2.jpg')

suit = detectSuit(card_img,"copag")
rank = detectRank(card_img,"copag",suit)

print(rank + suit)

# Display boxed image
#cv2.namedWindow("Result",cv2.WINDOW_NORMAL)
#boxed_img = cv2.resize(card_img,(2256,1504))
#cv2.imshow("Result",boxed_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()