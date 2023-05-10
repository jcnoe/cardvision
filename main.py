import cv2
import numpy as np
import time

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
path = "cards/copag/bases/hearts/a.jpg"
card_img = cv2.imread(path)

suit_start = time.time()
suit = detectSuit(card_img,"copag")
suit_end = time.time()
rank_start = time.time()
rank = detectRank(card_img,"copag",suit)
rank_end = time.time()

print(rank + suit)

suit_time = str(suit_end - suit_start)
rank_time = str(rank_end - rank_start)
print("Suit execution time " + suit_time + " seconds")
print("Rank execution time is " + rank_time + " seconds")
