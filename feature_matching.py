# perform feature matching here
from feature_extraction import extract_features
import cv2, os
import numpy as np
# return the matching result

def match_features(image_path_1, image_path_2):
    """
    input: 2 image_paths.
    """
    key1, des1 = extract_features(image_path_1)
    key2, des2 = extract_features(image_path_2)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = []
    pts1 = []
    pts2 = []

    for m, n in matches:
        if m.distance < 0.45 * n.distance:
            good_matches.append(m)
            pts1.append(key1[m.queryIdx].pt)
            pts2.append(key2[m.trainIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    
    return good_matches, pts1, pts2


import matplotlib.pyplot as plt
if __name__ == '__main__':
    def visualize_matches(image_path_1, image_path_2):
        img1 = cv2.imread(image_path_1)
        img2 = cv2.imread(image_path_2)
        key1, des1 = extract_features(image_path_1)
        key2, des2 = extract_features(image_path_2)
        
        name1 = image_path_1.split('/')[1].split('.')[0]
        name2 = image_path_2.split('/')[1].split('.')[0]
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = []

        for m, n in matches:
            if m.distance < 0.45 * n.distance:
                good_matches.append(m)   
                
    # Draw matches
        img_matches = cv2.drawMatches(img1, key1, img2, key2, good_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Convert BGR image to RGB
        img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)            
    
        # Display the matches
        plt.figure(figsize=(12, 8))
        plt.imshow(img_matches)
        plt.title("Good Matches")
        plt.axis("off")
        plt.savefig(os.path.join('matches', name1+name2+'.png'))

    # Usage example
    
    
    images = ['images/0000.png', 
            'images/0001.png', 
            'images/0002.png',
            'images/0003.png', 
            'images/0004.png',
            'images/0005.png', 
            'images/0006.png',
            'images/0007.png',
            'images/0008.png', 
            'images/0009.png',   
            'images/0010.png']
    
    for i in range(len(images)):
        for j in range(i):
            visualize_matches(images[j], images[i])