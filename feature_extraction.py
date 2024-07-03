
# perform feature extraction here
# return the feature vector
import cv2


def extract_features(image_path):
    """
    input: image_path (one)
    output: its sift features
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return keypoints, descriptors


import matplotlib.pyplot as plt
import os
if __name__ == '__main__':
    # Load the image
    
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
    
    for imp in images:
    
        image = cv2.imread(imp)
        if image is None:
            print("Error: Image could not be read.")
        
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create a SIFT object
        sift = cv2.SIFT_create()
        
        # Detect SIFT features
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        
        # Draw keypoints on the image
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Convert image to RGB for displaying in matplotlib
        image_with_keypoints = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)
        
        # Display the image with keypoints
        plt.figure(figsize=(10, 8))
        plt.imshow(image_with_keypoints)
        plt.title("Image with SIFT keypoints")
        plt.axis('off')
        plt.savefig(os.path.join('SIFT', imp))