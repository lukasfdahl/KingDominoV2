import cv2
import numpy as np


def main():
    # Load the images
    board_image = cv2.imread('King Domino dataset/Train/2.jpg', cv2.IMREAD_GRAYSCALE)
    crown_image = cv2.imread('King Domino dataset/CrownTemplate.png', cv2.IMREAD_GRAYSCALE)
    
    # Step 1: Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Step 2: Detect keypoints and descriptors
    kp_board, des_board = sift.detectAndCompute(board_image, None)
    kp_crown, des_crown = sift.detectAndCompute(crown_image, None)
    
    # Step 3: Use a brute-force matcher to find the best matches between board and crown descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_crown, des_board)
    
    # Step 4: Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Step 5: Draw matches
    matched_image = cv2.drawMatches(crown_image, kp_crown, board_image, kp_board, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Step 6: Show the matched image
    cv2.imshow("Matched SIFT", matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Step 7: Count the number of matches (this represents crown detection)
    print(f"Number of crowns detected: {len(matches)}")







if __name__ == "__main__":
    main()