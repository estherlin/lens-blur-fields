# @author: @davelindell
# @date: 2023-04-07

import cv2
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def detect_circle_centers_with_subpixel_accuracy(image, dp, minDist, param1, param2, minRadius, maxRadius, neighborhood_size):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # return circles[0]

    if circles is not None:
        refined_circles = []
        for circle in circles[0]:
            center, radius = circle[:2], circle[2]

            # Define the local neighborhood around the detected circle center
            x_min, x_max = int(np.floor(center[0] - neighborhood_size)), int(np.ceil(center[0] + neighborhood_size))
            y_min, y_max = int(np.floor(center[1] - neighborhood_size)), int(np.ceil(center[1] + neighborhood_size))

            # Extract the intensity values within the neighborhood
            neighborhood_intensity = gray[y_min:y_max, x_min:x_max]

            # plt.imshow(neighborhood_intensity)
            # plt.show()
            # exit()

            # Generate coordinate grids for the neighborhood
            x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))

            # Compute the centroid coordinates using intensity-weighted localization
            centroid_x = np.sum(neighborhood_intensity * x_grid) / np.sum(neighborhood_intensity)
            centroid_y = np.sum(neighborhood_intensity * y_grid) / np.sum(neighborhood_intensity)

            # Add the refined circle to the list
            refined_circles.append([centroid_x, centroid_y, radius])

        # Convert the list of refined circles to a NumPy array
        refined_circles = np.array(refined_circles)

        return refined_circles

    return None


if __name__ == '__main__':
    # Load the image
    image_path = "circles.png"
    image = 255 - cv2.imread(image_path)

    # Detect circle centers with subpixel accuracy
    refined_circles = detect_circle_centers_with_subpixel_accuracy(
        image, dp=1, minDist=20, param1=100, param2=30, minRadius=20, maxRadius=80, neighborhood_size=50
    )

    # Visualize the results
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if refined_circles is not None:
        for circle in refined_circles:
            center, radius = (int(circle[0]), int(circle[1])), int(circle[2])

            # Draw the circle
            cv2.circle(image_rgb, center, radius, (0, 255, 0), 2)

            # Draw the circle center
            cv2.circle(image_rgb, center, 2, (0, 0, 255), 3)

    # Display the image with detected circles
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()
