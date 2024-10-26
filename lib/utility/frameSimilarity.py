import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compare_image_similarity(image1, image2):
    """
    Compare two images and return a similarity coefficient between 0 and 1.
    
    Args:
    image1 (numpy.ndarray): First input image
    image2 (numpy.ndarray): Second input image
    
    Returns:
    float: Similarity coefficient between 0 and 1
    """
    # Ensure images are the same size
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM between the two images
    similarity, _ = ssim(gray1, gray2, full=True)
    
    return similarity

# Example usage
if __name__ == "__main__":
    
    import os

    # Define the folder path
    folder_path = r"C:\Users\fpengzha\Downloads\clipped_segments\batch5"

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Sort the image files to ensure consistent ordering
    image_files.sort()

    # Initialize the list to store unique images
    unique_images = []

    # Loop through the images and compare
    for i in range(len(image_files)):
        current_img = cv2.imread(os.path.join(folder_path, image_files[i]))
        
        if i == 0:
            # Always add the first image
            unique_images.append(current_img)
        else:
            # Compare with the last unique image
            similarity_score = compare_image_similarity(unique_images[-1], current_img)
            print(f"{image_files[i-1]}, {image_files[i]}, {similarity_score}")
            
            # If images are not similar (you can adjust the threshold as needed)
            if similarity_score < 0.28:
                unique_images.append(current_img)
    
    print(f"Number of unique images: {len(unique_images)}")
    # Create output folder if it doesn't exist
    output_folder = os.path.join(folder_path, "output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save unique images to the output folder
    for i, img in enumerate(unique_images):
        output_path = os.path.join(output_folder, f"unique_image_{i+1}.jpg")
        cv2.imwrite(output_path, img)


