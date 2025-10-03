from sklearn.preprocessing import StandardScaler
from PIL import Image
from skimage.feature import hog
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt

def extract_combined_features(image_path, image_size=(64, 64)):
    try:
        image_rgb = Image.open(image_path).resize(image_size)
        image_gray = image_rgb.convert('L')
        gray_array = np.array(image_gray)

        hog_features = hog(
            gray_array,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )

        rgb_array = np.array(image_rgb)
        hist_features = []
        for i in range(3):
            hist, _ = np.histogram(
                rgb_array[:, :, i],
                bins=32,
                range=(0, 256),
                density=True
            )
            hist_features.extend(hist)

        combined = np.concatenate((hog_features, hist_features))
        return combined

    except Exception as e:
        print(f"Skipping {image_path}: {e}")
        return None

def main():
    img_path = input("Enter Image Path please: ") # request for the image path

    display_img = Image.open(img_path)

    features = extract_combined_features(img_path) # extract the features from the img

    if features is None:
        print("Can't recognize the image")
        return None

    image = np.array([features]) # convert it ti numpy array, the way that module deal with photos

    scaler = joblib.load("DT_scaler.pkl") # import the same scaler
    model = joblib.load("DT_model.pkl") # import the module

    image_scaled = scaler.transform(image) # transform the np array

    prediction = model.predict(image_scaled) # do the prediction

    # Load the labels
    with open("../label_map.pkl", "rb") as f:
        label_map = pickle.load(f)

    predicted_class = label_map[prediction[0]]

    # ===== Show the image with label =====
    plt.figure(figsize=(16, 8))
    plt.imshow(display_img)
    plt.title(f"Label: {predicted_class}", fontsize=30)  # Increase fontsize here
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    # ===== Show the image with label =====

    print(f"The prediction is: {predicted_class}")

if __name__ == '__main__':
    main()
