from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
import os
from PIL import Image
import numpy as np
import pickle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler


def load_images_with_combined_features(folder_path, image_size=(64, 64), limit_per_class=1000):
    X, y = [], []
    label_map = {}
    current_label = 0

    for class_name in sorted(os.listdir(folder_path)):
        class_path = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_path):
            continue

        label_map[current_label] = class_name
        count = 0

        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                if count >= limit_per_class:
                    break
                image_path = os.path.join(class_path, filename)
                features = extract_combined_features(image_path, image_size=image_size)
                if features is not None:
                    X.append(features)
                    y.append(current_label)
                    count += 1

        current_label += 1

    return np.array(X), np.array(y), label_map

def extract_combined_features(image_path, image_size=(64, 64)):
    try:
        # Open the image from the given file path and resize it
        image_rgb = Image.open(image_path).resize(image_size)

        # Convert the image to grayscale for HOG feature extraction
        image_gray = image_rgb.convert('L')

        # Convert the grayscale image to a NumPy array
        gray_array = np.array(image_gray)

        # HOG (Histogram of Oriented Gradients) Features 
        # Extract texture and edge information from the grayscale image
        hog_features = hog(
            gray_array,
            orientations=9,             # Number of orientation bins
            pixels_per_cell=(8, 8),     # Size of the cell in pixels
            cells_per_block=(2, 2),     # Number of cells per block
            block_norm='L2-Hys'         # Normalization method
        )

        # Convert the RGB image to a NumPy array
        rgb_array = np.array(image_rgb)

        # Color Histogram Features 
        # Compute normalized histograms for R, G, B channels separately
        hist_features = []
        for i in range(3):  # Loop over Red, Green, Blue channels
            hist, _ = np.histogram(
                rgb_array[:, :, i],     # Select the color channel
                bins=32,                # Number of bins for the histogram
                range=(0, 256),         # Pixel intensity range
                density=True            # Normalize the histogram
            )
            hist_features.extend(hist)  # Add histogram data to the list

        # Combine HOG and Histogram Features 
        combined = np.concatenate((hog_features, hist_features))

        return combined

    except Exception as e:
        # Print and skip any image that causes an error
        print(f"Skipping {image_path}: {e}")
        return None

# The following commented code has to run at least one time on your device

# Load the Dataset (just once) and save label map (using pickle) 
X, y, label_map = load_images_with_combined_features(r"C:\Users\khale\OneDrive\Desktop\COURSES\AI\project_2\dataset\animals")

np.save("X.npy", X)#save
np.save("y.npy", y)#save
with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)#write


# Read the loaded date 
X = np.load("X.npy")
y = np.load("y.npy")
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)


# X is the matrix of features for all items
# X[i] is the feature vector (i.e. 1D array containing all features for one item) 
# y[i] is The label (e.g., 0 = cat, 1 = dog, 2 = spider) for X[i]

# For averaging
accuracies = []
precisions, recalls, f1s = [], [], []
conf_matrices = []

# Prepare cross-validation
nfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)

# Split the data into 5 folds, each fold consists from (training_set, testing_set) 
fold = 1
for training_index, testing_index in nfold.split(X, y):# the loop calls next() automatically to give the proper splitted sets for the fold
    # train_index: A NumPy array holds all indices of the training items in this fold, e.x. [0 4 7 ... 2887 2996 2999] 

    # Get the training and testing data for this fold
    X_train, X_test = X[training_index], X[testing_index]
    y_train, y_test = y[training_index], y[testing_index]

    # Normalize feature values
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    """

    # Train Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)

    # Save performance measurements
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    # Precision, Recall, F1 (macro average)
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    precisions.append(p)
    recalls.append(r)
    f1s.append(f)
    # Confusion matrix (optional)
    cm = confusion_matrix(y_test, y_pred)
    conf_matrices.append(cm)

    fold += 1

print("\n=== Average Cross-Validation Results ===")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
print(f"Average Precision: {np.mean(precisions):.4f}")
print(f"Average Recall:    {np.mean(recalls):.4f}")
print(f"Average F1-score:  {np.mean(f1s):.4f}")

# Convert list of confusion matrices to a NumPy array and compute mean
avg_cm = np.mean(conf_matrices, axis=0).astype(int)
# Print in clean table format
print("\nAverage Confusion Matrix:")
labels = list(label_map.values())
header = "Predicted →\t" + "\t".join(labels)
print(header)
for i, row in enumerate(avg_cm):
    row_str = "\t".join(f"{val:.2f}" for val in row)
    print(f"Actual {labels[i]}:\t{row_str}")

# After evaluating measurements using n-fold cross-validation, 
# the complete model should be constructed from the entire dataset
final_model = GaussianNB()
final_model.fit(X, y)
# save the final model:
with open("final_model.pkl", "wb") as f:
    pickle.dump(final_model, f)