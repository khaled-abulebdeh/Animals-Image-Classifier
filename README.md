# Animals Image Classifier 🐾

An image classification project using three machine learning approaches: **Decision Tree**, **Naive Bayes**, and **Feedforward Neural Networks**. The classifiers were trained on a custom dataset of animal images using `scikit-learn`. This project was developed as part of the Artificial Intelligence course (ENCS3340) at the Faculty of Engineering & Technology.

## Table Of Contents

* [Project Description](#project-description)
* [Feedforward Neural Networks Approach](#feedforward-neural-networks-approach)
* [Decision Tree Approach](#decision-tree-approach)
* [Naive Bayes Approach](#naive-bayes-approach)
* [Requirements](#requirements)
* [Contributors](#contributors)

***

## Project Description

This project investigates the performance of classical machine learning models in image classification. The dataset contains **3000 images** across three classes: `cats`, `pandas`, and `spiders`. Each image was resized to **64x64**, and features were extracted using:

* **Histogram of Oriented Gradients (HOG)** for texture and shape
* **RGB color histograms** for color distribution

The extracted features were then used to train and evaluate the three models.

---

## Feedforward Neural Networks Approach

The **FNN** was implemented using `MLPClassifier` from Scikit-learn. The architecture included two hidden layers with **2048** and **1024** neurons, using the **ReLU** activation function and **Adam** optimizer. The model was trained using the **holdout method** (80% training / 20% testing). It achieved an overall accuracy of **79%** on the test set.

Key points:

* Uses adaptive learning rate and early stopping.
* Well-suited for capturing non-linear relationships.
* Slightly lower performance than Decision Tree but more flexible for complex data.

---

## Decision Tree Approach

The **Decision Tree** was implemented using a **Random Forest Classifier** with 600 estimators. It was trained using the holdout method and also used feature scaling. The model achieved an accuracy of **81%**, outperforming other models in overall precision.

Key points:

* Best overall accuracy among the three models.
* Prone to overfitting without ensemble methods.
* Easy to interpret and analyze using tree plotting.

---

## Naive Bayes Approach

The **Naive Bayes** classifier was used as a baseline model. It assumes feature independence and was evaluated using **5-fold cross-validation**. Despite its simplicity, it performed surprisingly well due to good feature engineering, though less accurate than the FNN and Random Forest.

Key points:

* Fast and lightweight.
* Assumes independence between features.
* Performed well due to robust feature extraction.

---

## Requirements

* `Python 3.8+`
* `scikit-learn`
* `numpy`
* `matplotlib`
* `opencv-python`
* `joblib`
* `seaborn` (optional for visualization)

You can install all requirements using:

```bash
pip install -r requirements.txt
```

## Contributors
* Khaled Abu Lebdeh
* Abdalraheem Shuaibi

