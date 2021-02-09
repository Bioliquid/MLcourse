from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from IPython.display import Image
from imutils import paths
import numpy as np
import cv2
import os

def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

imagePaths = sorted(list(paths.list_images('train')))
data = []
labels = []

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath, 1)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    hist = extract_histogram(image)
    data.append(hist)
    labels.append(label)

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), labels, test_size=0.25, random_state=4)

model = LinearSVC(random_state=4, C=1.47)
model.fit(trainData, trainLabels)

predictions = model.predict(testData)

print(np.round(model.coef_[0][152], 2))
print(np.round(model.coef_[0][66], 2))
print(np.round(model.coef_[0][123], 2))

print(np.round(f1_score(testLabels, predictions, average='macro'), 2))

singleImage = cv2.imread('test/cat.1020.jpg')
histt = extract_histogram(singleImage)
histt2 = histt.reshape(1, -1)
print(model.predict(histt2))

singleImage = cv2.imread('test/cat.1010.jpg')
histt = extract_histogram(singleImage)
histt2 = histt.reshape(1, -1)
print(model.predict(histt2))

singleImage = cv2.imread('test/dog.1046.jpg')
histt = extract_histogram(singleImage)
histt2 = histt.reshape(1, -1)
print(model.predict(histt2))

singleImage = cv2.imread('test/dog.1013.jpg')
histt = extract_histogram(singleImage)
histt2 = histt.reshape(1, -1)
print(model.predict(histt2))