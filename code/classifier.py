import glob
import numpy as np
from datetime import datetime
from sklearn.externals import joblib

from udacityCode import featureExtractByName

if __name__ == "__main__":
    carImages = glob.glob('../classification_images/vehicles/*/*/*.png')
    nonCarImages = glob.glob('../classification_images/non-vehicles/*/*/*.png')

    print("Number of car images: ", len(carImages))
    print("Number of non-car images: ", len(nonCarImages))

    startTime = datetime.now()

    # Only picked a subset of the images while developing this code and testing different classifier parameters.
    carImages = carImages[:]  #
    nonCarImages = nonCarImages[:]

    # Calculated the features for one image to get the shape of the feature data.
    testFeature = featureExtractByName(carImages[0])
    print(testFeature.shape)

    carLabels = np.ones(2*len(carImages))
    nonCarLabels = np.zeros(2*len(nonCarImages))
    features_car = np.zeros((len(carLabels), len(testFeature)))
    features_nonCar = np.zeros((len(nonCarLabels), len(testFeature)))
    print("Starting to extract features")

    # Augemented the datasets by flipping left to right and right to left.
    for i, name in enumerate(carImages):
        features_car[2*i, :] = featureExtractByName(name, flip=False, rescale=256)
        features_car[2*i + 1, :] = featureExtractByName(name, flip=True, rescale=256)

    for i, name in enumerate(nonCarImages):
        features_nonCar[2 * i, :] = featureExtractByName(name, flip=False, rescale=256)
        features_nonCar[2 * i + 1, :] = featureExtractByName(name, flip=True, rescale=256)


    print("Done extracting features")
    middleTime = datetime.now()
    print(middleTime - startTime)
    print("Starting to fit features")

    """
    'Trained' and used a StandardScaler to have mean zero and varaiance one for all features in the data sets.
    """
    allFeatures = np.concatenate((features_car, features_nonCar))
    allFeatures = allFeatures.astype(np.float64)
    from sklearn.preprocessing import StandardScaler
    featureScaler = StandardScaler().fit(allFeatures)

    features_car = featureScaler.transform(features_car.astype(np.float64))
    features_nonCar = featureScaler.transform(features_nonCar.astype(np.float64))


    # Split the car and noncar data into training and tests sets.
    from sklearn.model_selection import train_test_split
    features_car_train, features_car_test, labels_car_train, labels_car_test = train_test_split(features_car, carLabels, test_size=0.10)
    features_nonCar_train, features_nonCar_test, labels_nonCar_train, labels_nonCar_test = train_test_split(features_nonCar, nonCarLabels, test_size=0.10)


    # Merged and shuffled the training set for both cars and noncars
    features_train = np.concatenate((features_car_train, features_nonCar_train))
    labels_train = np.concatenate((labels_car_train, labels_nonCar_train))
    from sklearn.utils import shuffle
    features_train, labels_train = shuffle(features_train, labels_train)

    """
    Training the classifiers.
    Some code is commented as it was mainly used when I experimented with different parameters for the classifier.
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn import svm
    # SVC = svm.SVC()
    # clfSVC = GridSearchCV(SVC, {'kernel':('linear', 'rbf'), 'C':[1, 10, 50, 100]}, verbose=4, n_jobs=7)
    # C = 10, kernel = rbf worked best for this.
    clfSVC = svm.SVC(kernel='rbf', C=10)
    clfSVC.fit(features_train, labels_train)
    # print(clfSVC.best_params_)
    print("SVM Score:")
    print("Car: ", clfSVC.score(features_car_test, labels_car_test))
    print("nonCar: ", clfSVC.score(features_nonCar_test, labels_nonCar_test))

    print(datetime.now() - middleTime)
    joblib.dump(clfSVC, '../model/SVC.pkl')
    joblib.dump(featureScaler, '../model/scaler.pkl')
else:
    clfSVC = joblib.load('../model/SVC.pkl')
    featureScaler = joblib.load('../model/scaler.pkl')
