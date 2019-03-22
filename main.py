import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
#from tqdm import tqdm 
from random import shuffle
from xgboost import XGBClassifier
train_data = pd.read_csv("heart.csv")

#print(train_data.head())

from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split

feature_df = train_data[["age","sex","cp","trestbps","thalach","chol","restecg","exang","oldpeak","slope","ca","thal"]]
x = np.asarray(feature_df)
y = np.asarray(train_data["target"])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 5)

clf = SVC(kernel='linear', C = 0.05)

clf.fit(x_train,y_train)

y_p = clf.predict(x_test)

from sklearn.metrics import confusion_matrix
print("------------------------------------------------")

print("---------------------SVM------------------------")

print("--------------Confusion matrix------------------")
print(confusion_matrix(y_test,y_p))

fit_accuracy = clf.score(x_train, y_train)
test_accuracy = clf.score(x_test, y_test)
    
print(f"Train accuracy: {fit_accuracy:0.2%}")
print(f"Test accuracy: {test_accuracy:0.2%}")
    
print("------------------------------------------------")
print("---------------------XGB------------------------")
from sklearn.metrics import accuracy_score
model = XGBClassifier()
model.fit(x_train, y_train)
# make predictions for test data
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print("------------------------------------------------")