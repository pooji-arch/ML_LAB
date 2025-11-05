#DECISSION TREE
from matplotlib import pyplot as plt
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
df = pandas.read_csv("data.csv")
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
tree.plot_tree(dtree,feature_names=features)
plt.show()
"""INPUT (data.csv):

Age,Experience,Rank,Nationality,Go
36,10,9,UK,NO
42,12,4,USA,NO
23,4,6,N,NO
52,4,4,USA,NO
43,21,8,USA,YES
44,14,5,UK,NO
66,3,7,N,YES
35,14,9,UK,YES
52,13,7,N,YES
35,5,9,N,YES
24,3,5,USA,NO
18,3,7,UK,YES
45,9,9,UK,YES"""

#SPAM DETECTION
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
spam = pd.read_csv('data.csv')
z = spam['EmailText']
y = spam["Label"]
z_train, z_test,y_train, y_test = train_test_split(z,y,test_size = 0.2)
cv = CountVectorizer()
features = cv.fit_transform(z_train)
model = svm.SVC()
model.fit(features,y_train)
features_test = cv.transform(z_test)
print(model.score(features_test,y_test))
features_test = cv.transform(z_test)
print("Accuracy: {}".format(model.score(features_test,y_test)))
import pickle
from tkinter import *
def check_spam():
    text = spam_text_Entry.get()
    with open('data.csv') as file:
        contents = file.read()
        if text in contents:
            print(text,"text is spam")
            my_string_var.set("Result: text is spam")
        else:
             print(text,"text not a spam")
             my_string_var.set("Result: text not a spam")
win = Tk()
win.geometry("400x600")
win.configure(background="cyan")
win.title("Email Spam Detector")
title = Label(win, text="Email Spam Detector", bg="gray",width="300",height="2",fg="white",font=("Calibri 20 bold italic underline")).pack()
spam_text = Label(win, text="Enter your Text: ",bg="cyan", font=("Verdana 12")).place(x=12,y=100)
spam_text_Entry = Entry(win, textvariable=spam_text,width=33)
spam_text_Entry.place(x=155, y=105)
my_string_var = StringVar()
my_string_var.set("Result: ")
print_spam = Label(win,textvariable=my_string_var,bg="cyan",font=("Verdana 12")).place(x=12,y=200)
Button = Button(win, text="Submit",width="12",height="1",activebackground="red",bg="Pink",command=check_spam, font=("Verdana 12")).place(x=12,y=150)
win.mainloop()
"""
Input (data.csv):
EmailText,Label
sale,spam
gasssss,ham
huge,spam
tint,spam
ginger,spam"""

#FACE DETECTION ABDHULKALAM
import cv2
face_cascade = cv2.CascadeClassifier('C:\\Users\\Admin\\AppData\\
Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\
data\\haarcascade_frontalface_default.xml')
img = cv2.imread('C:\\Users\\Admin\\Desktop\\New notes\\LAB\\kalam.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#REGRESSION
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
class LocallyWeightedRegression:
    #maths behind Linear Regression:
    # theta = inv(X.T*W*X)*(X.T*W*Y)this will be our theta whic will
    # be learnt for each point
    # initializer of LocallyWeighted Regression that stores tau as parameters
    def __init__(self, tau = 0.01):
        self.tau = tau
    def kernel(self, query_point, X):
        Weight_matrix = np.mat(np.eye(len(X)))
        for idx in range(len(X)):
            Weight_matrix[idx,idx] = np.exp(np.dot(X[idx]-query_point, (X[idx]-query_point).T)/         
            (-2*self.tau*self.tau))
        return Weight_matrix
    # function that makes the predictions of the output of a given query point
    def predict(self, X, Y, query_point):
        q = np.mat([query_point, 1])
        X = np.hstack((X, np.ones((len(X), 1))))
        W = self.kernel(q, X)
        theta = np.linalg.pinv(X.T*(W*X))*(X.T*(W*Y))
        pred = np.dot(q, theta)
        return pred
    #function that fits and predicts the output of all query points
    def fit_and_predict(self, X, Y):
        Y_test, X_test = [], np.linspace(-np.max(X), np.max(X), len(X))
        for x in X_test:
            pred = self.predict(X, Y, x)
            Y_test.append(pred[0][0])
        Y_test = np.array(Y_test)
        return Y_test
    # function that computes the score rmse
    def score(self, Y, Y_pred):
        return np.sqrt(np.mean((Y-Y_pred)**2))
    # function that fits as well as shows the scatter plot of all points
    def fit_and_show(self, X, Y):
        Y_test, X_test = [], np.linspace(-np.max(X), np.max(X), len(X))
        for x in X_test:
            pred = self.predict(X, Y, x)
            Y_test.append(pred[0][0])
        Y_test = np.array(Y_test)
        plt.style.use('seaborn')
        plt.title("The scatter plot for the value of tau = %.5f"% self.tau)
        plt.scatter(X, Y, color = 'red')
        plt.scatter(X_test, Y_test, color = 'green')
        plt.show()
# reading the csv files of the given dataset
dfx = pd.read_csv('weightedX.csv')
dfy = pd.read_csv('weightedY.csv')
# store the values of dataframes in numpy arrays
X = dfx.values
Y = dfy.values
# normalising the data values
u = X.mean()
std = X.std()
X = ((X-u)/std)
tau = 0.2
model = LocallyWeightedRegression(tau)
Y_pred = model.fit_and_predict(X, Y)
model.fit_and_show(X, Y)
"""
INPUT:(weightedX.csv)
1.2421
2.3348
0.13264
2.347
6.7389
3.7089
11.853
-1.8708
4.5025
3.2798
1.7573
3.3784
11.47
9.0595
-2.8174
9.3184
8.4211
0.86215
7.5544
-3.9883
INPUT:(weightedY.csv)
1.1718
1.8824
0.34283
2.1057
1.6477
2.3624
2.1212
-0.79712
2.0311
1.9795
1.471
2.4611
1.9819
1.1203
-1.3701
1.0287
1.3808
1.2178
1.4084
-1.5209"""

#K means
import sys
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
x=[4,5,10,4,3,11,14,6,10,12]
y=[21,19,24,17,16,25,24,22,21,21]
data = list(zip(x, y))
inertias = []
for i in range(1,11):
    kmean=KMeans(n_clusters=i)
    kmean.fit(data)
    inertias.append(kmean.inertia_)
plt.plot(range(1,11),inertias,marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

#CHARACTER RECOGNITION
import numpy as np
import pandas as pd
# Load data
data=pd.read_csv('HR_comma_sep.csv')
data.head()
# Import LabelEncoder
from sklearn import preprocessing

# Creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
data['salary']=le.fit_transform(data['salary'])
data['Departments']=le.fit_transform(data['Departments'])
# Spliting data into Feature and
X=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','Departments','salary']]
y=data['left']

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)  # 70% training and 30% test
# Import MLPClassifer
from sklearn.neural_network import MLPClassifier

# Create model object
clf = MLPClassifier(hidden_layer_sizes=(6,5),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.01)

# Fit data onto the model
clf.fit(X_train,y_train)
# Make prediction on test dataset
ypred=clf.predict(X_test)

# Import accuracy score
from sklearn.metrics import accuracy_score

# Calcuate accuracy
accuracy_score(y_test,ypred)
"""
INPUT :(HR_comma_sep.csv)
satisfaction_level,last_evaluation,number_project,average_montly_hours, time_spend_company,Work_accident,left,promotion_last_5years,Departments, salary
0.38,0.53,2,157,3,0,1,0,sales,low
0.80,0.86,5,262,6,0,1,0,sales,medium
0.11,0.88,7,272,4,0,1,0, sales,medium
0.72,0.87,5,223,5,0,1,0, sales,low
0.37,0.52,2,159,3,0,1,0,sales,low"""

#DIMENSIONALITY REDUCTION TECHNIQUES
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA, FastICA, NMF, FactorAnalysis
penguins = sns.load_dataset("penguins")
penguins = (penguins.dropna())
penguins.head()
data = (penguins.select_dtypes(np.number))
data.head()
random_state = 0
pca_pl = make_pipeline(StandardScaler(),PCA(n_components=2,random_state=random_state))
pcs = pca_pl.fit_transform(data)
pcs[0:5,:]
pcs_df = pd.DataFrame(data = pcs ,columns = ['PC1', 'PC2'])
pcs_df['Species'] = penguins.species.values
pcs_df['Sex'] = penguins.sex.values
pcs_df.head()
plt.figure(figsize=(12,10))
with sns.plotting_context("talk",font_scale=1.25):
    sns.scatterplot(x="PC1", y="PC2",data=pcs_df,hue="Species",style="Sex",s=100)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA", size=24)
plt.savefig("PCA_Example_in_Python.png",format='png',dpi=75)
plt.show()

#BAYESIAN NETWORK
import numpy as np import csv
import pandas as pd

from pgmpy.models import BayesianModel

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination #read Cleveland Heart Disease data
heartDisease = pd.read_csv('heart.csv') heartDisease = heartDisease.replace('?',np.nan) #display the data
print('Few examples from the dataset are given below') print(heartDisease.head())
#Model Bayesian Network
Model=BayesianModel([('age','trestbps'),('age','fbs'),

('sex','trestbps'),('exang','trestbps'),('trestbps','heartdise

ase'),('fbs','heartdisease'),('heartdisease','restecg'),

('heartdisease','thalach'),('heartdisease','chol')])

#Learning CPDs using Maximum Likelihood Estimators print('\n Learning CPD using Maximum likelihood estimators') model.fit(heartDisease,estimator=MaximumLikelihoodEstimator) # Inferencing with Bayesian Network
print('\n Inferencing with Bayesian Network:') HeartDisease_infer = VariableElimination(model)
#computing the Probability of HeartDisease given Age print('\n 1. Probability of HeartDisease given Age=30')
q=HeartDisease_infer.query(variables=['heartdisease'],evidence

={'age':28})

print(q['heartdisease'])

#computing the Probability of HeartDisease given cholesterol 
print('\n 2. Probability of HeartDisease given cholesterol=100') q=HeartDisease_infer.query(variables=['heartdisease'],evidence={'chol':100})

print(q['heartdisease'])

#OBSTACLE DETECTION
import cv2
import time
from ultralytics import YOLO
import pyttsx3

# Load YOLO model
model = YOLO("yolov8n.pt")

# Start camera (you can replace with IP cam URL)
cap = cv2.VideoCapture(0)

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty("rate", 160)

def speak(text):
    print("🔊", text)
    engine.say(text)
    engine.runAndWait()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    H, W = frame.shape[:2]

    # Detect objects
    results = model.predict(frame, conf=0.5, verbose=False)
    detections = results[0].boxes

    if detections is not None:
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls)]
            cx = (x1 + x2) // 2

            # Detect position
            if cx < W / 3:
                position = "left"
                action = "move right"
            elif cx > 2 * W / 3:
                position = "right"
                action = "move left"
            else:
                position = "center"
                action = "move back"

            # Draw and announce
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({position})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            speak(f"{label} ahead on your {position}, {action}.")
            time.sleep(2)  # delay between alerts

    cv2.imshow("Obstacle Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()







