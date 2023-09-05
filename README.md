# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
# Step 1 :
Import the standard libraries such as pandas module to read the corresponding csv file.

# Step 2 :
Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

# Step 3 :
Import LabelEncoder and encode the corresponding dataset values.

# Step 4 :

Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y.

# Step 5 :
Predict the values of array using the variable y_pred.

# Step 6 :
Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.

# Step 7 :
Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report.

# Step 8: End the program.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VAISHNAVI S
RegisterNumber: 212222230165 
*/
```
```
import pandas as pd

data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x
y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
# HEAD OF THE DATA :
![Screenshot 2023-09-05 153300](https://github.com/Vaishnavi-saravanan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541897/5c3576b4-f2ce-404e-b572-d6abe32886e8)


# COPY HEAD OF THE DATA :
![Screenshot 2023-09-05 153312](https://github.com/Vaishnavi-saravanan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541897/0b5d7591-11cf-4844-b89d-1940f07d1253)


# NULL AND SUM :
![Screenshot 2023-09-05 153320](https://github.com/Vaishnavi-saravanan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541897/764eff2d-fff3-426f-b786-24e4d99ebd64)


# DUPLICATED :
![Screenshot 2023-09-05 153331](https://github.com/Vaishnavi-saravanan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541897/778be6b7-c7bf-498d-b6a4-e26bc68ce5cb)

# X VALUE :
![Screenshot 2023-09-05 153349](https://github.com/Vaishnavi-saravanan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541897/362a759a-bad9-4b57-a328-ac84f99512cb)

# Y VALUE :
![Screenshot 2023-09-05 153355](https://github.com/Vaishnavi-saravanan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541897/a9ed480d-b0fc-45b1-878b-945962687b24)

# PREDICTED VALUES :
![Screenshot 2023-09-05 153404](https://github.com/Vaishnavi-saravanan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541897/c44241d5-79c9-4af3-8037-d587232f708a)

# ACCURACY :
![Screenshot 2023-09-05 153643](https://github.com/Vaishnavi-saravanan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541897/8f15756a-6cc1-4044-8d14-4c984b106547)

# CONFUSION MATRIX :
![Screenshot 2023-09-05 153648](https://github.com/Vaishnavi-saravanan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541897/c8ff8fcb-63a1-4de4-9a2a-52eff7ce0075)
![Screenshot 2023-09-05 153717](https://github.com/Vaishnavi-saravanan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541897/b8960013-4d5e-46de-9043-54a4f0864448)
# CLASSIFICATION REPORT :
![Screenshot 2023-09-05 154014](https://github.com/Vaishnavi-saravanan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541897/bbf79709-eba0-437b-918e-b0c297b66ee3)

# RESULT :
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
