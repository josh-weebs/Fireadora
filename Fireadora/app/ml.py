import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("Forest_fire.csv")
data = np.array(data)

X = data[1:-1, 1:4]
y = data[1:-1, 4]
y = y.astype('int')
X = X.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()
lin_reg=LinearRegression()
svm_class=OneVsOneClassifier(SVC(random_state=0))
lin_reg.fit(X_train,y_train)

log_reg.fit(X_train, y_train)
svm_class.fit(X_train,y_train)
y_pred=lin_reg.predict(X_test)

print("\nAccuracy of linear regression:",lin_reg.score(X_test,y_test))
print("\nAccuracy of logistic regression:",log_reg.score(X_test,y_test))
print("\nAccuracy of support vector regression:",svm_class.score(X_test,y_test))
pickle.dump(log_reg,open('model.pkl','wb'))

