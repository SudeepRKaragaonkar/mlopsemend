from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
iris=load_iris()
x,y=iris.data,iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression(max_iter=200)
model.fit(x_train,y_train)
joblib.dump(model,"iris_model.joblib")
print("Model trained and saved as iris_model.joblib")
print("Training accuracy:", model.score(x_train, y_train))