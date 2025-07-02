from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib
x,y=load_iris(return_X_y=True)
model=LogisticRegression(max_iter=200)
model.fit(x,y)
joblib.dump(model,"iris_model.joblib")

