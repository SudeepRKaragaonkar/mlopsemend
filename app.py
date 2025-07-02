from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import streamlit as st
import numpy as np
iris=load_iris()
x,y=iris.data,iris.target
model=LogisticRegression(max_iter=200).fit(x,y)
st.title("Iris Flower Species Prediction")
inputs=[st.slider(label,min_value=val[0],max_value=val[1],value=val[2],step=0.1) for label,val in zip(["sepal length","sepal width","petal length","petal width"],[(4.0, 8.0, 5.1), (2.0, 4.5, 3.5), (1.0, 7.0, 1.4), (0.1, 2.5, 0.2)])]
if st.button("Predict"):
    result=model.predict([inputs])[0]
    st.success(f"The predicted species is: {iris.target_names[result]}")
