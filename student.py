import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
data = pd.read_csv("student-por.csv")
X = data[["studytime"]]
y = data["G3"]
model = LinearRegression()
model.fit(X, y)
prediction = model.predict(pd.DataFrame([[5]], columns=["studytime"]))
print("Predicted G3:", prediction[0])