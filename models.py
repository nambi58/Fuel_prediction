import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
df = pd.read_csv("fuel_data.csv")
x=df.filter(['drivenKM'])
y=df.filter(['fuelAmount'])
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train, y_train)
pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
