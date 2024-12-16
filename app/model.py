import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

df = pd.read_csv("data/tips.csv")
print(df.head())

print(df.isnull().sum())

lb = LabelEncoder()
df['sex'] = lb.fit_transform(df['sex'])
df['smoker'] = lb.fit_transform(df['smoker'])
df['day'] = lb.fit_transform(df['day'])
df['time'] = lb.fit_transform(df['time'])

x = df.drop(columns = ['total_bill'], axis = 1) # Input Data
y = df['total_bill'] # Target Data

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

lr=LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

mse = mean_squared_error(y_pred, y_test)
score = r2_score(y_pred,y_test)
print(mse, score)

# Save the model
joblib.dump(lr, 'models/tips_model.pkl')
