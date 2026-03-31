import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/Superstore.csv", encoding='latin1')
df['Order Date'] = pd.to_datetime(df['Order Date'])

monthly_sales = df.groupby(pd.Grouper(key='Order Date', freq='M'))['Sales'].sum().reset_index()

monthly_sales['Month'] = monthly_sales['Order Date'].dt.month
monthly_sales['Year'] = monthly_sales['Order Date'].dt.year

X = monthly_sales[['Month','Year']]
y = monthly_sales['Sales']

model = LinearRegression()
model.fit(X,y)

future = pd.DataFrame({
    'Month':[1,2,3,4,5,6],
    'Year':[2025]*6
})

predictions = model.predict(future)

future['Predicted Sales'] = predictions
future.to_csv("outputs/predictions.csv", index=False)

print("Done")
