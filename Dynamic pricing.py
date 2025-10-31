import pandas as pd         
import numpy  as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler 
from sklearn.ensemble import RandomForestRegressor as RFR 
from sklearn.metrics import r2_score

#load data 

data = pd.read_excel(r"C:\Users\subha\OneDrive\Documents\Dynamic prices.xlsx")

data.fillna(0)
print(data.head())
print(data.info())

# processing of data 

le = LabelEncoder()
data['Product_category']=le.fit_transform(data['Product_category'])
data['Demand']=le.fit_transform(data['Demand'])
data['Season']=le.fit_transform(data['Season'])

x = data.drop('Selling',axis=1)
y=data['Selling']

#training and testing sets 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43)

#model

model = RFR()
model.fit(x_train,y_train)





# Evaluation and performance of model
y_pred = model.predict(x_test)

print("performance",r2_score(y_test,y_pred))
print("Evaluation", y_pred)


#user input data 

product = input("enter product category ") 
rating = float(input('Enter product rating'))
demand = input("Enter demand for product")
season = input("Enter season type ")
supply = int(input("enter the amount of product to be supplied"))
discount = int(input("enter discount"))
competitor = int(input('enter price of competitor'))

#encoding 

product_encoded = le.fit_transform([product])
demand_encoded = le.fit_transform([demand])
season_encoded = le.fit_transform([season])

# make user input a dataset for prediction of price
user_data = {
    'Product_category':product_encoded,
    'Demand':demand_encoded,
    
    "Supply":supply,
    'Competitor_price':competitor,
    "Season":season_encoded,
    'Rating':rating,
    'Discount':discount
}

df = pd.DataFrame(user_data)

predicted_price = model.predict(df)

#predict 

print(f'Predicted selling price{predicted_price}')