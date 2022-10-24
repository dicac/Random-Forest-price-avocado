#Loading libraries
import pandas as pd
from sklearn.preprocessing import LabelEncode
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

avodata =  pd.read_csv('/avocado.csv', index_col = 0)

labelEncoder = LabelEncoder()
data = avodata

for e in data.columns:
    if data[e].dtype == 'object':
        labelEncoder.fit(list(data[e].values))
        data[e] = labelEncoder.transform(data[e].values)
        
#Choosing target variable and predictor variables

x = data.drop(['AveragePrice', 'Date'], axis = 1)
y = data['AveragePrice']

#Splitting the data into training and test datasets

train_X, test_X, train_y, test_y = train_test_split(x, y, random_state = 0)

##Fitting and predictions
forest_model = RandomForestRegressor(n_estimators=100, random_state=0)
forest_model.fit(train_X, train_y)

preds = forest_model.predict(test_X)

#Model accuracy
mean_absolute_error(test_y, preds)
mean_squared_error(test_y, preds)
r2_score(test_y, preds)

