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

train_X, test_X, train_y, test_y = train_test_split(x, y, random_state = 0, shuffle=False )

##Scaling the values
train_s = train_X.values
test_s = test_X.values

minmax = MinMaxScaler()

train_X = minmax.fit_transform(train_s)
test_X = minmax.transform(test_s)

##Fitting and predictions
forest_model = RandomForestRegressor(n_estimators=100, random_state=0)
forest_model.fit(train_X, train_y)

preds = forest_model.predict(test_X)

#Model accuracy
mae = np.round(mean_absolute_error(test_y, preds), 3)
print('Mean Absolute Error:', mae)
  
mse = np.round(mean_squared_error(test_y, preds),3)
print('Mean Squared Error:', mse)

score = np.round(r2_score(test_y, preds),3)
print('The accuracy is:', score)

## Plot
fig, ax = plt.subplots(figsize=(8, 5))
plt.scatter(test_y, preds, alpha = 0.50)
    plt.title('Actual Value & Predicted Value')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.xlim(0, 3.5)
plt.ylim(0, 3.5)
ax.spines['top'].set_visible(False) 
ax.spines['right'].set_visible(False)
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.9)
plt.show()

