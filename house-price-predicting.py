##import libraries we need
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')
'**********************************************'
##Importing the dataset
dataset = pd.read_csv('C:\\Users\\Rahaf\\OneDrive\\Desktop\\project\\kc_house_data.csv\\kc_house_data.csv')
'**********************************************'
##check the features
# print(dataset.columns)
'**********************************************'
##checking the dataframe structure
# print(dataset.head())
'**********************************************'
#analysis price
#descriptive statistics summary(price)
# print(dataset['price'].describe())
'**********************************************'
#missing data
#checking data types and null vlues
#return the numbers of missing value in dataset
# print(dataset.info())
# print(dataset.isnull().sum()) #->There are no missing values detected from the dataset 
# '**********************************************'
# #histogram
# sns.distplot(dataset['price'])
# plt.show()
# '**********************************************'
# ##Relationship with numerical variables

# # #descriptive statistics summary
# plt.figure(figsize=(10,10))
# sns.jointplot(x=dataset.lat.values, y=dataset.long.values, size=10)
# plt.ylabel('Longitude', fontsize=12)
# plt.xlabel('Latitude', fontsize=12)
# plt.show()
# plt1 = plt()
# sns.despine
# '**********************************************'
# dataset['bedrooms'].value_counts().plot(kind='bar')
# plt.title('number of Bedroom')
# plt.xlabel('Bedrooms')
# plt.ylabel('Count')
# sns.despine
# plt.show()
# '**********************************************'
# dataset['bathrooms'].value_counts().plot(kind='bar')
# plt.title('number of Bathrooms')
# plt.xlabel('Bathrooms')
# plt.ylabel('Count')
# sns.despine
# plt.show()
'**********************************************'
##scatter plot sqft(living)/price
# var = 'sqft_living'
# data = pd.concat([dataset['price'], dataset[var]], axis=1)
# data.plot.scatter(x=var, y='price', ylim=(0,800000))
# plt.show()
# '**********************************************'
# ##scatter plot sqft(above)/price
# var = 'sqft_above'
# data = pd.concat([dataset['price'], dataset[var]], axis=1)
# data.plot.scatter(x=var, y='price', ylim=(0,800000))
# plt.show()
# '**********************************************'
# ##scatter plot lat/price
# plt.scatter(dataset.price,dataset.lat) 
# plt.xlabel("Price")
# plt.ylabel('Latitude')
# plt.title("Latitude vs Price")
# plt.show()
# '**********************************************'
# ##scatter plot condition/price
# plt.scatter(dataset.price,dataset.condition) #drop------------>important
# plt.xlabel("Price")
# plt.ylabel('condition')
# plt.title("condition vs Price")
# plt.show()
# '**********************************************'
# ##scatter plot waterfront/price
# plt.scatter(dataset.waterfront,dataset.price) #drop , id---------->important
# plt.title("Waterfront vs Price ( 0= no waterfront)")
# plt.show()
# '**********************************************'
# ##Relationship with categorical features
# ##box plot view/price
# var = 'view'
# data = pd.concat([dataset['price'], dataset[var]], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x=var, y="price", data=data)
# fig.axis(ymin=0, ymax=800000)
# plt.show()

# '**********************************************'
# ##Correlation matrix (heatmap style)
# corrmat = dataset.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True)
# plt.show()
# '**********************************************'
# #'price' correlation matrix (zoomed heatmap style)
# #saleprice correlation matrix
# k = 5 #number of variables for heatmap
# cols = corrmat.nlargest( k,'price')['price'].index
# cm = np.corrcoef(dataset[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()
# '**********************************************'
# ##Scatter plots between 'price' and correlated variables (move like Jagger style)
# #scatterplot
# sns.set()
# cols = ['price','sqft_living15', 'sqft_above', 'grade']
# sns.pairplot(dataset[cols], size = 2.5)
# plt.show()
# '**********************************************'
# ##Outliars!
# #standardizing data
# price_scaled = StandardScaler().fit_transform(dataset['price'][:,np.newaxis])
# low_range = price_scaled[price_scaled[:,0].argsort()][:10] #rearrange ascending
# high_range= price_scaled[price_scaled[:,0].argsort()][-10:] #rearange descending
# print('outer range (low) of the distribution:')
# print(low_range)
# print('\nouter range (high) of the distribution:')
# print(high_range)
# '**********************************************'
# ##to show outliers point
# '**********************************************'
# ##Bivariate analysis -------------------------------->important
# ##bivariate analysis price/sqft_living15
# var = 'sqft_living15'
# data = pd.concat([dataset['price'], dataset[var]], axis=1)
# data.plot.scatter(x=var, y='price', ylim=(0,800000))
# plt.show()

# #deleting points
# print(dataset.sort_values(by = 'sqft_living15', ascending = False)[:2])
dataset = dataset.drop(dataset[dataset['id'] ==2524069078].index)
dataset = dataset.drop(dataset[dataset['id'] ==3303850390].index)
# '**********************************************'
# ##bivariate analysis price/sqft_above
# # var = 'sqft_above'
# data = pd.concat([dataset['price'], dataset[var]], axis=1)
# data.plot.scatter(x=var, y='price', ylim=(0,800000))
# plt.show()

# print(dataset.sort_values(by = 'sqft_above', ascending = False)[:2]) #------------------>important
dataset = dataset.drop(dataset[dataset['id'] ==1225069038].index)
dataset = dataset.drop(dataset[dataset['id'] ==9208900037].index)
# '**********************************************'
# ##In the search for normality
# ##histogram and normal probability plot
# sns.distplot(dataset['price'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(dataset['price'], plot=plt)
# plt.show()

# #applying log transformation #---------------------->important
dataset['price'] = np.log(dataset['price'])
# '**********************************************'
# ##show after log
# ##In the search for normality
##histogram and normal probability plot
# sns.distplot(dataset['price'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(dataset['price'], plot=plt)
# plt.show()
# '**********************************************'
# ##histogram and normal probability plot
# sns.distplot(dataset['grade'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(dataset['grade'], plot=plt)
# plt.show()

# # #applying log transformation ---------------------->important
# dataset['grade'] = np.log(dataset['grade'])

# sns.distplot(dataset['grade'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(dataset['grade'], plot=plt)
# plt.show()
# '**********************************************'
# #histogram and normal probability plot
# sns.distplot(dataset['sqft_living15'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(dataset['sqft_living15'], plot=plt)
# plt.show()

# #applying log transformation ------------------------------->important
dataset['sqft_living15'] = np.log(dataset['sqft_living15'])

# sns.distplot(dataset['sqft_living15'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(dataset['sqft_living15'], plot=plt)
# plt.show()

# '**********************************************'
# #histogram and normal probability plot
# sns.distplot(dataset['sqft_above'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(dataset['sqft_above'], plot=plt)
# plt.show()

##applying log transformation

dataset['sqft_above'] = np.log(dataset['sqft_above'])

# sns.distplot(dataset['sqft_above'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(dataset['sqft_above'], plot=plt)
# plt.show()
# '**********************************************'
labels = dataset['price']
conv_dates = [1 if values == 2014 else 0 for values in dataset.date ]
dataset['date'] = conv_dates
# '**********************************************'
# ## drop not important features
new_data = dataset.drop(['id', 'waterfront','condition','price','sqft_lot'],axis=1)
print('data', new_data.shape) 
# '**********************************************'
# #analysis price
# #descriptive statistics summary(price)
# print(new_data['price'].describe())
# '**********************************************' #ending preprocessing and analysis
# '**********************************************'
# '**********************************************'

# #Splitting Dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(new_data, labels, random_state=42, train_size=0.7, shuffle=True)

print('train',X_train.shape, new_data.columns)
#Model1 (Linear Regression)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error

# model = LinearRegression()
# model.fit(X_train, y_train)
# #Calculating Details
# print('Linear Regression Train Score is : ' , model.score(X_train, y_train))
# print('Linear Regression Test Score is : ' , model.score(X_test, y_test))
# print('Linear Regression Coef is : ' , model.coef_)
# print('Linear Regression intercept is : ' , model.intercept_)
# print('----------------------------------------------------')
# #Calculating Prediction
# y_pred = model.predict(X_test)

# print('Predicted Value for Linear Regression is : ' , y_pred[:10])
# #Calculating Mean Absolute Error
# MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
# print('Mean Absolute Error Value is : ', MAEValue)

# #----------------------------------------------------
# #Calculating Mean Squared Error
# MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
# print('Mean Squared Error Value is : ', MSEValue)

# #----------------------------------------------------
# #Calculating Median Squared Error
# MdSEValue = median_absolute_error(y_test, y_pred)
# print('Median Squared Error Value is : ', MdSEValue )
# # #----------------------------------------------------
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_train2 = poly_reg.fit_transform(X_train)#------------------------->important
X_test2 = poly_reg.fit_transform(X_test)#--------------------------->important
# No Polynomial for y
from sklearn.linear_model import LinearRegression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_train2, y_train )
print(lin_reg_2.score(X_train2, y_train))
print(lin_reg_2.score(X_test2, y_test))
y_pred2 = lin_reg_2.predict(X_test2)
print(y_pred2)
print(lin_reg_2.coef_,'\n', lin_reg_2.intercept_)
joblib.dump(lin_reg_2, 'reg_2.sav')
# #----------------------------------------------------
# from sklearn.metrics import mean_absolute_error
# print(mean_absolute_error(y_test, y_pred2))

# from sklearn.metrics import mean_squared_error
# print(mean_squared_error(y_test, y_pred2))

# from sklearn.metrics import median_absolute_error
# print(median_absolute_error(y_test, y_pred2))
#----------------------------------------------------
#Applying Ridge Regression Model 
# from sklearn.linear_model import Ridge
# RidgeRegressionModel = Ridge(alpha=0.01,random_state=33)
# RidgeRegressionModel.fit(X_train, y_train)
# #Calculating Details
# print('Ridge Regression Train Score is : ' , RidgeRegressionModel.score(X_train, y_train))
# print('Ridge Regression Test Score is : ' , RidgeRegressionModel.score(X_test, y_test))

# print('Ridge Regression Coef is : ' , RidgeRegressionModel.coef_)
# print('Ridge Regression intercept is : ' , RidgeRegressionModel.intercept_)
# # print('----------------------------------------------------')

# # #Calculating Prediction
# y_pred = RidgeRegressionModel.predict(X_test)
# print('Predicted Value for Ridge Regression is : ' , y_pred[:10])

# #Calculating Mean Absolute Error
# MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
# print('Mean Absolute Error Value is : ', MAEValue)

# #----------------------------------------------------
# #Calculating Mean Squared Error
# MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
# print('Mean Squared Error Value is : ', MSEValue)

# #----------------------------------------------------
# #Calculating Median Squared Error
# MdSEValue = median_absolute_error(y_test, y_pred)
# print('Median Squared Error Value is : ', MdSEValue )


#Applying Lasso Regression Model
'----------------------------------------------------'
# from sklearn.linear_model import Lasso
# LassoRegressionModel = Lasso(alpha=0.02,random_state=33,normalize=False)
# LassoRegressionModel.fit(X_train, y_train)
# #Calculating Details
# print('Lasso Regression Train Score is : ' , LassoRegressionModel.score(X_train, y_train))
# print('Lasso Regression Test Score is : ' , LassoRegressionModel.score(X_test, y_test))
# print('Lasso Regression Coef is : ' , LassoRegressionModel.coef_)
# print('Lasso Regression intercept is : ' , LassoRegressionModel.intercept_)
# print('----------------------------------------------------')
# #Calculating Prediction
# y_pred = LassoRegressionModel.predict(X_test)
# print('Predicted Value for Lasso Regression is : ' , y_pred[:10])


# #Calculating Mean Absolute Error
# MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
# print('Mean Absolute Error Value is : ', MAEValue)

# #----------------------------------------------------
# #Calculating Mean Squared Error
# MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
# print('Mean Squared Error Value is : ', MSEValue)

# #----------------------------------------------------
# #Calculating Median Squared Error
# MdSEValue = median_absolute_error(y_test, y_pred)
# print('Median Squared Error Value is : ', MdSEValue )
# '----------------------------------------------------'
#Linear Regression with Normalization
# from sklearn.preprocessing import StandardScaler
# X_train_scale = pd.DataFrame(StandardScaler().fit_transform(X_train), columns = X_train.columns)
# X_train_scale.set_index(X_train.index, inplace = True)
# X_test_scale = pd.DataFrame(StandardScaler().fit_transform(X_test), columns = X_test.columns)
# X_test_scale.set_index(X_test.index, inplace = True)
# model_norm = LinearRegression(normalize=False)
# model_norm.fit(X_train_scale, y_train)
# print(model_norm.score(X_train_scale, y_train))
'***********************************************************************************'
#Applying SGDRegressor Model -------------------------------------> so baaad
# from sklearn.linear_model import SGDRegressor
# SGDRegressionModel = SGDRegressor(alpha=0.1,random_state=33,penalty='l2',loss = 'huber')
# SGDRegressionModel.fit(X_train, y_train)

# #Calculating Details
# print('SGD Regression Train Score is : ' , SGDRegressionModel.score(X_train, y_train))
# print('SGD Regression Test Score is : ' , SGDRegressionModel.score(X_test, y_test))
# print('SGD Regression Coef is : ' , SGDRegressionModel.coef_)
# print('SGD Regression intercept is : ' , SGDRegressionModel.intercept_)
# print('----------------------------------------------------')

# #Calculating Prediction
# y_pred = SGDRegressionModel.predict(X_test)
# print('Predicted Value for SGD Regression is : ' , y_pred[:10])
# #Calculating Mean Absolute Error
# MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
# print('Mean Absolute Error Value is : ', MAEValue)

# #----------------------------------------------------
# #Calculating Mean Squared Error
# MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
# print('Mean Squared Error Value is : ', MSEValue)

# #----------------------------------------------------
# #Calculating Median Squared Error
# MdSEValue = median_absolute_error(y_test, y_pred)
# print('Median Squared Error Value is : ', MdSEValue )

































'**********************************************'



















