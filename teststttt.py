
from flask import Flask,request,jsonify
import requests
import json
import sqlite3
import pandas as pd 
import joblib
from sklearn.preprocessing import PolynomialFeatures


conn=sqlite3.connect("database12.db")
cursor=conn.cursor()
# cond =request.args.get('Cond')
Cond='Date = "2014-10-13"'
query = f'''Select * from HOUSES1 where {Cond};'''
rep = cursor.execute(query).fetchall()
conn.commit()
conn.close()
print(rep)
# df = pd.DataFrame(rep,columns=['ID', 'date','price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15'])
# df.to_csv("report.csv")