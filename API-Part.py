from flask import Flask,request,jsonify
import requests
import json
import sqlite3
import pandas as pd 
import joblib
from sklearn.preprocessing import PolynomialFeatures
import firebase_admin
from firebase_admin import credentials, db



##update data
def update(ID,PRICE):
    conn=sqlite3.connect("database12.db")
    cursor=conn.cursor()
    sql =f'UPDATE HOUSES12 SET PRICE = {PRICE} where ID == {ID}'
    cursor.execute(sql)
    conn.commit()
    conn.close()
    return PRICE

app=Flask(__name__)

##Insert on table
@app.route("/insert",methods=['GET']) 
def insert():
    conn=sqlite3.connect("database12.db")
    cursor=conn.cursor()
    date=float(request.args.get('date'))
    bedrooms=int(request.args.get('bedrooms'))
    bathrooms=int(request.args.get('bathrooms'))
    sqft_living=int(request.args.get('sqft_living'))
    sqft_lot=int(request.args.get('sqft_lot'))
    floors=int(request.args.get('floors'))
    waterfront=int(request.args.get('waterfront'))
    view=int(request.args.get('view'))
    condition=int(request.args.get('condition'))
    grade=int(request.args.get('grade'))
    sqft_above=int(request.args.get('sqft_above'))
    sqft_basement=int(request.args.get('sqft_basement'))
    yr_built=int(request.args.get('yr_built'))
    yr_renovated=int(request.args.get('yr_renovated'))
    zipcode=int(request.args.get('zipcode'))
    lat=float(request.args.get('lat'))
    long=float(request.args.get('long'))
    sqft_living15=int(request.args.get('sqft_living15'))
    sqft_lot15=int(request.args.get('sqft_lot15'))
    price=555
    flag=0

    sql="""INSERT INTO HOUSES12 (DATE,PRICE,BEDROOMS,BATHROOMS,SQFT_LIVING,SQFT_LOT,FLOORS,WATERFRONT,VIEW,CONDITION,GRADE,SQFT_ABOVE,SQFT_BASEMENT,YR_BUILT,YR_RENOVATED,ZIPCODE,LAT,LONG,SQFT_LIVING15,SQFT_LOT15, flag) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
    data_tuple=(date,price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,grade,sqft_above,
    sqft_basement,yr_built,yr_renovated,zipcode,lat,long,sqft_living15,sqft_lot15, flag)
    cursor.execute(sql,data_tuple)
    conn.commit()
    conn.close()
    #preprocessing datainput 
    newdata1 = pd.DataFrame ([data_tuple], columns= ['date','price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15'])
    newdata2 = newdata1.drop(['waterfront','condition','price','sqft_lot'],axis=1)
    
    
    poly_reg = PolynomialFeatures(degree = 2)
    preprocessing_data= poly_reg.fit_transform(newdata2)

    #prediction
    pricemodel =joblib.load('reg_2.sav')
    price= pricemodel.predict(preprocessing_data)

    #update price value 
    id = cursor.lastrowid
    x= update(id,float(price))
    #copy on firebase
    #1) get data from local database
    conn=sqlite3.connect("database12.db")
    cursor=conn.cursor()
    query = f'''Select * from HOUSES12 where flag=0;'''
    rep = cursor.execute(query).fetchall()
    conn.commit()
    conn.close()
    #2)connect to firbase 
    try:
        cred = credentials.Certificate('C:\\Users\\Rahaf\\OneDrive\\Desktop\\project\\price-prediction-1bd14-firebase-adminsdk-4yog2-c6da0478dd.json')
        firebase_admin.initialize_app(cred,{'databaseURL':'https://price-prediction-1bd14-default-rtdb.firebaseio.com/','timeout':30})
        ref ="/"
        root =db.reference(ref)
        root.push(rep)
        conn=sqlite3.connect("database12.db")
        cursor=conn.cursor()
        query = f'''update HOUSES12 SET flag = 1 where flag=0;'''
        rep = cursor.execute(query).fetchall()
        conn.commit()
        conn.close()
        return str(price)
    except:
        return str(price)


'''http://127.0.0.1:5000/insert?date=2014-10-13&bedrooms=5&bathrooms=3&sqft_living=250&sqft_lot=150&floors=3&waterfront=0&view=1&condition=4&grade=6&sqft_above=200&sqft_basement=100&yr_built=2012&yr_renovated=2014&zipcode=1150&lat=1540&long=555&sqft_living15=600&sqft_lot15=200'''

app=Flask(__name__)

##report on table
@app.route("/report",methods=['GET']) 
def report():
    conn=sqlite3.connect("database12.db")
    cursor=conn.cursor()
    cond =request.args.get('Cond')
    query = f'''Select * from HOUSES12 where {cond};'''
    rep = cursor.execute(query).fetchall()
    conn.commit()
    conn.close()
    df = pd.DataFrame(rep,columns=['ID', 'date','price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15'])
    df.to_csv("report.csv")
    return str ("found report file on desktop")





if __name__=="__main__":
    app.run(debug=True)






































    







