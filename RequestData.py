import requests
# class ReguestData:
#     '''inserting into table''

#     def insert(self,date ,bedrooms ,bathrooms ,sqft_living ,sqft_lot ,floors ,waterfront ,view ,condition ,grade ,sqft_above ,sqft_basement ,yr_built ,yr_renovated ,zipcode ,lat ,long ,sqft_living15 ,sqft_lot15 ):

res=requests.get('http://127.0.0.1:5000/insert',params={'date':1,'bedrooms':3 ,'bathrooms':1 ,'sqft_living':1180 ,'sqft_lot':5650,'floors':1,'waterfront':0,'view':0,'condition':3 ,'grade':7 ,'sqft_above':1180 ,'sqft_basement':0,'yr_built':1955 ,'yr_renovated':0,'zipcode':98178,'lat':47.5112 ,'long':-122.257,'sqft_living15':1340,'sqft_lot15':5650})
        
print(res.status_code)
print(res.text)

# '''reprot data'''
        
# res=requests.get('http://127.0.0.1:5000/report',params={'Cond':'Date = "2014-10-13"'})

# print(res.status_code)
# print(res.text)


    

