import sqlite3
conn=sqlite3.connect("database12.db")
cur=conn.cursor()

sql='''CREATE TABLE "HOUSES12" (
    "ID"    INTEGER,
    "DATE"    STR,
    "PRICE"    float,
    "BEDROOMS"    INTEGER,
    "BATHROOMS"    INTEGER,
    "SQFT_LIVING"    INTEGER,
    "SQFT_LOT"    INTEGER,
    "FLOORS"    INTEGER,
    "WATERFRONT"    INTEGER,
    "VIEW"    INTEGER,
    "CONDITION"    INTEGER,
    "GRADE"    INTEGER,
    "SQFT_ABOVE"    INTEGER,
    "SQFT_BASEMENT"    INTEGER,
    "YR_BUILT"    INTEGER,
    "YR_RENOVATED"    INTEGER,
    "ZIPCODE"    INTEGER,
    "LAT"    INTEGER,
    "LONG"    INTEGER,
    "SQFT_LIVING15"    INTEGER,
    "SQFT_LOT15"    INTEGER,
    PRIMARY KEY("ID" AUTOINCREMENT)
);'''
cur.execute('''alter table HOUSES12 add column flag;''')
#print("housestable has been created")
conn.commit()
conn.close()