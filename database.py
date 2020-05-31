import os
import sqlite3
from datetime import datetime

# create a default path to connect to and create (if necessary) a database
# called 'database.sqlite3' in the same directory as this script

class databaseFPD():

    def __init__(self, dbname):
        self.dbname = dbname
        self.connect = sqlite3.connect(dbname)
        self.cur = self.connect.cursor() 

    def is_table(self, table_name):
        """ This method seems to be working now"""
        query = """SELECT sql FROM sqlite_master WHERE type='table'"""
        cursor =  self.cur.execute(query)
        result = cursor.fetchone()
        if result == None:
            return False
        else:
            return True 

    def createTable(self):

        date_sql = ''' CREATE TABLE Date (
            id integer PRIMARY KEY,
            date text NOT Null
        )'''

        self.cur.execute(date_sql)  

        product_sql = ''' CREATE TABLE Products  (
            id integer PRIMARY KEY,
            paw_id interger,
            date_id integer,
            area_paw integer,
            area_cyst integer,
            ratio float,
            grade integer,
            FOREIGN KEY (date_id) REFERENCES Date (id)

        )'''
              
        self.cur.execute(product_sql)

    def create_date(self, date):
        sql = '''INSERT INTO Date (date) VALUES (?)'''
        
        self.cur.execute(sql, [date])
        return self.cur.lastrowid

    def create_product(self, paw_id, date_id, area_paw, area_cyst, ratio, grade):

        sql = '''INSERT INTO Products (paw_id, date_id, area_paw, area_cyst, ratio, grade)
                VALUES (?, ?, ?, ?, ?, ?)'''

        self.cur.execute(sql, (paw_id, date_id, area_paw, area_cyst, ratio, grade))
        return self.cur.lastrowid

    def update_products(self, paw_id, date_id, area_paw, area_cyst, ratio, grade, position):

        update_sql = "UPDATE Products SET paw_id = ?, date_id = ?, area_paw = ?, area_cyst = ?, ratio = ?, grade = ? WHERE id = ?"
        self.cur.execute(update_sql, (paw_id, date_id, area_paw, area_cyst, ratio, grade, position))
        return self.cur.lastrowid

    def checkTime(self, checkDate):

        self.cur.execute("SELECT id, date FROM Date")
        results = self.cur.fetchall()

        if results == []:
            date_id = self.create_date(checkDate)

        else:
            dateInDB = results[-1][1] 
            if dateInDB == checkDate:
                date_id = results[-1][0]
            else:
                date_id = self.create_date(checkDate)

        return date_id


    def checkProduct(self, date_id, paw_id, area_paw, area_cyst, ratio, grade):
        self.cur.execute("SELECT id, paw_id, date_id FROM Products")
        results = self.cur.fetchall()
        checkPawExist = []
        
        if results == []:
            product = self.create_product(paw_id, date_id, area_paw, area_cyst, ratio, grade)

        else:
            for i in range(len(results)):
                checkPawExist.append(results[i][1])

            checkIdDate = results[-1][2]
            lastPaw =results[-1][1]
            if checkIdDate != date_id or int(paw_id) > lastPaw or int(paw_id) not in checkPawExist:
                product = self.create_product(paw_id, date_id, area_paw, area_cyst, ratio, grade)

            elif int(paw_id) <= lastPaw:
                position = self.callSomeInfo(date_id, paw_id)
                product = self.update_products(paw_id, date_id, area_paw, area_cyst, ratio, grade, position)

        return product
            
    def callProducts(self):
        self.cur.execute("SELECT id, paw_id, date_id, area_paw, area_cyst, ratio, grade FROM Products")
        results = self.cur.fetchall()
        return results

    def callDate(self):
        self.cur.execute("SELECT id, date FROM Date")
        results = self.cur.fetchall()
        return results

    def callSomeInfo(self, date_id, paw_id):
        find_sql = "SELECT id FROM Products WHERE paw_id = ? AND date_id = ?"
        self.cur.execute(find_sql, (paw_id, date_id))
        results = self.cur.fetchall()
        # print("date {} paw {}".format(date_id, paw_id))
        # print("callSomeInfo: ", results)
        return results[0][0]


    def main(self, resultByID, final_result):

        ###Check table in database file before create.
        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y")
        if self.is_table("Products") == False:
            self.createTable() #For create table

        try:
            date_id = self.checkTime(date_time)
            for ikey, value in final_result.items():
                for jkey, jvalue in final_result[ikey].items():
                    product = self.checkProduct(date_id, jkey, resultByID['2'][jkey], resultByID['1'][jkey], final_result['ratio'][jkey], final_result['grade'][jkey])

            self.connect.commit()
        except:
            # rollback all database actions since last commit
            self.connect.rollback()
            raise RuntimeError("Uh oh, an error occurred ...")


if __name__ == "__main__":
    DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 'FPD_Database_V2.db')
    dbFPD = databaseFPD(DEFAULT_PATH)
    
    resultByID = {'2': {'1': 510, '2': 3701}, '1': {'1': 0, '2': 800}}
    ratio = {'ratio': {'1': 0.0, '2': 3.6206430694406917}, 'grade': {'1': 1, '2': 2}}
    dbFPD.main(resultByID, ratio)

    callRes = dbFPD.callProducts()
    callDate = dbFPD.callDate()
    print(callRes)
    # print(callRes)
    


