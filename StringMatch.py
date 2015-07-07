from timeit import default_timer as timer
import numpy as np
import sys, math
from numbapro import vectorize, float64, float32, void, cuda
from numba import *
import MySQLdb

@cuda.jit(argtypes=[f8[:], f8[:], f8[:]], target='gpu')
def cuda_match(a, b, c):
    i = cuda.grid(1)
    for j in range (len(b)) :
        if (a[i] == b[j]) :
            c[i] = b[j]
            break
        else :
            c[i] = 0

def cuda_match_cpu(a, b, c) :
    for i in range(len(a)) :
        print i
        for j in range(len(b)) :
            if a[i] == b[j] :
                c[i] = a[i]
                break
            else :
                c[i] = 0


griddim = 100, 1
blockdim = 1024, 1
list_master = []
list_ref = []
cc2 = []

#load data master
def LoadDataMaster(db,table) :
    db = MySQLdb.connect("localhost", "root", "", db)
    cursor = db.cursor()
    sql = "SELECT idstr from %s"%table 
    idString = []
    try:
		cursor.execute(sql)
		results = cursor.fetchall()
		for row in results :
			list_master.append(float64(row[0]))
    except :
		print "Error"
    db.close

#load data ref
def LoadDataRef(db,table) :
    db = MySQLdb.connect("localhost", "root", "", db)
    cursor = db.cursor()
    sql = "SELECT idstr from %s"%table 
    idString = []
    try:
		cursor.execute(sql)
		results = cursor.fetchall()
		for row in results :
			list_ref.append(float64(row[0]))
    except :
		print "Error"
    db.close

dbMasterName = raw_input("Masukkan DB Master :")
tblMasterName = raw_input("Masukkan Table Name :")
LoadDataMaster(dbMasterName, tblMasterName)
dbRefName = raw_input("Masukkan DB Ref :")
tblRefName = raw_input("Masukkan Table Name :")
LoadDataRef(dbRefName, tblRefName)


runningMethod =	raw_input("Run menggunakan gpu/cpu (g or c) : ");

# process of matching
#cuda_compare_configured = cuda_compare.configure(griddim, blockdim)
cuda_match_configured = cuda_match.configure(griddim, blockdim)

aa = np.asarray(list_master,dtype=np.float64)
bb = np.asarray(list_ref,dtype=np.float64)
cc = np.empty_like(aa,dtype=np.float64)



#cuda_compare_configured(aa, bb, cc)


if runningMethod == 'g' :
    print 'start of process matching in gpu'
    timeStart = timer() # start count time
    stream = cuda.stream()
    p = 0
    for i in range(int(math.ceil(aa.size/100000.0))) :
        cc2.append([])
        with stream.auto_synchronize():
            cuda_match_configured(aa[p:p+102399], bb, cc)
        cc2[i].append(cc)
        p+=100000
    timeFinish = timer() # end count time
    print 'end of process matching in gpu'
    
elif runningMethod == 'c' :
    print 'start of process matching in cpu'
    timeStart = timer() # start count time
    cuda_match_cpu(aa, bb, cc)
    timeFinish = timer() # end count time
    print 'end of process matching in cpu'
else :
    sys.exit

print 'hasil akhir matching :'
if (runningMethod == 'g') :
    print 'execution time gpu = ', timeFinish - timeStart,' detik'    
    count = 0
    for i in range(len(cc2)) :
        for j in range(len(cc2[i])) :
            count += len(cc2[i][j])-cc2[i][j].tolist().count(0)


    print 'jumlah data di table',tblMasterName,' = ',len(list_master)
    print 'jumlah data di table',tblRefName,' = ',len(list_ref)
    print 'ada ',count,' data yang match'
elif runningMethod == 'c' :
    print 'execution time cpu = ', timeFinish - timeStart,' detik'
    count = 0
    for i in range(len(cc)) :
        if cc[i] != 0 and cc[i] > 0 :
            cc2.append(cc[i])
            count += 1
    print 'jumlah data di table',tblMasterName,' = ',len(list_master)
    print 'jumlah data di table',tblRefName,' = ',len(list_ref)
    print 'ada ',count,' data yang match'
    
save = raw_input('simpan id str match ke file? (y/n)')
if save == 'y' :
    if runningMethod == 'g' :
        dBase = raw_input("Masukkan Nama Database : ")
        fileName = raw_input("Masukkan Nama Table : ")
        db = MySQLdb.connect("localhost", "root", "", dBase)
        cursor = db.cursor()
        for i in range(len(cc2)) :
            for j in range(len(cc2[i])) :
                for k in range(len(cc2[i][j])) :
                #if cc2[i][j] != 0 :
                    if cc2[i][j][k] != 0 and cc2[i][j][k] > 0 :
                        cursor.execute("INSERT into %s values (null, %d) "% (fileName, int(cc2[i][j][k] )))
                        db.commit()
                        #f.write(str(cc2[i][j][k])+'\n')
        print 'Thanks for using our program'
    elif runningMethod == 'c' :
        dBase = raw_input("Masukkan Nama Database : ")
        fileName = raw_input("Masukkan Nama Table : ")
        db = MySQLdb.connect("localhost", "root", "", dBase)
        cursor = db.cursor()
        for i in range(len(cc2)) :
            #f = open(fileName, 'ab')
            cursor.execute("INSERT into %s values (null, %d) "% (fileName, int(cc2[i])))
            db.commit()
        print 'Thanks for using our program'
else :
    print 'Thanks for using our program'