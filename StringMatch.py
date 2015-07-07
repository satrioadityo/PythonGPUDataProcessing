# Library yang digunakan untuk mengimplementasikan matching string cuda python
import MySQLdb
import numpy as np
import sys, math
from numbapro import vectorize, float64, float32, void, cuda
from numba import *
from timeit import default_timer as timer

# Fungsi cuda_match :
# fungsi yang digunakan untuk matching id string yang diproses menggunakan gpu
@cuda.jit(argtypes=[f8[:], f8[:], f8[:]], target='gpu')
def cuda_match(a, b, c):
    i = cuda.grid(1)
    for j in range (len(b)) :
        if (a[i] == b[j]) :
            c[i] = b[j]
            break
        else :
            c[i] = 0

# Fungsi cuda_match_cpu :
# fungsi yang digunakan untuk matching id string yang diproses menggunakan cpu
def cuda_match_cpu(a, b, c) :
    for i in range(len(a)) :
        #print i
        for j in range(len(b)) :
            if a[i] == b[j] :
                c[i] = a[i]
                break
            else :
                c[i] = 0


# griddim adalah jumlah thread yang akan digunakan tiap block
griddim = 100, 1
# blockdim adalah jumlah block dalam 1 grid
blockdim = 1024, 1

list_master = [] # arraylist untuk menampung id string
list_ref = []    # arraylist untuk menampung id string
cc2 = []         # arraylist untuk menampung hasil matching id string

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

# masukan untuk program, database mana yang akan digunakan
dbMasterName = raw_input("Masukkan DB Master : ")
tblMasterName = raw_input("Masukkan Table Name : ")
LoadDataMaster(dbMasterName, tblMasterName)     # proses load data
dbRefName = raw_input("Masukkan DB Reference : ")
tblRefName = raw_input("Masukkan Table Name : ")
LoadDataRef(dbRefName, tblRefName)              # proses load data


# pilihan untuk menentukan methode running program yang akan digunakan
runningMethod =	raw_input("Run menggunakan GPU/CPU (g or c) : ");

# process of matching
#cuda_compare_configured = cuda_compare.configure(griddim, blockdim)
cuda_match_configured = cuda_match.configure(griddim, blockdim)

# karna gpu menggunakan konsep array oriented, maka konversikan list menjadi array
aa = np.asarray(list_master,dtype=np.float64)
bb = np.asarray(list_ref,dtype=np.float64)
cc = np.empty_like(aa,dtype=np.float64)


#jika metode gpu yang dipilih
if runningMethod == 'g' :
    print '\n*** Start of process matching in GPU ***'
    timeStart = timer() # start count time
    stream = cuda.stream()
    p = 0
    # pemrosesan data dilakukan secara iteratif, per 100000 data
    for i in range(int(math.ceil(aa.size/100000.0))) :
        cc2.append([])
        # proses menjalankan fungsi matching
        with stream.auto_synchronize():
            cuda_match_configured(aa[p:p+102399], bb, cc)
        cc2[i].append(cc)
        p+=100000
    timeFinish = timer() # end count time
    print '*** End of process matching in GPU ***'
    
#jika metode cpu yang dipilih
elif runningMethod == 'c' :
    print '\n*** Start of process matching in CPU ***'
    timeStart = timer() # start count time
    cuda_match_cpu(aa, bb, cc)
    timeFinish = timer() # end count time
    print '*** End of process matching in CPU ***'
else :
    sys.exit

print '\nHasil akhir matching :'
print '======================================================'
if (runningMethod == 'g') :
    print 'Execution time GPU = ', timeFinish - timeStart,' detik'    
    count = 0
    # hitung jumlah yang match
    for i in range(len(cc2)) :
        for j in range(len(cc2[i])) :
            count += len(cc2[i][j])-cc2[i][j].tolist().count(0)


    print 'Jumlah data di table',tblMasterName,' = ',len(list_master)
    print 'Jumlah data di table',tblRefName,' = ',len(list_ref)
    print 'Ada ',count,' data yang match'
elif runningMethod == 'c' :
    print 'Execution time CPU = ', timeFinish - timeStart,' detik'
    count = 0
    # hitung julah yang match
    for i in range(len(cc)) :
        if cc[i] != 0 and cc[i] > 0 :
            cc2.append(cc[i])
            count += 1
    print 'Jumlah data di table',tblMasterName,' = ',len(list_master)
    print 'Jumlah data di table',tblRefName,' = ',len(list_ref)
    print 'Ada ',count,' data yang match'
print '======================================================\n'
    
#pilihan apakah hasil yang match akan disimpan?
save = raw_input('simpan id str match ke file? (y/n)')
if save == 'y' :
    # proses penyimpanan jika user memilih ingin menyimpan
    if runningMethod == 'g' :
        dBase = raw_input("Masukkan Nama Database : ")
        fileName = raw_input("Masukkan Nama Table : ")
        db = MySQLdb.connect("localhost", "root", "", dBase)
        cursor = db.cursor()
        for i in range(len(cc2)) :
            for j in range(len(cc2[i])) :
                for k in range(len(cc2[i][j])) :
                    if cc2[i][j][k] != 0 and cc2[i][j][k] > 0 :
                        cursor.execute("INSERT into %s values (null, %d) "% (fileName, int(cc2[i][j][k] )))
                        db.commit()
        print 'Thanks for using our program'
    # proses penyimpanan jika user memilih ingin menyimpan
    elif runningMethod == 'c' :
        dBase = raw_input("Masukkan Nama Database : ")
        fileName = raw_input("Masukkan Nama Table : ")
        db = MySQLdb.connect("localhost", "root", "", dBase)
        cursor = db.cursor()
        for i in range(len(cc2)) :
            cursor.execute("INSERT into %s values (null, %d) "% (fileName, int(cc2[i])))
            db.commit()
        print 'Thanks for using our program'
# jika tidak ingin disimpan, keluar
else :
    print 'Thanks for using our program'