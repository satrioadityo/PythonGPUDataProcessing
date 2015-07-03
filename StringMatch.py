from timeit import default_timer as timer
import numpy as np
import sys
from numbapro import vectorize, float64, float32, void, cuda
from numba import *
import csv

@cuda.jit(argtypes=[f8[:], f8[:], f8[:]], target='gpu')
def cuda_compare(a, b, c):
    i = cuda.grid(1)
    if (a[i] == b[i]) :
        c[i] = a[i]
    else :
        c[i] = 0

@cuda.jit(argtypes=[f8[:], f8[:], f8[:]], target='gpu')
def cuda_match(a, b, c):
    i = cuda.grid(1)
    for j in range (len(b)) :
        if (a[i] == b[j]) :
            c[i] = a[i]
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
N = griddim[0] * blockdim[0]
M = griddim[1] * blockdim[1]
print N,M
file1 = raw_input("Nama File 1 : ");
file2 =	raw_input("Nama File 2 : ");

f = open(file1,'r')
f2 = open(file2,'r')
#f = open('acc_master.csv','r')
#f2 = open('acc_ref.csv','r')
#f = open('sample1-10000.csv','r')
#f2 = open('sample2-10000.csv','r')
#f = open('acc_master_test.csv','r')
#f2 = open('acc_ref_test.csv','r')

list_master = []
list_ref = []

loadTime = timer()
print f
for line in f:
	#print line
	list_master.append(float64(line))

print f2
for line in f2:
	#print line
	list_ref.append(float64(line))

endofLoadTime = timer()

print 'loading time = ', endofLoadTime - loadTime
f.close()
f2.close()	
# process of matching
#cuda_compare_configured = cuda_compare.configure(griddim, blockdim)
cuda_match_configured = cuda_match.configure(griddim, blockdim)

aa = np.asarray(list_master,dtype=np.float64)
bb = np.asarray(list_ref,dtype=np.float64)
cc = np.empty_like(aa,dtype=np.float64)

timeStart = timer() # start count time

#cuda_compare_configured(aa, bb, cc)
print 'start of process matching in gpu'
stream = cuda.stream()

cc2 = []

for i in range(aa.size/100000) :
    cc2.append([])
    with stream.auto_synchronize():
        cuda_match_configured(aa, bb, cc)
    cc2[i].append(cc)
#cuda_match_configured(aa, bb, cc)
#cuda_match_cpu(aa, bb, cc)
print 'end of process matching in gpu'

timeFinish = timer() # end count time

#print str(cc)

print 'hasil akhir matching :'
count = 0
for i in range(len(cc2)) :
    for j in range(len(cc2[i])) :
        for k in range(len(cc2[i][j])) :
        #if cc2[i][j] != 0 :
            if cc2[i][j][k] != 0 and cc2[i][j][k] > 0 :
                #print cc2[i][j][k]
                count += 1

print 'jumlah data ',file1,' = ',len(list_master)
print 'jumlah data ',file2,' = ',len(list_ref)
print 'ada ',count,' data yang match'
print 'execution time gpu = ', timeFinish - timeStart,' detik'
#print 'execution time cpu = ', timeFinish - timeStart


save = raw_input('simpan id str match ke file? (y/n)')
if save == 'y' :
    fileName = raw_input('nama file ? ')
    for i in range(len(cc2)) :
        for j in range(len(cc2[i])) :
            for k in range(len(cc2[i][j])) :
            #if cc2[i][j] != 0 :
                if cc2[i][j][k] != 0 and cc2[i][j][k] > 0 :
                    f = open(fileName, 'ab')
                    f.write(str(cc2[i][j][k])+'\n')
    print 'Thanks for using our program'
else :
    print 'Thanks for using our program'