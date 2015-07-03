from timeit import default_timer as timer
import numpy as np
import sys, math
from numbapro import vectorize, float64, float32, void, cuda
from numba import *

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

file1 = raw_input("Nama File 1 : ");
file2 =	raw_input("Nama File 2 : ");
runningMethod =	raw_input("Run menggunakan gpu/cpu (g or c) : ");

f = open(file1,'r')
f2 = open(file2,'r')

list_master = []
list_ref = []

loadTime = timer()
print f
for line in f:
	list_master.append(float64(line))

print f2
for line in f2:
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

stream = cuda.stream()
cc2 = []

if runningMethod == 'g' :
    print 'start of process matching in gpu'
    p = 0
    for i in range(int(math.ceil(aa.size/100000.0))) :
        cc2.append([])
        with stream.auto_synchronize():
            cuda_match_configured(aa[p:p+102399], bb, cc)
        cc2[i].append(cc)
        p+=100000
    print 'end of process matching in gpu'
    
elif runningMethod == 'c' :
    print 'start of process matching in cpu'
    cuda_match_cpu(aa, bb, cc)
    print 'end of process matching in cpu'
else :
    sys.exit

timeFinish = timer() # end count time

print 'hasil akhir matching :'
if (runningMethod == 'g') :
    print 'execution time gpu = ', timeFinish - timeStart,' detik'
    for j in range (len(cc2)) :
        for k in range(len(cc2[j])) :
            print len(cc2[j][k])
    
    count = 0
    for i in range(len(cc2)) :
        for j in range(len(cc2[i])) :
            count += len(cc2[i][j])-cc2[i][j].tolist().count(0)


    print 'jumlah data ',file1,' = ',len(list_master)
    print 'jumlah data ',file2,' = ',len(list_ref)
    print 'ada ',count,' data yang match'
    
elif runningMethod == 'c' :
    count = 0
    for i in range(len(cc)) :
        if cc[i] != 0 and cc[i] > 0 :
            cc2.append(cc[i])
            count += 1
    print 'jumlah data ',file1,' = ',len(list_master)
    print 'jumlah data ',file2,' = ',len(list_ref)
    print 'ada ',count,' data yang match'
    print 'execution time cpu = ', timeFinish - timeStart,' detik'

save = raw_input('simpan id str match ke file? (y/n)')
if save == 'y' :
    if runningMethod == 'g' :
        fileName = raw_input('nama file ? ')
        for i in range(len(cc2)) :
            for j in range(len(cc2[i])) :
                for k in range(len(cc2[i][j])) :
                #if cc2[i][j] != 0 :
                    if cc2[i][j][k] != 0 and cc2[i][j][k] > 0 :
                        f = open(fileName, 'ab')
                        f.write(str(cc2[i][j][k])+'\n')
        print 'Thanks for using our program'
    elif runningMethod == 'c' :
        fileName = raw_input('nama file ? ')
        for i in range(len(cc2)) :
            f = open(fileName, 'ab')
            f.write(str(cc2[i])+'\n')
        print 'Thanks for using our program'
else :
    print 'Thanks for using our program'