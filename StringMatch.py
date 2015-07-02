from timeit import default_timer as timer
import numpy as np
from numbapro import vectorize, float64, float32, void, cuda
from numba import *

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
    for k in range (len(b)) :
        if (a[i] == b[k]) :
            c[i] = a[i]
            break
        else :
            c[i] = 0


griddim = 50, 1
blockdim = 32, 1, 1
N = griddim[0] * blockdim[0]

#f = open('acc_master.csv','r')
#f2 = open('acc_ref.csv','r')
f = open('acc_master_test.txt','r')
f2 = open('acc_ref_test.txt','r')

list_master = []
list_ref = []

print 'isi file acc_master.csv'
for line in f:
	print line
	list_master.append(float64(line))

print 'isi file acc_ref.csv'
for line in f2:
	print line
	list_ref.append(float64(line))
	
# process of matching
#cuda_compare_configured = cuda_compare.configure(griddim, blockdim)
cuda_match_configured = cuda_match.configure(griddim, blockdim)

aa = np.asarray(list_master,dtype=np.float64)
bb = np.asarray(list_ref,dtype=np.float64)
cc = np.empty_like(aa,dtype=np.float64)
timeStart = timer() # start count time

#cuda_compare_configured(aa, bb, cc)
print 'start of matching in gpu'
cuda_match_configured(aa, bb, cc)
print 'end of matching in gpu'

timeFinish = timer() # end count time

#print str(cc)

print 'hasil akhir matching :'
for i in range(len(cc)) :
    if cc[i] != 0 :
        print str(cc[i])

print 'execution time = ', timeFinish - timeStart