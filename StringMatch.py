from timeit import default_timer as timer
import numpy as np
from numbapro import vectorize, float64, float32, void, cuda
from numba import *

@cuda.jit(argtypes=[f4[:], f4[:], f4[:]])
def cuda_sum(a, b, c):
    i = cuda.grid(1)
    #i = 0
    c[i] = a[i] + b[i]


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
	list_master.append(float32(line))

print 'isi file acc_ref.csv'
for line in f2:
	print line
	list_ref.append(float32(line))
	
print 'hasil compare tiap element'
#compare
print "N", N
cuda_sum_configured = cuda_sum.configure(griddim, blockdim)
count = 0
match = []
a = list_master
print a
b = list_ref
print b
c = np.empty_like(a)
print c

a = np.array(np.random.random(5)*1000, dtype=np.float32)
print a
b = np.array(np.random.random(5)*1000, dtype=np.float32)
print b
c = np.empty_like(a)
#print c
timeStart = timer()

cuda_sum_configured(a, b, c)
print 'res '
print c
#cuda_sum_configured(a, b, c)
#vector_comp2(list_master,list_ref)

#for i in range(len(list_master)):
#    for j in range(len(list_ref)):
        #print list_master[i],"master"
        #print list_ref[j],"ref"
#        aaa = vector_comp(list_master[i], list_ref[j])
#        if aaa != 0 :
#            match.append(str(list_master[i]))
timeFinish = timer()

#for i in range(len(match)):
#    print str(match[i])

print 'execution time = ', timeFinish - timeStart