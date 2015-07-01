from timeit import default_timer as timer
import numpy
from numbapro import vectorize, float64, void

@vectorize([float64(float64, float64)], target='gpu')
def vector_comp(a, b):
    if a == b :
        return  a
    else :
        return 0

@vectorize([void(float64[:], float64[:])], target='gpu')
def vector_comp2(a, b):
    match = 0
    for i in range(len(a)):
        for j in range(len(b)):
            #print list_master[i],"master"
            #print list_ref[j],"ref"
            if a[i] == b[j] :
                match = a[i]
                print match

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
	
print 'hasil compare tiap element'
#compare
count = 0
match = []

timeStart = timer()

vector_comp2(list_master,list_ref)

#for i in range(len(list_master)):
#    for j in range(len(list_ref)):
        #print list_master[i],"master"
        #print list_ref[j],"ref"
#        aaa = vector_comp(list_master[i], list_ref[j])
#        if aaa != 0 :
#            match.append(str(list_master[i]))
timeFinish = timer()

for i in range(len(match)):
    print str(match[i])

print 'execution time = ', timeFinish - timeStart