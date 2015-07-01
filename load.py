f = open('acc_master.csv','r')
f2 = open('acc_ref.csv','r')

list_master = []
list_ref = []

print 'isi file acc_master.csv'
for line in f:
	print line
	list_master.append(line)

print 'isi file acc_ref.csv'
for line in f2:
	print line
	list_ref.append(line)
	
print 'hasil compare tiap element'
#compare
count = 0
for i in range(len(list_master)) :
	for j in range(len(list_ref)) :
		if list_master[i] == list_ref[j] :
			count+=1

print 'jumlah yang match ',count