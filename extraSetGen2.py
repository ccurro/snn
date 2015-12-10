# python extraSetGen2.py | awk 'BEGIN {srand()} !/^$/ { if (rand() <= .5) print $0 > "r.train"; else print $0 > "r.test"}'
import numpy as np
from scipy.stats import rayleigh
from scipy.stats import rice

def genSequence(dist, len):
	a = 0.1
	x = [0]
	for i in range(0,len):
		x.append(x[-1] + a*dist.rvs(size=1)[0])

	return x

nExamples = 500

for i in range(0,nExamples):
	seq = genSequence(rice(1),10)
	for e in seq:
		print('{:0.3f}'.format(e),end=' ')
	print('1')

	seq = genSequence(rayleigh,10)
	for e in seq:
		print('{:0.3f}'.format(e),end=' ')
	print('0')