# python extraSetGen.py | awk 'BEGIN {srand()} !/^$/ { if (rand() <= .5) print $0 > "mult.train"; else print $0 > "mult.test"}'
import numpy as np

nSym = 32
nBits = np.int(np.log2(nSym))

for i in range(0,nSym):
	for j in range(0,nSym):
		a = np.binary_repr(i,width=nBits)
		b = np.binary_repr(j,width=nBits)
		s = np.binary_repr(i*j,width=2*nBits)
		for e in a:
			print(e,end=' ')
		for e in b:
			print(e,end=' ')
		for i in range(0,len(s)):
			if i == len(s) -1:
				print(s[i])
			else:
				print(s[i], end=' ')