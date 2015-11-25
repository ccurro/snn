import os
import sys
import csv
import numpy as np

class Input:
	def __init__(self,fName):
		self.l = []
		self.listPosition = 1
		with open(fName,'r') as csvfile:
			for row in csv.reader(csvfile, delimiter=' '):
				self.l.append(np.array(row,dtype=np.float64))

		self.nNodes = (self.l[0][1] + self.l[0][2]).astype(np.int)

	def getLayerSizes(self):
		return self.l[0].astype(np.int).tolist()

	def getNextNodesWeights(self):
		w = self.l[self.listPosition]
		self.listPosition += 1
		return w

def main():
	inobj = Input('sample.NNWDBC.init')

	print(inobj.getLayerSizes())

	for i in range(0,inobj.nNodes):
		print(inobj.getNextNodesWeights())


if __name__ == "__main__":
    main()