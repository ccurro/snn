import os
import sys
import csv
import numpy as np
from pprint import pprint

class DataFeeder:
	def __init__(self,fName):
		self.l = []		
		self.listPosition = 1
		with open(fName,'r') as csvfile:
			for row in csv.reader(csvfile, delimiter=' '):
				self.l.append(np.array(row,dtype=np.float64))
		self.listMax = int(self.l[0][0])
		self.nClasses = int(self.l[0][2])

	def getNextExample(self):
		example = self.l[self.listPosition][0:-self.nClasses]
		target = self.l[self.listPosition][-self.nClasses:len(self.l[self.listPosition])].tolist()
		if self.listPosition == self.listMax:
			self.listPosition = 1
		else:
			self.listPosition += 1
		return example,target

def main():
	feed = DataFeeder('wdbc.train')
	pprint(vars(feed))
	for i in range(0,2*feed.listMax):
		example, target = feed.getNextExample()
		assert(np.size(example) == 30)
		assert(np.size(target) == 1)

if __name__ == "__main__":
    main()