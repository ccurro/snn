def getLearningRate():
	while True:
		try:
			learningRate = input('Please enter learning rate (positive number): ')
			learningRate = float(learningRate)
			if not learningRate > 0:
				raise ValueError
			return learningRate
		except ValueError as e:
			print('Learning rate not positive number, try again: ')

def getNEpochs():
	while True:
		try:
			nEpochs = input('Please enter number of epochs to train (positive integer): ')
			nEpochs = int(nEpochs)
			if not nEpochs > 0:
				raise ValueError
			return nEpochs
		except ValueError as e:
			print('Number of epochs is not a positive integer, try again: ')


def getFile(printStr):
	while True:
		try:
			fName = input(printStr)
			with open(fName) as file:
				return fName
		except IOError as e:
			print('Unable to open file, try again: ')

def getTrain():
	initFile = getFile('Please enter filename for the initial network: ')
	trainFile = getFile('Please enter filename for the training set: ')
	outFile = input('Please enter filename to save the trained network: ')
	learningRate = getLearningRate()
	nEpochs = getNEpochs()
	return initFile, trainFile, outFile, learningRate, nEpochs

def getTest():
	trainedFile = getFile('Please eneter filename for the trained network: ')
	testFile = getFile('Please enter filename for the test set: ')
	outFile = input('Please enter filename to save the test metrics: ')
	return trainedFile, testFile, outFile

if __name__ == '__main__':
	getTrain()
	getTest()