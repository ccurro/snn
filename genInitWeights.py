import numpy as np

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def initWeights(layerSizes):
	for i in range(0,len(layerSizes)):
		if i == len(layerSizes) -1:
			print(layerSizes[i])
		else:
			print(layerSizes[i], end=' ')

	for layer in range(1,len(layerSizes)):
		for node in range(layerSizes[layer]):
			print(' '.join(map(str,np.round(np.random.rand(layerSizes[layer-1]+1)/10,3))))

initWeights([11, 20, 1])