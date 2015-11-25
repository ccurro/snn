import numpy as np
import itertools
import input
import data 

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

class Node:
	def __init__(self,nInputs,feeder,isInput):
		if isInput:
			pass
		else:
			self.feeder = feeder
			self.initWeights()

	def initWeights(self):
		self.w = self.feeder.getNextNodesWeights()

	def sigmoid(self):
		self.activation = 1/(1+np.exp(-self.activation))

	def forward(self,features):
		assert(np.size(features) < np.size(self.w))
		self.features = np.concatenate([-np.ones((1)), features])	
		self.activation = np.inner(self.w,self.features)
		self.sigmoid()
		self.dsig = self.activation*(1 - self.activation)

	def update(self):
		self.grad = self.features*self.delta
		self.w = self.w + 0.1*self.grad

	def reset(self):
		# Only strictly necessary to del activation
		if hasattr(self,'activation'):
			del self.activation
			del self.features
			del self.delta
			del self.dsig
			del self.grad
		if hasattr(self,'label'):
			del self.label

class Layer:
	def __init__(self,layerSize,nInputs,feeder,isInput):
		self.nodes = []
		for i in range(0,layerSize):
			self.nodes.append(Node(nInputs,feeder,isInput))

	def forward(self,features):
		for node in self.nodes:
			node.forward(features)
			if hasattr(self,'activations'):
				self.activations = np.append(self.activations,node.activation)
			else:
				self.activations = np.array(node.activation)

	def backward(self):
		s = np.zeros(np.size(self.nodes[0].w))
		for i in range(1,np.size(self.nodes[0].w)):
			for node in self.nodes:
				s[i-1] = s[i-1] + node.w[i]*node.delta
		return s

	def reset(self):
		if hasattr(self,'activations'):
			del self.activations
		for node in self.nodes:
			node.reset()

class Network:
	def __init__(self,f):
		self.feeder = f
		layerSizes = self.feeder.getLayerSizes()
		for size in layerSizes:
			if hasattr(self,'layers'):
				self.layers.append(Layer(size,prevLayerSize+1,self.feeder,False))
				prevLayerSize = size
			else:
				self.layers = []
				self.layers.append(Layer(size,size,self.feeder,True))
				prevLayerSize = size

	def forward(self, features):
		if hasattr(self,'activations'):
			self.reset()

		for layer in self.layers:
			if hasattr(self,'activations'):
				layer.forward(self.activations)
				self.activations = layer.activations
			else:
				self.activations = features

	def backward(self,labels):
		outputLayer = self.layers[-1]
		for i in range(0,len(outputLayer.nodes)):
			outputLayer.nodes[i].label = labels[i]

		for node in outputLayer.nodes:
			node.delta = node.dsig*(node.label - node.activation)

		for l in reversed(range(1,len(self.layers)-1)):
			layer = self.layers[l]
			s = self.layers[l+1].backward()
			for i in range(0,len(layer.nodes)):
				layer.nodes[i].delta = layer.nodes[i].dsig*s[i]

		for layer in self.layers:
			for node in layer.nodes:
				if hasattr(node,'delta'):
					node.update()

	def reset(self):
		del self.activations
		for layer in self.layers:
			layer.reset()

inputFeeder = input.Input('sample.NNWDBC.init')
dataFeeder = data.DataFeeder('wdbc.train')
model = Network(inputFeeder)

for epoch in range(0,100):
	nCorrect = 0
	for example in range(1,dataFeeder.listMax+1):
		features, target = dataFeeder.getNextExample()
		model.forward(features)
		model.backward(target)
		if (all(np.round(model.activations) == target)):
			nCorrect += 1

	print('Pct Correct', nCorrect / dataFeeder.listMax)


for layer in model.layers:
	for node in layer.nodes:
		if hasattr(node,'delta'):
			print(node.w)

