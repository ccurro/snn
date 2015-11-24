import numpy as np
import itertools
import input
import data 

class Node:
	def __init__(self,nInputs,feeder,isInput):
		self.feeder = feeder
		if isInput:
			self.w = np.zeros((nInputs))
		else:
			self.initWeights()
		self.activation = 0

	def initWeights(self):
		self.w = self.feeder.getNextNodesWeights()
		print(self.feeder.listPosition)
		# self.w = np.random.rand(np.size(self.w))*0.1

	def sigmoid(self):
		self.activation = 1/(1+np.exp(-self.activation))

	def forward(self,features):
		assert(np.size(features) < np.size(self.w))
		# bias at end
		# self.features = np.concatenate([features,-np.ones((1))])
		# bias at beginning
		self.features = np.concatenate([-np.ones((1)), features])	
		self.activation = np.inner(self.w,self.features)
		self.sigmoid()
		self.dsig = self.activation*(1 - self.activation)

	def update(self):
		self.grad = self.features*self.delta
		self.w = self.w + 0.1*self.grad

	def resetActivations(self):
		if hasattr(self,'activation'):
			del self.activation

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

	def resetActivations(self):
		if hasattr(self,'activations'):
			del self.activations
		for node in self.nodes:
			node.resetActivations()

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
			self.resetActivations()

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
			for i in range(0,len(layer.nodes)):
				s = 0
				for j in range(0,len(self.layers[l+1].nodes)):
					s = s + self.layers[l+1].nodes[j].w[i]*self.layers[l+1].nodes[j].delta

				layer.nodes[i].delta = layer.nodes[i].dsig*s

		# update nodes
		for layer in self.layers:
			for node in layer.nodes:
				if hasattr(node,'delta'):
					node.update()
		# quit()

	def resetActivations(self):
		del self.activations
		for layer in self.layers:
			layer.resetActivations()

inputFeeder = input.Input('sample.NNWDBC.init')
dataFeeder = data.DataFeeder('wdbc.mini_train')
model = Network(inputFeeder)

for epoch in range(0,1):
	nCorrect = 0
	for example in range(1,dataFeeder.listMax+1):
		features, target = dataFeeder.getNextExample()
		model.forward(features)
		# print(model.activations)
		model.backward([target])
		if (np.round(model.activations) == target):
			nCorrect += 1

	print('Pct Correct', nCorrect / dataFeeder.listMax)


for layer in model.layers:
	for node in layer.nodes:
		print(node.w)

