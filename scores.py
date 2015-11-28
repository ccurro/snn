import numpy as np
import warnings
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def score(targets, predictions,softPredictions):
	t = np.sum(targets,0)
	p = np.sum(predictions,0)
	c = confusion_matrix(t,p)
	print(c)

	# Supress deprecation warnings
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		if np.max(t) > 1:
			for clss in range(0,np.max(t)+1):
				tb = (t == clss)
				pb = (p == clss)
				print(accuracy_score(tb,pb),
					precision_score(tb,pb),
					recall_score(tb,pb),
					f1_score(tb,pb))
		else:
			print(accuracy_score(t,p),
					precision_score(t,p),
					recall_score(t,p),
					f1_score(t,p))

		print(precision_score(t,p,average='micro'),
			recall_score(t,p,average='micro'),
			f1_score(t,p,average='micro'))
		print(precision_score(t,p,average='macro'),
			recall_score(t,p,average='macro'),
			f1_score(t,p,average='macro'))