import numpy as np
import warnings
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def score(targets, predictions,fName):
	f = open(fName,'w')
	t = np.sum(targets,0)
	p = np.sum(predictions,0)
	c = confusion_matrix(t,p)
	print(t)
	print(c)

	# Supress deprecation warnings
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		if np.max(t) > 1:
			for clss in range(1,np.max(t)+1):
				tb = (t == clss)
				pb = (p == clss)
				cb = confusion_matrix(tb,pb,labels=[0, 1])
				print(accuracy_score(tb,pb),
					precision_score(tb,pb),
					recall_score(tb,pb),
					f1_score(tb,pb))
				f.write('{0:d} '.format(cb[0][0]))
				f.write('{0:d} '.format(cb[0][1]))				
				f.write('{0:d} '.format(cb[1][0]))				
				f.write('{0:d} '.format(cb[1][1]))
				f.write('{0:.3f} '.format(accuracy_score(tb,pb)))
				f.write('{0:.3f} '.format(precision_score(tb,pb)))
				f.write('{0:.3f} '.format(recall_score(tb,pb)))
				f.write('{0:.3f}\n'.format(f1_score(tb,pb)))
		else:
			print(accuracy_score(t,p),
					precision_score(t,p),
					recall_score(t,p),
					f1_score(t,p))
			f.write('{0:d} '.format(c[0][0]))
			f.write('{0:d} '.format(c[0][1]))				
			f.write('{0:d} '.format(c[1][0]))				
			f.write('{0:d} '.format(c[1][1]))
			f.write('{0:.3f} '.format(accuracy_score(t,p)))
			f.write('{0:.3f} '.format(precision_score(t,p)))
			f.write('{0:.3f} '.format(recall_score(t,p)))
			f.write('{0:.3f}\n'.format(f1_score(t,p)))

		print(precision_score(t,p,average='micro'),
			recall_score(t,p,average='micro'),
			f1_score(t,p,average='micro'))
		print(precision_score(t,p,average='macro'),
			recall_score(t,p,average='macro'),
			f1_score(t,p,average='macro'))
		f.write('{0:.3f} '.format(precision_score(t,p,average='micro')))
		f.write('{0:.3f} '.format(recall_score(t,p,average='micro')))
		f.write('{0:.3f}\n'.format(f1_score(t,p,average='micro')))
		f.write('{0:.3f} '.format(precision_score(t,p,average='macro')))
		f.write('{0:.3f} '.format(recall_score(t,p,average='macro')))
		f.write('{0:.3f}\n'.format(f1_score(t,p,average='macro')))
		f.close()