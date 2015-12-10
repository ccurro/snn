import numpy as np
import warnings
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def score(targets, predictions,fName):
	f = open(fName,'w')
	t = np.sum(targets,0)
	p = np.sum(predictions,0)
	c = confusion_matrix(t,p,labels=[0,1])
	cm = np.zeros(np.shape(c))

	recall_macro = 0
	prec_macro = 0
	acc_macro = 0

	# Supress deprecation warnings
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		if np.max(t) > 1:
			for clss in range(1,np.max(t)+1):
				tb = (t >= clss)
				pb = (p >= clss)
				cb = confusion_matrix(tb,pb,labels=[0, 1])
				cm = cm + cb
				recall_macro = recall_macro + recall_score(tb,pb)/np.max(t)
				prec_macro = prec_macro + precision_score(tb,pb)/np.max(t)
				acc_macro = acc_macro + accuracy_score(tb,pb)/np.max(t)
				f.write('{0:d} '.format(cb[1][1]))
				f.write('{0:d} '.format(cb[0][1]))				
				f.write('{0:d} '.format(cb[1][0]))				
				f.write('{0:d} '.format(cb[0][0]))
				f.write('{0:.3f} '.format(accuracy_score(tb,pb)))
				f.write('{0:.3f} '.format(precision_score(tb,pb)))
				f.write('{0:.3f} '.format(recall_score(tb,pb)))
				f.write('{0:.3f}\n'.format(f1_score(tb,pb)))

			A = cm[1][1]
			B = cm[0][1]
			C = cm[1][0]
			D = cm[0][0]

			acc_micro = (A + D)/(A + B + C + D)
			prec_micro = A/(A + B)
			recall_micro = A/(A + C)
			f1_micro = (2*prec_micro*recall_micro)/(prec_micro+recall_micro)

			f.write('{0:.3f} '.format(acc_micro))
			f.write('{0:.3f} '.format(prec_micro))
			f.write('{0:.3f} '.format(recall_micro))
			f.write('{0:.3f}\n'.format(f1_micro))

		else:
			f.write('{0:d} '.format(c[1][1]))
			f.write('{0:d} '.format(c[0][1]))				
			f.write('{0:d} '.format(c[1][0]))				
			f.write('{0:d} '.format(c[0][0]))
			recall_macro = recall_score(t,p)
			prec_macro = precision_score(t,p)
			acc_macro = accuracy_score(t,p)
			f.write('{0:.3f} '.format(acc_macro))
			f.write('{0:.3f} '.format(prec_macro))
			f.write('{0:.3f} '.format(recall_macro))
			f.write('{0:.3f}\n'.format(f1_score(t,p)))

			f.write('{0:.3f} '.format(accuracy_score(t,p)))
			f.write('{0:.3f} '.format(precision_score(t,p,average='micro')))
			f.write('{0:.3f} '.format(recall_score(t,p,average='micro')))
			f.write('{0:.3f}\n'.format(f1_score(t,p,average='micro')))

		f.write('{0:.3f} '.format(acc_macro))
		f.write('{0:.3f} '.format(prec_macro))
		f.write('{0:.3f} '.format(recall_macro))
		f.write('{0:.3f}\n'.format((2*prec_macro*recall_macro)/(prec_macro+recall_macro)))
		f.close()
