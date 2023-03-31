import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import re
import sys

from sklearn import svm, datasets
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import RocCurveDisplay  
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score,cross_val_predict, LeaveOneOut

from sklearn.metrics import confusion_matrix  
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, OrdinalEncoder    
from sklearn.impute import KNNImputer  


def scale(X_train, X_validate, scale_features = True):
	if scale_features:
		scaler = StandardScaler()
		scaler = scaler.fit(X_train) 

		X_validate = pd.DataFrame(scaler.transform(X_validate), columns = X_validate.columns)  

	return X_validate     	

def reduce_features(fitted_model, 
					X,
					y,
					eps=0, 
					permutation_repeats = 20): 
	
	df_imps = pd.DataFrame()
	df_imps['feature'] = X.columns

	sel_features_all = {}
	for i in range(permutation_repeats):
		imps = permutation_importance(fitted_model, X, y,)
		df_imps[f'importance_{i}'] = imps.importances_mean
		
		sel_features = df_imps.loc[df_imps[f'importance_{i}'] > eps, 'feature'].values

		for f in sel_features:
			if f in sel_features_all:
				sel_features_all[f] += 1
			else:
				sel_features_all[f] = 1   

	sel_features = []
	for f,n in sel_features_all.items():
		if n >= permutation_repeats//2:
			sel_features.append(f)
	
	return sel_features, df_imps    

def classify_leave_one_out_cv(clf, X, y, model="", save_to="", select_features=True, scale_features = True, title = "", logfile="log", saveIndivdualPred = True, **kwargs):
	# #####################################    
	# Classification  

	# Run classifier with leave-one-out cross-validation   
	try:
		kf = LeaveOneOut()

		df_results = pd.DataFrame()  	
		
		all_y = []
		all_probs =  []
		all_predicts=[]

		if select_features:
			selected_features_ind = {}
			importances_df = pd.DataFrame()  

		for i, (train, test) in enumerate(kf.split(X, y)):
		
			X_train = X.iloc[train].copy()
			y_train = y.iloc[train].copy()

			with open(f'{logfile}.out', 'a') as f:
				f.write(f"Split: {i+1} / {len(X)}; Model: {model}\n")
			

			X_test = X.iloc[test] # no need to copy, only a signle element
			y_test = y.iloc[test].iloc[0] # no need to copy, only a signle element
			
			if scale_features:
				#https://stackoverflow.com/questions/38058774/scikit-learn-how-to-scale-back-the-y-predicted-result
				scalerX = StandardScaler().fit(X_train)
				X_train[:] = scalerX.transform(X_train)
				X_test[:] = scalerX.transform(X_test)


			clf.fit(X_train, y_train) #hyperparameter tuning
			
			if select_features: # permutation feature selection
				sel_features, imps = reduce_features(clf, 
													X_train,
													y_train,
													**kwargs) 
				
				importances_df = pd.concat([importances_df, imps])         

				if sel_features:  
					for sf in sel_features: 
						# individual features based analysis
						if sf in selected_features_ind:
							selected_features_ind[sf] += 1
						else:
							selected_features_ind[sf] = 1 

			y_pred = clf.predict(X_test)  
			y_predProba = clf.predict_proba(X_test)[:,1]

			all_y.append(y_test)
			all_probs.append(y_predProba)
			all_predicts.append(y_pred[0])   
		

		all_y = np.array(all_y)
		all_probs = np.array(all_probs)
		all_predicts = np.array(all_predicts)

		tn, fp, fn, tp = confusion_matrix(all_y, all_predicts).ravel()
		prec = (tp/(tp+fp))
		rec = (tp/(tp+fn))
		f1 = (tp/(tp+0.5*(fp+fn)))
		acc = ((tp+tn)/(tp+tn+fp+fn))

		# roc and auc
		fpr, tpr, _ = roc_curve(all_y,all_probs)

		auc_val = auc(fpr, tpr)
		# plt.plot(fpr, tpr, lw=2, alpha=0.5, label='LOOCV ROC (AUC = %0.2f)' % (auc_val))
		# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance level', alpha=.8)
		# plt.xlim([-0.05, 1.05])
		# plt.ylim([-0.05, 1.05])
		# plt.xlabel('False Positive Rate')
		# plt.ylabel('True Positive Rate')
		# plt.title(model)
		# plt.legend(loc="lower right")
		# plt.show()

		results = {"precision": prec,
			"recall": rec,
			"f1": f1,
			"accuracy": acc,
			"model": model,
			"auc": auc_val}  

		df_results = df_results.append(results, ignore_index=True)  

		if select_features: 
			# individual features based analysis
			final_features_ind = []
			for sf, n in selected_features_ind.items():
				if n >= len(X)//2:
					final_features_ind.append(sf)
			final_features_ind.sort() 

		final_results = {}
		final_results['df_results'] = df_results 


		if select_features:
			final_results['selected_features_ind'] = final_features_ind
			importances_df = importances_df.assign(model=model)   
			final_results['importances_df'] = importances_df        
		if saveIndivdualPred:
			df_indPred = pd.DataFrame(np.column_stack((all_y, all_probs, all_predicts)), columns=["y_true", "y_predProb", "y_pred"])
			final_results["df_indPred"] = df_indPred

	except Exception as e:
		with open(f'{logfile}.out', 'a') as f:
			f.write(f"\n\n{str(e)}\n")		
		return 

	return final_results


