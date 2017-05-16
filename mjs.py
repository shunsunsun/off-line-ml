# coding=utf-8
import time,sys
from sklearn import metrics  
import pickle as pickle  
import pprint
import pandas as pd
import numpy   as np
import matplotlib.pyplot as plt  
import  xdrlib
import xlrd,xlwt,re
import time,datetime
from sqlalchemy import create_engine
reload(sys)  
sys.setdefaultencoding('utf8') 
#import xgboost as xgb
#import operator
# Multinomial Naive Bayes Classifier  
def naive_bayes_classifier(train_x, train_y):  
    from sklearn.naive_bayes import MultinomialNB  
    model = MultinomialNB(alpha=0.01)  
    model.fit(train_x, train_y)  
    return model  
  
  
# KNN Classifier  
def knn_classifier(train_x, train_y):  
    from sklearn.neighbors import KNeighborsClassifier  
    model = KNeighborsClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
  
# Logistic Regression Classifier  
def logistic_regression_classifier(train_x, train_y):  
    from sklearn.linear_model import LogisticRegression  
    model = LogisticRegression(penalty='l2')  
    model.fit(train_x, train_y)  
    return model  
  
  
# Random Forest Classifier  
def random_forest_classifier(train_x, train_y):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=8)  
    model.fit(train_x, train_y)  
    return model  
  
  
# Decision Tree Classifier  
def decision_tree_classifier(train_x, train_y):  
    from sklearn import tree  
    model = tree.DecisionTreeClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
  
# GBDT(Gradient Boosting Decision Tree) Classifier  
def gradient_boosting_classifier(train_x, train_y):  
		from sklearn.ensemble import GradientBoostingClassifier
		from sklearn.grid_search import GridSearchCV 
		from sklearn import cross_validation, metrics
		import matplotlib.pylab as plt
	
		from matplotlib.pylab import rcParams
		
		model = GradientBoostingClassifier()  
		
		#现在还有两个问题，一个是样本集的交叉问题，如何设置训练样本和测试样本，另外一个问题就是分类器参数调优的问题，这需要如下的确定参数
		param_grid1={'loss':['deviance','exponential'], 'learning_rate':[0.01,0.05,0.1,0.2], 'n_estimators':[50,100,150,200], 'subsample':[0.7,0.8,0.9,1.0],
		'min_samples_split':[200,500,800,1000,1200,1500,1800,2000], 'min_samples_leaf':[30,50,80,100,120,150,200], 
		'max_depth':[3,4,5,6,7,8,9],'max_features':[2,3,4,5,6,7,8,9,10,'sqrt','log2',None]} 
		
		param_grid={'learning_rate':[0.01,0.05,0.1], 'n_estimators':[50,100,200], 'subsample':[0.8,0.9],
		'min_samples_split':[200,500], 'min_samples_leaf':[30,50],'max_depth':[7,9]} 
		
		grid_search = GridSearchCV(model, param_grid, n_jobs = -1,verbose=1)  
		grid_search.fit(train_x, train_y)  
		best_parameters = grid_search.best_estimator_.get_params()  
		for para, val in list(best_parameters.items()):  
       				 print(para, val)  
		
		model = GradientBoostingClassifier(learning_rate=best_parameters['learning_rate'],n_estimators=best_parameters['n_estimators'], 
		subsample=best_parameters['subsample'],	min_samples_split=best_parameters['min_samples_split'],min_samples_leaf=best_parameters['min_samples_leaf'],
		max_depth=best_parameters['max_depth'],verbose=1,warm_start='False')  

		model.fit(train_x, train_y)
		print model.feature_importances_   #特征重要性	
		#print model.get_params(True)
		print model.train_score_,model.oob_improvement_
		return model  
		
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
def gradient_boosting_classifier1(train_x, train_y):  
	gdbt = GradientBoostingClassifier(random_state=10)
	return modelfit(gdbt, train_x, train_y)

def modelfit(alg, train_x, train_y, performCV=True, printFeatureImportance=True, cv_folds=5):# just test this alg
    #Fit the algorithm on the data
    alg.fit(train_x, train_y)

    #Predict training set:
    dtrain_predictions = alg.predict(train_x)
    dtrain_predprob = alg.predict_proba(train_x)[:,1]
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data, train_target, test_size=0.4, random_state=0)

    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, train_x, train_y, cv=cv_folds, scoring='roc_auc',n_jobs=-1)

    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(train_y.values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(train_y, dtrain_predprob)
    

    if performCV:
		print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))

    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_).sort_values(ascending=False)
	feat_imp = feat_imp[feat_imp>np.mean(feat_imp)]
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
	plt.show()
    #print 'alg.feature_importances_:'
    #print alg.feature_importances_
    return alg
#model = GradientBoostingClassifier(learning_rate=0.05,n_estimators=200,subsample=0.8)    
# SVM Classifier  
def svm_classifier(train_x, train_y):  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    model.fit(train_x, train_y)  
    return model  
  
# SVM Classifier using cross validation  
def svm_cross_validation(train_x, train_y):  
    from sklearn.grid_search import GridSearchCV  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}  
    grid_search = GridSearchCV(model, param_grid, n_jobs = -1, verbose=1)  
    grid_search.fit(train_x, train_y)  
    best_parameters = grid_search.best_estimator_.get_params()  
    for para, val in list(best_parameters.items()):  
        print(para, val)  
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)  
    model.fit(train_x, train_y)  
    return model  
# XGBOOST 特征重要性筛选
def choosefeatuer_base_xgboost(train_x,train_y):
	xgb_params = {'booster':'gbtree','objective':'binary:logistic','eta': 0.01,'max_depth':5,'silent':0,'colsample_bytree':0.7}
	num_rounds=1000
	dtrain=xgb.DMatrix(train_x,label=train_y)
	gdbt=xgb.train(xgb_params,dtrain,num_rounds)
	importance=gdbt.get_fscore()
	importance=sorted(importance.items())
	key=operator.itemgetter(1)
	print importance
#http://pandas.pydata.org/pandas-docs/stable/index.html    	
#file:///C:/Users/maojiashun/Desktop/test/Python%E4%B8%ADDataFrame%E5%87%BD%E6%95%B0%E6%93%8D%E4%BD%9C%E6%95%B0%E6%8D%AE%E5%BA%93%20-%20CJK%E7%9A%84%E5%8D%9A%E5%AE%A2%20-%20%E5%8D%9A%E5%AE%A2%E9%A2%91%E9%81%93%20-%20CSDN.NET.htm
def read_data():  
		#col_names = [ ,'label']
		#engine= create_engine('oracle://uibs:uibs@58.251.157.179:1521/orcl',echo = True)
		#data = pd.read_sql_table(table_name, con=engine, schema=None, index_col=None, coerce_float=True, parse_dates=None, columns=None, chunksize=None)
		#data = pd.read_sql_table(table_name = 'atm_label',con = engine,columns = ['a','b','label'])
		from sklearn.datasets import load_svmlight_file
		train_data = load_svmlight_file('demo/data/agaricus.txt.train')
		X = train_data[0].toarray()
		y = train_data[1]
		data=pd.DataFrame(X)
		data['label']=y
		
		train = data[:int(len(data)*0.7)]
		test = data[int(len(data)*0.7):]
		train_y = train.label
		train_x = train.drop('label', axis=1)
		test_y = test.label
		test_x = test.drop('label', axis=1)
		#print train_x['vs'].apply(lambda x: time.mktime(time.strptime(str(x),"%Y/%m/%d")))
		#del test_x['vs']
		return train_x, train_y, test_x, test_y
      
if __name__ == '__main__':    
		thresh = 0.5  
		model_save_file = "a.txt"  
		model_save = {}  
		test_classifiers = ['GBDT']
		classifiers = {'GBDT':gradient_boosting_classifier}
		
		test_classifiers1 = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM','SVMCV', 'GBDT']  
		classifiers1 = {'NB':naive_bayes_classifier,
						'KNN':knn_classifier,  
					'LR':logistic_regression_classifier,  
					'RF':random_forest_classifier,  
					'DT':decision_tree_classifier,  
					'SVM':svm_classifier,  
					'SVMCV':svm_cross_validation,  
					'GBDT':gradient_boosting_classifier
					}
		print('reading training and testing data...')
		train_x, train_y, test_x, test_y = read_data()
		
		for classifier in test_classifiers:  
				print('******************* %s ********************' % classifier)  
				start_time = time.time()  
				model = classifiers[classifier](train_x, train_y)  
				print('training took %fs!' % (time.time() - start_time))  
				predict = model.predict(test_x) 
				predict_proba = model.predict_proba(test_x)[:,1] 	
				if model_save_file != None:  
						model_save[classifier] = model  
				#precision = metrics.precision_score(test_y, predict)  
				#recall = metrics.recall_score(test_y, predict)  
				#print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))  
				#accuracy = metrics.accuracy_score(test_y, predict)  
				#print('accuracy: %.2f%%' % (100 * accuracy))   
				#roc   auc
				
				#precision_p, recall_p, thresholds = metrics.precision_recall_curve(test_y, predict_proba,pos_label=1) 
				fpr, tpr, thresholds_p = metrics.roc_curve(test_y, predict_proba,pos_label=1,drop_intermediate=False)
				print fpr, tpr, thresholds_p
				#print precision_p, recall_p, thresholds,predict_proba
				plt.plot(fpr, tpr,color=(1, 0, 1),lw=1)
				plt.xlim([0.0, 1.0])
				plt.ylim([0.0, 1.0])
				plt.title('%s ROC curve for diabetes classifier' % classifier)
				plt.xlabel('False Positive Rate (1 - Specificity)')
				plt.ylabel('True Positive Rate (Sensitivity)')
				plt.grid(True)
				plt.show()
				print ('roc_auc_score:%.2f%%' % metrics.roc_auc_score(test_y, predict))

				classify_report = metrics.classification_report(test_y, predict)
				confusion_matrix = metrics.confusion_matrix(test_y, predict)

				overall_accuracy = metrics.accuracy_score(test_y, predict)

				precision_for_each_class = metrics.precision_score(test_y, predict, average=None)
				average_precision = np.mean(precision_for_each_class)

				print('classify_report :', classify_report)
				print('confusion_matrix :', confusion_matrix)
				print('precision_for_each_class :', precision_for_each_class)
				print('average_precision: {0:f}'.format(average_precision))
				print('overall_accuracy: {0:f}'.format(overall_accuracy))


		if model_save_file != None:  
			pickle.dump(model_save, open(model_save_file, 'wb'),0)  
			#print model_save
			#dd = pickle.load(open(model_save_file, 'rb'))
			#pprint.pprint(dd)

