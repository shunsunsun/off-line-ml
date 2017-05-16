import get_data

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
