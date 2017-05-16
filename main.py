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
