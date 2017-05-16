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
