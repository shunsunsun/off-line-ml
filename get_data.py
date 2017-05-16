from sqlalchemy import create_engine
import pandas as pd
def read_data():
	col_names = [ ,'label']
	engine= create_engine('oracle://uibs:uibs@58.251.157.179:1521/orcl',echo = True)
	#data = pd.read_sql_table(table_name, con=engine, schema=None, index_col=None, coerce_float=True, parse_dates=None, columns=None, chunksize=None)
	data = pd.read_sql_table(table_name = 'atm_label',con = engine,columns = ['a','b','label'])
	train = data[:int(len(data)*0.7)]
	test = data[int(len(data)*0.7):]
	train_y = train.label
	train_x = train.drop('label', axis=1)
	test_y = test.label
	test_x = test.drop('label', axis=1)
	return train_x, train_y, test_x, test_y
def read_data_zf():
	engine= create_engine('oracle://uibs:uibs@58.251.157.179:1521/orcl',echo = True)
	col_names = [ 'a','label']
	data_f = pd.read_sql_table(table_name = 'atm_f',con = engine,columns = col_names)
	data_z = pd.read_sql_table(table_name = 'atm_z',con = engine,columns = col_names)
