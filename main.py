import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
st.write('''
	# 重回帰分析アプリ'''
	)

upload_file = st.file_uploader('csvファイル対応', type='csv')

if upload_file is not None:
	df = pd.read_csv(upload_file)
	df = df.dropna()
	df = df.round()
	df = pd.get_dummies(df)
	#st.write('何を予測しますか?')
	st.dataframe(data=df)

	option = st.selectbox(
    '何を予測しますか?',
	(df.columns))
	st.write('###', option, 'を予想します')

	X =  df.drop(option,axis=1) #columns is droped
	y = df[option]

	model = sm.OLS(y, sm.add_constant(X))
	result = model.fit()
	st.write(result.params)

	df_future = {'predict': result.predict(sm.add_constant(X))
	,'True': y }
	df_F = pd.DataFrame(df_future)
	st.line_chart(df_F)
	st.write(df_F)