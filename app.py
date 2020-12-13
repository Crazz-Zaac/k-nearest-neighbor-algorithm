import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from settings import DATASET_DIR as dataset
from sklearn import metrics

def main():
	df = pd.read_csv('dataset/teleCust1000t.csv')
	num = 6
	st.title("K-Nearest Neighbor Implementation")

	st.sidebar.title("Evaluating different parameters")
	st.sidebar.subheader("View dataset")
	num = st.sidebar.slider("Number of data", 5, 30, 5)

	val = df['custcat'].value_counts().to_frame()
	val.rename(index={1:'Basic-service (1)', 2:'E-Service customers (2)', 3: 'Plus Service (3)', 4:'Total Service (4)'}, inplace=True)

	if st.sidebar.button("View"):
		st.subheader("Viewing data")
		data = df.head(num)
		st.write(data)
		st.write(val)
	else:
		df.head()

	st.subheader("Visualizing data")
	plt.figure(figsize=(8,3))
	plt.hist(df['custcat'], bins=20, rwidth=0.9)
	plt.grid(axis='y', alpha=0.75)
	plt.xlabel('Class')
	plt.ylabel('Counts')
	plt.title('Custcat')
	st.pyplot()
	st.write(val)

	#normalizing data (except 'custcat' column) 
	st.subheader("Normalized data:")
	X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values
	st.write(X[0:5])
	#taking only custcat column
	st.subheader("Labels (Custcat column)")
	y = df['custcat'].values
	st.write(y[0:5])

	X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

	#Train Test split
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

	st.subheader("Number of Train and Test data")
	# if st.sidebar.button("Check number of datas"):
	st.write("Train set: ", X_train.shape, y_train.shape)
	st.write("Test set: ", X_test.shape, y_test.shape)

	#classification
	from sklearn.neighbors import KNeighborsClassifier
	st.sidebar.subheader("Set different values for 'k'")
	k = st.sidebar.slider("Value of k", 1, 10)
	if st.sidebar.button('Train Model and Predict'):
		neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
		st.subheader("Predicted value (yhat)")
		yhat = neigh.predict(X_test)
		st.write(yhat[0:10])


		#accuracy evaluation
		st.write("Accuracy Evaluation")
		st.write('Train set accuracy: ', metrics.accuracy_score(y_train, neigh.predict(X_train)))
		st.write("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

	st.sidebar.subheader("Check accuracy for all 'K' at once")
	if st.sidebar.button("Check accuracy"):
		Ks = 10
		mean_acc = np.zeros((Ks-1))
		std_acc = np.zeros((Ks-1))

		for n in range(1,Ks):
		    
		    #Train Model and Predict  
		    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
		    yhat=neigh.predict(X_test)
		    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

		    
		    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

		st.subheader("Accuracy for different values of k")
		st.write(mean_acc)
		st.write("The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)

	st.sidebar.write("\n")




if __name__=='__main__':
	main()


