import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import plotly.express as px
#Analyse data
import matplotlib as mpl
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import scipy.stats as st
from sklearn import ensemble, tree, linear_model
import missingno as msno



##Functions

def data_msno(dataframe_features):
	msno.matrix(dataframe_features.sample(200))
	plt.savefig("./results/msno_missing_values.png")
	plt.show()
	
def data_dendrogram(dataframe_features):
	msno.dendrogram(dataframe_features)
	plt.savefig("./results/msno_dendrogram_values.png")
	plt.show()

def correlation(dataframe_features):
	f , ax = plt.subplots(figsize = (14,12))
	plt.title('Correlation of Numeric Features',y=1,size=16)
	sns.heatmap(dataframe_features.corr(),square = True,  vmax=0.8)
	plt.savefig("./results/features_correlation.png")
	plt.show()


def k_analysis(features):
	from sklearn.cluster import KMeans
	import matplotlib.pyplot as plt

	# Assuming X is your dataset
	inertias = []
	K_range = range(1, 10)

	for K in K_range:
		kmeans = KMeans(n_clusters=K, random_state=0).fit(features)
		inertias.append(kmeans.inertia_)

	plt.figure(figsize=(16,8))
	plt.plot(K_range, inertias, 'bx-')
	plt.xlabel('k')
	plt.ylabel('Inertia')
	plt.title('The Elbow Method showing the optimal k')
	plt.savefig("./results/features_Elbow.png")
	plt.show()

	from sklearn.metrics import silhouette_score
	import matplotlib.pyplot as plt

	# Assuming X is your dataset
	silhouette_scores = []
	K_range = range(2, 10)  # Starts from 2 because silhouette_score needs at least 2 clusters

	for K in K_range:
		kmeans = KMeans(n_clusters=K, random_state=0).fit(features)
		preds = kmeans.predict(features)
		score = silhouette_score(features, preds)
		silhouette_scores.append(score)

	plt.figure(figsize=(16,8))
	plt.plot(K_range, silhouette_scores, 'bx-')
	plt.xlabel('k')
	plt.ylabel('Silhouette Score')
	plt.title('The Silhouette Method showing the optimal k')
	plt.savefig("./results/features_Silhouette.png")
	plt.show()

def PCA_ANALYSIS(dataframe_features,labels,features_names):
	#To reduce the dimensionality and make a better classifier
	pca=PCA(n_components=2)
	components = pca.fit_transform(dataframe_features)

	#exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

	#fig=px.area(
	#	x=range(1, exp_var_cumul.shape[0] + 1),
	#	y=exp_var_cumul,
	#		labels={"x": "# Components", "y": "Explained Variance"}
	#)
	
	##fig = px.scatter(components, x=0, y=1, color=labels)
	
	loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

	fig = px.scatter(components, x=0, y=1, color=labels)

	for i, feature in enumerate(features_names):
		fig.add_annotation(
		    ax=0, ay=0,
		    axref="x", ayref="y",
		    x=loadings[i, 0],
		    y=loadings[i, 1],
		    showarrow=True,
		    arrowsize=2,
		    arrowhead=2,
		    xanchor="right",
		    yanchor="top"
		)
		fig.add_annotation(
		    x=loadings[i, 0],
		    y=loadings[i, 1],
		    ax=0, ay=0,
		    xanchor="center",
		    yanchor="bottom",
		    text=feature,
		    yshift=5,
		)

	fig.show()
	
	
##Load dataset
data = load_breast_cancer()

#Filter features 
features=data["data"]
labels=data["target"]

features_names=['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error','concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']

#Visualize first samples
dataframe_features=pd.DataFrame(features, columns=features_names)

#Visualize missing values
#data_msno(dataframe_features)

#Dendrogram - Analyse the pairwise ont he features
#data_dendrogram(dataframe_features)

#Correlation Between the most dangerous
#correlation(dataframe_features)

#k_analysis(features)

##Visualize where the features belong to each component
PCA_ANALYSIS(dataframe_features,labels,features_names)

## WE GOT 2 COMPONENTS 
##On this figure it is possible to analyse that the 'worst area' and 'mean area' are the responsible to repesent the greatest influences on the components compositions, which can be used as the only two features to make a classifyer! Let's see it on practice.

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

selected_features = dataframe_features[['mean area', 'worst area']]


X_train, X_test, y_train, y_test = train_test_split(selected_features, labels, test_size=0.2, random_state=0)

clf = make_pipeline(StandardScaler(), SVC())
clf.fit(X_train, y_train)

# Calculate the score
score_value = clf.score(X_test, y_test) * 100
print(f"Score value: {score_value:.2f} %")

#Only using TWO of 30 VARIABLES is possible to get an accuracy value: 94.74 %


