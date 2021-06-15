import streamlit as st
import pandas as pd 
import matplotlib as plt
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score, r2_score, accuracy_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

header = st.beta_container()
dataset = st.beta_container()
model_training = st.beta_container()
st.set_option('deprecation.showPyplotGlobalUse', False)

# First define some helper functions to load in data and plot model metrics
@st.cache(persist = True)
def load_data(filename):
	data = pd.read_csv(filename)
	label = LabelEncoder()
		
	for col in data.columns:
			data[col] = label.fit_transform(data[col])

	return data
@st.cache(persist = True, suppress_st_warning = True) # Warning about calling st.write inside function
def plot_metrics(metrics_list):
	if 'Confusion Matrix' in metrics_list:
		st.subheader("Confusion Matrix")
		plot_confusion_matrix(model, X_test, y_test, display_labels = class_names)
		st.pyplot()

	if 'ROC Curve' in metrics_list:
		st.subheader("ROC Curve")
		plot_roc_curve(model, X_test, y_test)
		st.pyplot()

	if 'Precision-Recall Curve' in metrics_list:
		st.subheader("PR Curve")
		plot_precision_recall_curve(model, X_test, y_test)
		st.pyplot()

with header:
	st.title("Shroom Classification Demo")
	st.markdown("This is a model training demo via Streamlit. This application allows you to train an algorithm (logistic regression, random forest or gradient boosting machine) to classifiy mushrooms into poisonous/edible, and evaluate its performance.")
	st.markdown("This application is for illustrative purposes only, and as you'll notice this dataset isn't much of a challenge for the more sophisticated algorithms.")
	st.markdown("The mushrooms dataset contains 8124 hypothetical mushroom samples of 23 species of gilled mushrooms in the Agaricus and Lepiota Family. The dataset describes a variety of mushroom characteristics (e.g., color, odor, shapes, habitat, etc.) and denotes whether that sample is edible or poisonous.")
	st.markdown("The dataset and its original description is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/mushroom).")
	st.markdown("**Disclaimer**: I don't even eat mushrooms and I've done about a whole of three minutes of research on the topic. It appears there's no actual golden rule to determine a shroom's edibility, so take from that what you will.")

with dataset: 
	st.header("Mushrooms Dataset")
	st.markdown("The information below provides a quick look at the dataset and its features (columns). The majority of mushrooms is edible (0), although the dataset is fairly well balanced.")
	

	df = load_data('data/mushrooms.csv')
	st.write(df.head(10))
	class_names = ['edible', 'poisonous']
	y = df['class']
	X = df.drop(columns = ['class'])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

	st.subheader('Mushroom class distribution')
	label_dist = pd.DataFrame(df['class'].value_counts())
	st.bar_chart(label_dist)

	st.markdown("The features included in this dataset are as follows (note that class is what we're trying to predict):")
	st.table(df.columns)
	st.markdown("You can use the model training section in the sidebar to train an algorithm to fit this dataset in order to predict _class_. When done, performance evaluation metrics will appear below!")

	#st.markdown("A sample of the dataset (class is what we're trying to predict):")
	#st.table(df.assign(hack='').set_index('hack').head(5))

with model_training:
	st.sidebar.header("Model training")
	st.sidebar.markdown("Select an algorithm and its hyperparameter-values below, then select performance evaluation metrics. When done simply press the Classify button.")

	sel_col, disp_col = st.beta_columns(2)

	st.sidebar.subheader("Choose your model")
	classifier = st.sidebar.selectbox("Classifier", ("Logistic Regression", "Random Forest", "Gradient Booster"))

# LR Model
	if classifier == "Logistic Regression":
		st.sidebar.subheader("Model Hyperparameters")
		C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step = 0.01, key = 'C_LR')
		max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key = 'max_iter')
		metrics = st.sidebar.multiselect("Select metrics to plot", ('Confusion Matrix', "ROC Curve", "PR curve"))

		if st.sidebar.button("Classify", key = 'classify'):
			# st.sidebar("Logistic Regression Results")
			model = LogisticRegression(C = C, max_iter = max_iter)
			model.fit(X_train, y_train)
			accuracy = model.score(X_test, y_test)
			y_pred = model.predict(X_test)
			st.write("Accuracy: ", accuracy.round(2))
			st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
			st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
			plot_metrics(metrics)      

	if classifier == "Random Forest":
		st.sidebar.subheader("Model Hyperparameters")
		criterion = st.sidebar.selectbox("The criterion value for splitting trees", ("gini", "entropy"), key = 'criterion') 
		n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step = 10, key = 'n_estimators')
		max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step = 1, key = 'max_depth')
		bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key = 'bootstrap')
		metrics = st.sidebar.multiselect("Select metrics to plot", ('Confusion Matrix', "ROC Curve", "PR curve"))

		if st.sidebar.button("Classify", key = 'classify'):
			# st.sidebar("Random Forest Results")
			model = RandomForestClassifier(criterion = criterion, max_depth = max_depth, n_estimators = n_estimators, bootstrap = bootstrap, n_jobs = -1)
			model.fit(X_train, y_train)
			accuracy = model.score(X_test, y_test)
			y_pred = model.predict(X_test)
			st.write("Accuracy: ", accuracy.round(2))
			st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
			st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
			plot_metrics(metrics)  

	if classifier == "Gradient Booster":
		st.sidebar.subheader("Model Hyperparameters") 
		n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step = 10, key = 'n_estimators')
		learning_rate = st.sidebar.selectbox("The learning rate (alpha) for gradient descent", (0.001, 0.01, 0.1, 0.3), key = 'learning_rate')
		max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step = 1, key = 'max_depth')
		subsample = st.sidebar.number_input("Subsampling proportion", 0.3, 1.0, step = 0.1, key = 'subsample')
		metrics = st.sidebar.multiselect("Select metrics to plot", ('Confusion Matrix', "ROC Curve", "PR curve"))

		if st.sidebar.button("Classify", key = 'classify'):
			# st.sidebar("Gradient Boosting Results")
			model = GradientBoostingClassifier(max_depth = max_depth, n_estimators = n_estimators, learning_rate = learning_rate, subsample = subsample)
			model.fit(X_train, y_train)
			accuracy = model.score(X_test, y_test)
			y_pred = model.predict(X_test)
			st.write("Accuracy: ", accuracy.round(2))
			st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
			st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
			plot_metrics(metrics)   

