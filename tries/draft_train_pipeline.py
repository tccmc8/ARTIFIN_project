Called train_pipeline.py

This is pseudo code to just get an idea.

Imports:
import pandas as pd
Import kagglehub
Import bumpy as np
Import matplotlib.pyplot as put
From sklearn.linear_model import LinearRegression 
Etc.

Load data:
Def load_data:
	df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "atharvasoundankar/futuristic-smart-city-citizen-activity-dataset", file_path,)

	return df

Prepare 
Def check_missing_values:
	missing_val = df.isnull().sum()
	print("missing values in each column:")
	print(missing_val)
	
	return missing_val

Def merge_like:
	""" In some instances bike and bicycle are classified as different modes of transport when they are the same. This function would be used to merge the groups."""

	cycling = concat("bike", "bicycle")

	return cycling, another other categories to merge

Def intergerise:
	""" turning categories like male, female and other into integers 1,2 and 3. As well as the public transport category"""

	clean_df = dfdf[["Citizen_ID","Age","Work_Hours","Shopping_Hours","Entertainment_Hours", "Home_Energy_Consumption_kWh", "Charging_Station_Usage", "Steps_Walked", "Carbon_Footprint_kgCO2"]],
	
	for g df.column("Gender"):
		if g == "Male":
			g == 0
			clean_df.concat(g)
		if g == "Female":
			g == 1
			clean_df.concat(g)
		elif g == "Other":
			g == 2
			clean_df.concat(g)
	
	for trans df.column("Mode_of_Transport"):
		if trans == "Walking":
			trans == 0
			clean_df.concat(trans)
		if trans == "Cycling":
			trans == 1
			clean_df.concat(trans)
		if trans == "Car":
			trans == 2
			clean_df.concat(trans)
		if trans == "EV":
			trans == 3
			clean_df.concat(trans)
		if trans == "Public Transport":
			trans == 4
			clean_df.concat(trans)

	return clean_df


Train test split / train your model

Def train_models(X_train, X_val, X_test, y_train, y_val, y_test):
	model directory
	use mlflows to track the experiments and store it in mlruns
	name/ set up the experiment
	start prediction and log the parameters
	get the accuracy of each predication and model
