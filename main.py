# look into time diff
# 10ms target

from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse, abort

# Imports needed for the script
import numpy as np
import pandas as pd
import re
from sklearn import tree
import random

import time
import math


# Create Decision Tree with max_depth = 3
decision_tree = tree.DecisionTreeClassifier(max_depth = 3)


app = Flask(__name__)
api = Api(app)

post_args = reqparse.RequestParser()
post_args.add_argument('PassengerId', type = int, help = 'PassengerId is required')
post_args.add_argument('Pclass', type = int, help = 'Pclass is required')
post_args.add_argument('Name', type = str, help = 'Name is required')
post_args.add_argument('Sex', type = str, help = 'Sex is required')
post_args.add_argument('Age', type = float, help = 'Age is required')
post_args.add_argument('SibSp', type = int, help = 'SibSp is required')
post_args.add_argument('Parch', type = int, help = 'Parch is required')
post_args.add_argument('Ticket', type = str, help = 'Ticket is required')
post_args.add_argument('Fare', type = float, help = 'Fare is required')
post_args.add_argument('Cabin', type = str, help = 'Cabin is required')
post_args.add_argument('Embarked', type = str, help = 'Embarked is required')



class Predict(Resource):
	def post(self):
		begin_block = time.time()
		args = post_args.parse_args()

		inputs_dict = {
		 'PassengerId': args['PassengerId'],
		 'Pclass': args['Pclass'],
		 'Name': args['Name'],
		 'Sex': args['Sex'],
		 'Age': args['Age'],
		 'SibSp': args['SibSp'],
		 'Parch': args['Parch'],
		 'Ticket': args['Ticket'],
		 'Fare': args['Fare'],
		 'Cabin': args['Cabin'],
		 'Embarked': args['Embarked'],
		} 

		inputs = inputs_dict.copy()

		# inputs_df = pd.DataFrame(inputs)
		# print(inputs['PassengerId'])

		# inputs_df = test_dataset.loc[test_dataset['PassengerId'] == inputs['PassengerId']]
		# inputs_dict = inputs_df.set_index('PassengerId').T.to_dict('dict')
		# inputs_dict = inputs_dict[inputs['PassengerId']]
		# print(inputs_dict)

		# return {
		# 	'PassengerId': inputs['PassengerId'],
		# 	}	

		# preprocess inputs
		inputs_dict['Has_Cabin'] = (lambda x: 0 if type(x) == float else 1)(inputs_dict['Cabin'])
		
		inputs_dict['FamilySize'] = inputs_dict['SibSp'] + inputs_dict['Parch'] + 1
		
		inputs_dict['IsAlone'] = 0
		inputs_dict['IsAlone'] = 1 if inputs_dict['FamilySize'] == 1 else 0

		# inputs_dict['Age'] = int(inputs_dict['Age'])
		if inputs_dict['Sex'] == 'female':
		    inputs_dict['Sex'] = 0 
		elif inputs_dict['Sex'] == 'male':
		    inputs_dict['Sex'] = 1

		# Define function to extract titles from passenger names
		def get_title(name):
		    title_search = re.search(' ([A-Za-z]+)\.', name)
		    # If the title exists, extract and return it.
		    if title_search:
		        return title_search.group(1)
		    return ""

		inputs_dict['Title'] = get_title(inputs_dict['Name'])

		# Group all non-common titles into one single grouping "Rare"
		inputs_dict['Title'] = 'Rare' if inputs_dict['Title'] in ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'] else inputs_dict['Title']
		inputs_dict['Title'] = inputs_dict['Title'].replace('Mlle', 'Miss')
		inputs_dict['Title'] = inputs_dict['Title'].replace('Ms', 'Miss')
		inputs_dict['Title'] = inputs_dict['Title'].replace('Mme', 'Mrs')

		# Mapping titles
		title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
		inputs_dict['Title'] = title_mapping[inputs_dict['Title']]

		# Mapping Embarked
		if 'Embarked' not in inputs_dict['Embarked'] or not inputs_dict['Embarked'] or inputs_dict['Embarked'] == 'nan':
			inputs_dict['Embarked'] = 'S'

		embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
		inputs_dict['Embarked'] = embarked_mapping[inputs_dict['Embarked']]
		    
		# Mapping Fare
		if 'Fare' not in inputs_dict or inputs_dict['Fare'] == float('nan'):
			inputs_dict['Fare'] = median_fare

		if inputs_dict['Fare'] >= 31: 
		    inputs_dict['Fare'] = 3
		elif inputs_dict['Fare'] >= 14.454: 
		    inputs_dict['Fare'] = 2
		elif inputs_dict['Fare'] >= 7.91: 
		    inputs_dict['Fare'] = 1
		else:
		    inputs_dict['Fare'] = 0
		    
		# Mapping Age
		if 'Age' not in inputs_dict or math.isnan(inputs_dict['Age']):
			inputs_dict['Age'] = age_null_random_list[random.randint(0, len(age_null_random_list)-1)]

		# print(inputs_dict['Age'] == float('nan'))
		inputs_dict['Age'] = int(inputs_dict['Age'])

		if inputs_dict['Age'] >= 64: 
		    inputs_dict['Age'] = 4
		elif inputs_dict['Age'] >= 48: 
		    inputs_dict['Age'] = 3
		elif inputs_dict['Age'] >= 32: 
		    inputs_dict['Age'] = 2
		elif inputs_dict['Age'] >= 16: 
		    inputs_dict['Age'] = 1
		else:
		    inputs_dict['Age'] = 0

		# Feature selection: remove variables no longer containing relevant information
		del inputs_dict['Name']
		del inputs_dict['Ticket']
		del inputs_dict['Cabin']
		del inputs_dict['SibSp']
		del inputs_dict['PassengerId']

		# convert dict into dataframe
		inputs_df_prcsd = pd.DataFrame(inputs_dict, index=[0])

		# print(inputs_df_prcsd)

		# return {
		# 	'PassengerId': inputs['PassengerId'],
		# 	}

		# predict
		begin = time.time()
		survived = decision_tree.predict(inputs_df_prcsd)
		end = time.time()
		
		return {
			'PassengerId': inputs['PassengerId'],
			'Survived': str(survived.astype(int)[0]),
			'Time taken for predict': str((end-begin)*1000)[:4] + ' ms',
			'Time taken for block': str((end-begin_block)*1000)[:4] + ' ms'
		}

		


api.add_resource(Predict, '/predict')



if __name__ == '__main__':
	
	# Loading the data
	dataset = pd.read_csv('titanic_dataset/train.csv')
	test_dataset = pd.read_csv('titanic_dataset/test.csv')

	# preprocessing
	dataset['Has_Cabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
	dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
	dataset['IsAlone'] = 0
	dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
	dataset['Embarked'] = dataset['Embarked'].fillna('S')

	median_fare = dataset['Fare'].median()
	dataset['Fare'] = dataset['Fare'].fillna(median_fare)

	age_avg = dataset['Age'].mean()
	age_std = dataset['Age'].std()
	age_null_count = dataset['Age'].isnull().sum()
	age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

	# Next line has been improved to avoid warning
	dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
	dataset['Age'] = dataset['Age'].astype(int)
	dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

	# Define function to extract titles from passenger names
	def get_title(name):
	    title_search = re.search(' ([A-Za-z]+)\.', name)
	    # If the title exists, extract and return it.
	    if title_search:
	        return title_search.group(1)
	    return ""

	dataset['Title'] = dataset['Name'].apply(get_title)

	# Group all non-common titles into one single grouping "Rare"
	dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

	dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
	dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
	dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

	# Mapping titles
	title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
	dataset['Title'] = dataset['Title'].map(title_mapping)
	dataset['Title'] = dataset['Title'].fillna(0)

	# Mapping Embarked
	dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
	    
	# Mapping Fare
	dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
	dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
	dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
	dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
	dataset['Fare'] = dataset['Fare'].astype(int)
	    
	# Mapping Age
	dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
	dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
	dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
	dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
	dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;

	# Feature selection: remove variables no longer containing relevant information
	drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
	dataset = dataset.drop(drop_elements, axis = 1)

	# Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models
	y_train = dataset['Survived']
	x_train = dataset.drop(['Survived'], axis=1).values 

	decision_tree.fit(x_train, y_train)

	app.run(debug = True)



# fill empty data

# # Remove all NULLS in the Embarked column
# for dataset in full_data:
#     dataset['Embarked'] = dataset['Embarked'].fillna('S')


# # Remove all NULLS in the Fare column
# for dataset in full_data:
#     dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())


# # Remove all NULLS in the Age column
# for dataset in full_data:
#     age_avg = dataset['Age'].mean()
#     age_std = dataset['Age'].std()
#     age_null_count = dataset['Age'].isnull().sum()
#     age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
#     # Next line has been improved to avoid warning
#     dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
#     dataset['Age'] = dataset['Age'].astype(int)



