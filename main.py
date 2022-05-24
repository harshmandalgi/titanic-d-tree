# look into time diff
# 10ms target

from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse, abort

# Imports needed for the script
import numpy as np
import pandas as pd
import re
from sklearn import tree

import time


# Create Decision Tree with max_depth = 3
decision_tree = tree.DecisionTreeClassifier(max_depth = 3)


app = Flask(__name__)
api = Api(app)

post_args = reqparse.RequestParser()
post_args.add_argument('PassengerId', type = int, help = 'PassengerId is required', required = True)
# post_args.add_argument('Pclass', type = str, help = 'Pclass is required', required = True)
# post_args.add_argument('Name', type = str, help = 'Name is required', required = True)
# post_args.add_argument('Sex', type = str, help = 'Sex is required', required = True)
# post_args.add_argument('Age', type = str, help = 'Age is required', required = True)
# post_args.add_argument('SibSp', type = str, help = 'SibSp is required', required = True)
# post_args.add_argument('Parch', type = str, help = 'Parch is required', required = True)
# post_args.add_argument('Ticket', type = str, help = 'Ticket is required', required = True)
# post_args.add_argument('Fare', type = str, help = 'Fare is required', required = True)
# post_args.add_argument('Cabin', type = str, help = 'Cabin is required', required = True)
# post_args.add_argument('Embarked', type = str, help = 'Embarked is required', required = True)



class Predict(Resource):
	def post(self):
		begin_block = time.time()
		args = post_args.parse_args()

		inputs = {
		 'PassengerId': args['PassengerId'],
		#  'Pclass': args['Pclass'],
		#  'Name': args['Name'],
		#  'Sex': args['Sex'],
		#  'Age': args['Age'],
		#  'SibSp': args['SibSp'],
		#  'Parch': args['Parch'],
		#  'Ticket': args['Ticket'],
		#  'Fare': args['Fare'],
		#  'Cabin': args['Cabin'],
		#  'Embarked': args['Embarked'],
		}

		# inputs_df = pd.DataFrame(inputs)
		# print(inputs['PassengerId'])

		inputs_df = test_dataset.loc[test_dataset['PassengerId'] == inputs['PassengerId']]
		# print(inputs_df)

		# preprocess inputs
		inputs_df['Has_Cabin'] = inputs_df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
		
		inputs_df['FamilySize'] = inputs_df['SibSp'] + inputs_df['Parch'] + 1
		
		inputs_df['IsAlone'] = 0
		inputs_df.loc[inputs_df['FamilySize'] == 1, 'IsAlone'] = 1
		inputs_df['Age'] = inputs_df['Age'].astype(int)
		inputs_df['Sex'] = inputs_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

		# Define function to extract titles from passenger names
		def get_title(name):
		    title_search = re.search(' ([A-Za-z]+)\.', name)
		    # If the title exists, extract and return it.
		    if title_search:
		        return title_search.group(1)
		    return ""

		inputs_df['Title'] = inputs_df['Name'].apply(get_title)

		# Group all non-common titles into one single grouping "Rare"
		inputs_df['Title'] = inputs_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

		inputs_df['Title'] = inputs_df['Title'].replace('Mlle', 'Miss')
		inputs_df['Title'] = inputs_df['Title'].replace('Ms', 'Miss')
		inputs_df['Title'] = inputs_df['Title'].replace('Mme', 'Mrs')

		# Mapping titles
		title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
		inputs_df['Title'] = inputs_df['Title'].map(title_mapping)

		# Mapping Embarked
		inputs_df['Embarked'] = inputs_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
		    
		# Mapping Fare
		inputs_df.loc[ inputs_df['Fare'] <= 7.91, 'Fare'] 						        = 0
		inputs_df.loc[(inputs_df['Fare'] > 7.91) & (inputs_df['Fare'] <= 14.454), 'Fare'] = 1
		inputs_df.loc[(inputs_df['Fare'] > 14.454) & (inputs_df['Fare'] <= 31), 'Fare']   = 2
		inputs_df.loc[ inputs_df['Fare'] > 31, 'Fare'] 							        = 3
		inputs_df['Fare'] = inputs_df['Fare'].astype(int)
		    
		# Mapping Age
		inputs_df.loc[ inputs_df['Age'] <= 16, 'Age'] 					       = 0
		inputs_df.loc[(inputs_df['Age'] > 16) & (inputs_df['Age'] <= 32), 'Age'] = 1
		inputs_df.loc[(inputs_df['Age'] > 32) & (inputs_df['Age'] <= 48), 'Age'] = 2
		inputs_df.loc[(inputs_df['Age'] > 48) & (inputs_df['Age'] <= 64), 'Age'] = 3
		inputs_df.loc[ inputs_df['Age'] > 64, 'Age'] ;

		# Feature selection: remove variables no longer containing relevant information
		drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
		inputs_df = inputs_df.drop(drop_elements, axis = 1)

		# predict
		begin = time.time()
		survived = decision_tree.predict(inputs_df)
		end = time.time()
		print(survived)
		
		return {
			'PassengerId': inputs['PassengerId'],
			'Survived': str(survived.astype(int)[0]),
			'Time taken for predict': str(end-begin),
			'Time taken for block': str(end-begin_block)
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
	dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
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
	dataset.loc[ dataset['Age'] > 64, 'Age'] ;

	# Feature selection: remove variables no longer containing relevant information
	drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
	dataset = dataset.drop(drop_elements, axis = 1)

	# Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models
	y_train = dataset['Survived']
	x_train = dataset.drop(['Survived'], axis=1).values 

	decision_tree.fit(x_train, y_train)

	app.run(debug = True)







