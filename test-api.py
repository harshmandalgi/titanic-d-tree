import pandas as pd
import requests
import time


if __name__ == '__main__':
	test_dataset = pd.read_csv('titanic_dataset/test.csv')
	test_dict = test_dataset.T.to_dict('dict')

	answer_dataset = pd.read_csv('titanic_dataset/gender_submission.csv')
	# answer_dataset = pd.read_csv('submission.csv')

	API_ENDPOINT = 'http://127.0.0.1:5000/predict'

	# print(test_dict)
	wrong_preds = 0
	latency = 0
	runs = 0

	for i, row in test_dict.items():
		# print(row['PassengerId'])
		# row_dict = row.T.to_dict('dict')
		# print(row)

		runs += 1

		start = time.time()
		r = requests.post(url = API_ENDPOINT, data = row)
		end = time.time()

		latency += end - start
		response = r.json()

		survived = int(response['Survived'])
		if survived != answer_dataset.loc[answer_dataset['PassengerId'] == row['PassengerId']]['Survived'].values[0]:
			wrong_preds += 1

		
	print('Wrong predictions: ' + str(wrong_preds))
	print('Accuracy: ' + str(((answer_dataset.shape[0]-wrong_preds)/answer_dataset.shape[0])*100))
	print('Mean Latency: ' + str(latency*1000/answer_dataset.shape[0]) + ' ms')
	print(runs)