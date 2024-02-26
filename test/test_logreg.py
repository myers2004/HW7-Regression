"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
import numpy as np
# (you will probably need to import more things here)

def test_prediction():
	#Testing prediction fucntion agaisnt a known predictor output

	#Create an instance of the regressor class, loading data along the way
	X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
	)

	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.1, tol=0.01, max_iter=100, batch_size=10)

	#Make a prediction (1/(1 + exp(-X*W)) with known output
	X = np.array([[1,1,1], [-1,-1,-1], [0.25,0.5,0.125]])
	W = np.array([1,2,3])
	epsilon = 0.00000001
	know_predict = np.array([0.99752738, 0.00247262, 0.83548354])

	#Overwrite weights for testing
	log_model.W = W
	calc_predict = log_model.make_prediction(X)

	#Test that the predictions are the same
	assert len(calc_predict) == len(know_predict)
	for i in range(len(calc_predict)):
		assert np.isclose(calc_predict[i], know_predict[i], epsilon)

def test_loss_function():
	#Testing loss function against a know correct output 

	#Create an instance of the regressor class, loading data along the way
	X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
	)

	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.1, tol=0.01, max_iter=100, batch_size=10)

	epsilon = 0.00000001 #used for comapring two floats

	#true and predicted labels with known loss
	y_true = np.array([1,1,0])
	y_pred = np.array([0.2, 0.8, 0.1])
	known_loss = 0.64598066

	#Calculate loss and compare
	calc_loss = log_model.loss_function(y_true, y_pred)
	print(calc_loss)

	#Test theat the losses are the same
	assert np.isclose(known_loss, calc_loss,epsilon)

def test_gradient():
	#Testing gradient function against a known correct output 

	#Create an instance of the regressor class, loading data along the way
	X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
	)

	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.1, tol=0.01, max_iter=100, batch_size=10)

	#Use small arrays with known output to test
	X = np.array([[1,1,1], [-1,-1,-1], [0.25,0.5,0.125]])
	W = np.array([1,2,3])
	epsilon = 0.00000001 #used for comapring two floats

	#Overwrite weights for testing
	log_model.W = W

	y_true = np.array([1,1,0])

	known_gradient = np.array([0.40130855, 0.47093218, 0.36649673])

	calc_gradient = log_model.calculate_gradient(y_true, X)

	#Test that the calculated gradient is the same as the known
	assert len(known_gradient) == len(calc_gradient)
	for i in range(len(calc_gradient)):
		assert np.isclose(calc_gradient[i], known_gradient[i], epsilon)

def test_training():
	#Testing that weights are updated during training

	#Create an instance of the regressor class, loading data along the way
	X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
	)

	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)
	
	epsilon = 0.00000001

	#Testing the weights get updated by looking at 3 time steps:
	# W before training
	# The second to last W
	# W after training

	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.1, tol=0.01, max_iter=100, batch_size=10)
	W_init = log_model.W

	log_model.train_model(X_train, y_train, X_val, y_val)
	W_prev = log_model.prev_W
	W_last = log_model.W

	assert len(W_init) == len(W_prev) == len(W_last)

	#All elements of W may not change during a given iteration, so we have to check that at least one does
	one_changed = False
	if False in np.isclose(W_last, W_init, epsilon):
			one_changed = True
	assert one_changed

	#Do this for the other two comaprisons
	one_changed = False
	if False in np.isclose(W_last, W_prev, epsilon):
			one_changed = True
	assert one_changed

	one_changed = False
	if False in np.isclose(W_init, W_prev, epsilon):
			one_changed = True
	assert one_changed