import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

def predict(x_predict, y_test):
    filename = "TrainedModel.p"
    model = pickle.load(open(filename, 'rb'))

    scaler = StandardScaler()

    scaler.fit(x_predict)
    x_predict = scaler.transform(x_predict)

    y_predict = model.predict(x_predict)
    print("y_predict:", y_predict)
    print("y_test:", y_test)
    return y_predict

x_predict =[-0.00153712,0.00202408,0.00470084,14.9698,15.2133,15.9578,17.3327,19.6191,23.4201,30.1854,44.3415,88.5689,200,84.3097,43.386,29.8183,23.2531,19.5439,17.3088,15.9678,15.2498,15.0305]
x_predict = np.array(x_predict)
x_predict = x_predict.reshape(1,22)

#print(x_predict[0])
#print(np.shape(x_predict))
y_test = [1,0,-7.31E-04]
y_test = np.array(y_test)
y_test = y_test.reshape(1,3)

predict(x_predict,y_test)
