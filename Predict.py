import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn import model_selection
#from sklearn.preprocessing import StandardScaler

def predict(x_predict, y_test):
    filename = "TrainedModel.p"
    model = pickle.load(open(filename, 'rb'))

    filename = "TrainScale.p"
    scaler = pickle.load(open(filename, 'rb'))


    print("x_predict",x_predict)
    x_predict = scaler.transform(x_predict)
    print("x_predict", x_predict)

    y_predict = model.predict(x_predict)
    print("y_predict:", y_predict)
    print("y_test:", y_test)
    return y_predict

x_predict =[-0.0379823,-5.62E-05,4.30E-04,5.00028,5.0778,5.32202,5.77526,6.52976,7.78305,10.008,14.6372,28.8659,200,28.7221,14.6009,9.99199,7.7742,6.52431,5.77175,5.31976,5.07646,4.99972]
x_predict1 = [97.1831,-0.525232,-0.0655026,7.6424,7.6712,7.93927,8.49036,9.42563,10.9559,13.5449,18.3768,29.3664,65.1841,27.2412,8.68478,5.41058,4.02927,3.29218,2.85748,2.59448,2.44415,2.37903]

x_predict = np.array(x_predict)
print(np.shape(x_predict))
x_predict = x_predict.reshape(1,22)

x_predict1 = np.array(x_predict1)
print(np.shape(x_predict1))
x_predict1 = x_predict1.reshape(1,22)

#print(x_predict[0])
#print(np.shape(x_predict))
y_test = [1,0,-1.64E-05]
y_test1 = [1,0,0.001123673]
y_test = np.array(y_test)
y_test = y_test.reshape(1,3)

y_test1 = np.array(y_test1)
y_test1 = y_test1.reshape(1,3)

predict(x_predict,y_test)
predict(x_predict1,y_test1)
