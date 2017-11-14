import csv
import argparse
import glob
import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path_folder', type=str, help='path of folder containing data', default="train_data")
    parser.add_argument(
        '--type', type=str, help='json or csv or txt...', default=".csv")

    args = parser.parse_args()
    # print(args)
    return args


# read the data perform train and val split
def read_dataset():
    args = get_args()
    x_, y_ = [], []

    path = args.path_folder
    for file_ in glob.glob(path + "/*" + args.type):
        print(file_)
        with open(file_) as csvFile:
            reader_csv = csv.reader(csvFile, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
            for rows in reader_csv:
                y_attr = rows[0:3]
                x_attr = rows[3:]
                # print(x_attr)

                x_attr = [i if type(i) != str else 0 for i in x_attr]
                y_attr = [i if type(i) != str else 0 for i in y_attr]
                # print(x_attr)

                x_.append(x_attr)
                y_.append(y_attr)

    y = np.array(y_)
    x = np.array(x_)
    print(x.shape, y.shape)
    print(np.all(np.isfinite(x)), np.all(np.isfinite(y)))

    num = 0
    if num < 5:
        print("x:",x)
        print("y",y)
        num +=1

    return x, y


def train_mlp():
    x, y = read_dataset()

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=.10, random_state=7)

    model = MLPRegressor(hidden_layer_sizes=(30, 15, 15), solver='sgd', alpha=0.0001, batch_size='auto',
                         learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=300, shuffle=True,
                         random_state=None, tol=0.0001, verbose=True, warm_start=False, momentum=0.9,
                         nesterovs_momentum=True,
                         early_stopping=True, validation_fraction=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    scaler = StandardScaler()
    scaler.fit(x_train)

    print("x_train", x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    print("x_train", x_train)

    model.fit(x_train, y_train)

    return model.score(x_test, y_test), model,scaler

def get_model():
    score, model_, scaler_ = train_mlp()

    flag = True
    i = 0
    while flag or i < 20:
        if score > 0.60:
            filename = "TrainedModel.p"
            pickle.dump(model_, open(filename, 'wb'))

            filename = "TrainScale.p"
            pickle.dump(scaler_, open(filename, 'wb'))

            print("Model saved: score", score)
            flag = False
            i=20
        else:
            score, model_, scaler_ = train_mlp()
            i += 1


get_model()

#read_dataset()
