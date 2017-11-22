from pytocl.driver import Driver
from pytocl.car import State, Command


# # added
# import logging
import sys
import math
import csv
from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

# _logger = logging.getLogger(__name__)



class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    ...
    # def drive(self, carstate: State) -> Command:
    #     # Interesting stuff
    #     command = Command(...)
    #     return command

    #added
    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """
        f = open("Test_data3.txt","a+")
        f.write("\n*** "+str(carstate)+"\n")

        command = Command()

        # reding the model
        filename = "TrainedModel.p"
        model = pickle.load(open(filename, 'rb'))
        filename = "TrainScale.p"
        scaler = pickle.load(open(filename, 'rb'))
        
        # constructing the input 
        x_predict = [carstate.speed_x]
        x_predict.append(carstate.distance_from_center)
        x_predict.append(carstate.angle)
        [x_predict.append(i) for i in carstate.distances_from_edge]
        x_predict = np.array(x_predict)
        x_predict = x_predict.reshape(1,22)

        x_predict = scaler.transform(x_predict)

        # predicting the output 
        y_predict = model.predict(x_predict)
        

        command.accelerator = y_predict[0][0]
        if command.accelerator < 0:
        	command.brake = y_predict[0][1]
        command.steering = y_predict[0][2]

        if carstate.rpm < 2500 and carstate.gear > 0:
            command.gear = carstate.gear - 1
        elif carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        if not command.gear:
            command.gear = carstate.gear or 1


        
        # self.steer(carstate, 0.0, command)

        # # ACC_LATERAL_MAX = 6400 * 5
        # # v_x = max(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
        # v_x = 90

        # self.accelerate(carstate, v_x, command)

        # if self.data_logger:
        #     self.data_logger.log(carstate, command)

        f = open("Test_data3.txt","a+")
        f.write("### "+str(command))
        f.close()


        # data = []
        # data.append(command.accelerator)
        # data.append(command.brake)
        # data.append(command.steering)
        # data.append(carstate.speed_x)
        # data.append(carstate.distance_from_center)
        # data.append(carstate.angle)
        # [data.append(i) for i in carstate.distances_from_edge]

        # with open('data.csv','a+') as f:
        #     f.write("\n"+str(data))


        return command

    def accelerate(self, carstate, target_speed, command):
        # compensate engine deceleration, but invisible to controller to
        # prevent braking:
        speed_error = 1.0025 * target_speed * MPS_PER_KMH - carstate.speed_x
        acceleration = self.acceleration_ctrl.control(
            speed_error,
            carstate.current_lap_time
        )

        # stabilize use of gas and brake:
        acceleration = math.pow(acceleration, 3)

        if acceleration > 0:
            if abs(carstate.distance_from_center) >= 1:
                # off track, reduced grip:
                acceleration = min(0.4, acceleration)

            command.accelerator = min(acceleration, 1)

            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        #else:
             #command.brake = min(-acceleration, 1)

        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

    def steer(self, carstate, target_track_pos, command):
        steering_error = target_track_pos - carstate.distance_from_center
        command.steering = self.steering_ctrl.control(
            steering_error,
            carstate.current_lap_time
        )
