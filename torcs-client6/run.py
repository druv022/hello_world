#! /usr/bin/env python3

from pytocl.main import main
from my_driver import MyDriver
from torch.autograd import Variable
from nueral import read_networks
import torch
import numpy as np
from mutate import Mutate
import os
import pickle
# 10,4,7,

def sort(networks):
    print("*",len(networks))
    networks2=[]
    for i in range(0,len(networks)):
        for j in range(i+1,len(networks)):
            if(networks[i].fitness<networks[j].fitness):
                temp=networks[i]
                networks[i]=networks[j]
                networks[j]=temp
    le=int(len(networks)/8)
    le+=1

    for i in range(0,le):
        networks2.append(networks[i])
        #networks2.append(networks[len(networks)-1])
    return networks2

if __name__ == '__main__':
    print("started")
    # MyDriver.networks=[]
    # temp_networks=[]
    # for i in range(0,62):
    #      temp_networks.append(read_networks(i))

    #     #MyDriver.add_network(read_networks(i))
    #     #MyDriver.add_network(read_networks(35))
    # temp_networks=sort(temp_networks)
    # print("**",len(temp_networks))
    # for i in range(0,len(temp_networks)):
    #     temp_networks[i].fitness=0
    #     MyDriver.add_network(temp_networks[i])
    # MyDriver.index=-1
    # best_network = MyDriver.get_best_network()
    # MyDriver.network=best_network

    # folder_name="best/"
    # os.makedirs(folder_name)
    # path=folder_name+"/"+"best"+".pkl"
    # with open(path, 'wb') as output:
    #         pickle.dump(best_network, output, pickle.HIGHEST_PROTOCOL)

    my_driver=MyDriver(logdata=False)
    my_driver.msg='dd'
    my_driver.distance=0.0
    my_driver.speed_x=0.0
    my_driver.brake=0.0
    my_driver.count=0
    my_driver.net_score={}
    MyDriver.num_childs=50
    main(my_driver)