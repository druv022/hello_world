import torch
from torch.autograd import Variable
from random import randint
import tensorflow as tf
import torch.nn as nn
import numpy as np
import csv
import math
import os.path
import timeit
from collections import deque
import pickle
from multiprocessing import Pool



  


    def save_networks(networks,i):
        filename='evolution'+str(i)+'.pkl'
        with open(filename, 'wb') as output:
              pickle.dump(networks, output, pickle.HIGHEST_PROTOCOL)

    def read_networks(i):
        networks=[]
        filename=str(i)+'.pkl'
        with open(filename, 'rb') as input:
             networks=pickle.load(input)  
        return networks        


class Mutate:
      
    def __init__(self,param_list):
        self.param_list=param_list
        self.nn={}
    def create_parent(self,param_list):
        self.total_choices=1
        for i in range(0,len(param_list)):
            self.total_choices*=(param_list[i]+1)  
        choices=[]
        for i in range(0,self.total_choices):
            p=[]
            for j in range(0,len(param_list)):
                
                p.append(randint(0, param_list[j]))
            choices.append(p)
  
        return choices

    def create_nueral_nets(self,choices):
            self.nueral_list=[]
            for i in range(0,len(choices)):
                
                network=Network(choices[i])
                network.create_weights()
                self.nn[network]=0.0
                self.nueral_list.append(network)
    def find_best_parents(self):
        for i in range (0,len(self.nueral_list)):
             error=self.nueral_list[i].train_network()
             self.nn[self.nueral_list[i]]=error
            #network.train_network()
             print(error.data[0])

param_list=[4,0,1,6]
#mutate=Mutate(param_list)
#choices=mutate.create_parent(param_list)
#mutate.create_nueral_nets(choices)
#mutate.find_best_parents()

#p=[1,0,1,6]
#network=Network(p)
#network.create_weights()
#network=Network.read_networks()[0]
#network.activation_function=sig
#network.train_network()
#networks=[network]
#Network.save_networks(networks)
