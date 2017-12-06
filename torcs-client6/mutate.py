from nueral import Network
from random import randint
import numpy as np
import torch
import random
from datetime import datetime
import copy
class Mutate:
    def __init__(self):
        i=1

    def create_offspring(self,network_1,network_2,layer=None):
        #process=randint(0,1)
        process=1
        # cross_over
        if(process==0):
           network_1,network_2=Mutate.do_cross_over(network_1,network_2)
        if(process==1):
            network_1,network_2=Mutate.do_mutate_network(network_1,network_2,layer)
        if(process==2):
            network_1,network_2=Mutate.do_cross_over(network_1,network_2)
            network_1,network_2=Mutate.do_cross_over(network_1,network_2)
        return network_1,network_2

    def do_cross_over(network1,network2):
        network_n1=Network()
        network_n2=Network()
        network_n1.create_weights(False)
        network_n2.create_weights(False)
        split=randint(0,21)
        for i in range(0,split):
            network_n1.dictLayers[0].data[0,i]=network1.dictLayers[0].data[0,i]
            network_n2.dictLayers[0].data[0, i] = network2.dictLayers[0].data[0, i]
        for i in range(split,22):
            network_n1.dictLayers[0].data[0, i] = network2.dictLayers[0].data[0, i]
            network_n2.dictLayers[0].data[0, i] = network1.dictLayers[0].data[0, i]
        split=randint(0,19)
        for j in range(0,62):
            for i in range(0,split):
                network_n1.dictLayers[1].data[j, i] = network1.dictLayers[1].data[j, i]
                network_n2.dictLayers[1].data[j, i] = network2.dictLayers[1].data[j, i]
            for i in range(split, 19):
                network_n1.dictLayers[1].data[j, i] = network2.dictLayers[1].data[j, i]
                network_n2.dictLayers[1].data[j, i] = network1.dictLayers[1].data[j, i]
        split = randint(0,2)
        for j in range(0,19):
            for i in range(0,split):
                network_n1.dictLayers[2].data[j, i] = network1.dictLayers[2].data[j, i]
                network_n2.dictLayers[2].data[j, i] = network2.dictLayers[2].data[j, i]
            for i in range(split, 3):
                network_n1.dictLayers[2].data[j, i] = network2.dictLayers[2].data[j, i]
                network_n2.dictLayers[2].data[j, i] = network1.dictLayers[2].data[j, i]
        return network_n1,network_n2


    def do_mutate_network(self,network1,layer):
        #layer=0
        random.seed(datetime.now())
        layer=randint(0,2)
        deviation=random.uniform(0,0.1)
        print("layer " + str(layer)+" "+str(deviation))
        if(layer==0):
          random.seed(datetime.now())
          noise = np.random.normal(0, deviation, 22*62)
          noise= np.reshape(noise,(22,62))
          noise=(torch.from_numpy(noise)).float()
          #print(network1.dictLayers[0].data)
          network1.dictLayers[0].data=network1.dictLayers[0].data+noise
          layer=1
          #print(network1.dictLayers[0].data)
          #network2.dictLayers[0].data = network2.dictLayers[0].data + noise
        # network2.dictLayers[0].data = network2.dictLayers[0].data + noise
        if (layer == 1):
            random.seed(datetime.now())
            noise = np.random.normal(0, deviation, 62*19)
            noise = np.reshape(noise, (62, 19))
            noise = (torch.from_numpy(noise)).float()
            network1.dictLayers[1].data = network1.dictLayers[1].data + noise
            layer=2
           # network2.dictLayers[1].data = network2.dictLayers[1].data + noise
        if (layer == 2):
            random.seed(datetime.now())
            noise = np.random.normal(0, deviation, 19*3)
            noise = np.reshape(noise, (19,3))
            noise = (torch.from_numpy(noise)).float()
            network1.dictLayers[2].data = network1.dictLayers[2].data + noise
            #network2.dictLayers[2].data = network2.dictLayers[2].data + noise

        return network1
    '''
    def do_mutate_network_sin(self,network1):
        layer=0
        #random.seed(datetime.now())
        #layer=randint(0,2)
        deviation = random.uniform(0, 0.008)
        network2=Network.create_copy(network1)
        print("layer " + str(layer)+" "+str(deviation))
        if(layer==0):
          random.seed(datetime.now())
          noise = np.random.normal(0, deviation, 22*62)
          noise= np.reshape(noise,(22,62))
          noise=(torch.from_numpy(noise)).float()
          #print(network1.dictLayers[0].data)
          network2.dictLayers[0].data=network2.dictLayers[0].data+noise
          layer=1
          #print(network1.dictLayers[0].data)
          #network2.dictLayers[0].data = network2.dictLayers[0].data + noise
        # network2.dictLayers[0].data = network2.dictLayers[0].data + noise
        if (layer == 1):
            deviation = random.uniform(0, 0.006)
            noise = np.random.normal(0, deviation, 62*19)
            noise = np.reshape(noise, (62, 19))
            noise = (torch.from_numpy(noise)).float()
            network2.dictLayers[1].data = network2.dictLayers[1].data + noise
            #print(network1.dictLayers[1].data)
            layer=2
           # network2.dictLayers[1].data = network2.dictLayers[1].data + noise
        if (layer == 2):
            deviation = random.uniform(0, 0.001)
            noise = np.random.normal(0, deviation, 19*3)
            noise = np.reshape(noise, (19,3))
            noise = (torch.from_numpy(noise)).float()
            network2.dictLayers[2].data = network2.dictLayers[2].data + noise
            #print(network1.dictLayers[2].data)
            #network2.dictLayers[2].data = network2.dictLayers[2].data + noise

        return network2'''

    def do_mutate_network_sin(self,network1):
        #layer=0
        #random.seed(datetime.now())
        layer=randint(0,2)
        deviation = random.uniform(0, 0.09)
        network2=Network.create_copy(network1)
        network2.layer_changed=layer
        print("layer " + str(layer)+" "+str(deviation))
        if(layer==0):
          random.seed(datetime.now())
          num_layers=randint(0,22)
          for layer in range(0,num_layers):
              layer_val=randint(0,21)
              deviation = random.uniform(0, 0.01)
              noise = np.random.normal(0, deviation, 1 * 62)
              noise= np.reshape(noise,(1,62))
              noise=(torch.from_numpy(noise)).float()
              network2.dictLayers[0].data[layer_val]=network2.dictLayers[0].data[layer_val]+noise

        if (layer == 1):
            num_layers = randint(0, 62)
            for layer in range(0,num_layers):
                layer_val = randint(0, 61)
                deviation = random.uniform(0, 0.008)
                noise = np.random.normal(0, deviation, 1 * 19)
                noise = np.reshape(noise, (1, 19))
                noise = (torch.from_numpy(noise)).float()
                network2.dictLayers[1].data[layer_val] = network2.dictLayers[1].data[layer_val] + noise


        if (layer == 2):
            num_layers = randint(0, 19)
            for layer in range(0,num_layers):
                layer_val = randint(0, 18)
                deviation = random.uniform(0, 0.004)
                noise = np.random.normal(0, deviation, 1 * 3)
                noise = np.reshape(noise, (1, 3))
                noise = (torch.from_numpy(noise)).float()
                network2.dictLayers[2].data[layer_val] = network2.dictLayers[2].data[layer_val] + noise

        return network2

    def mutate_list(self,networks):
        networks2=[]
        for j in range(0,1):
            for i in range(0,len(networks)):
                index=i
                if(index<int(0.4*len(networks))):
                    layer=randint(0,2)
                    network1=self.do_mutate_network_sin(networks[index])
                    network1.fitness=0.0
                    networks2.append(network1)
                layer=randint(0,2)
                network1=self.do_mutate_network_sin(networks[index])
                network1.fithess=0.0

                networks2.append(network1)

       # for i in range(0, 4):
       #     index = randint(2, len(networks) - 1)
        #    network1, network2 = self.create_offspring(networks[0], networks[index])
        #    network1.fithess = 0.0
        #    network2.fitness = 0.0
         #   networks2.append(network1)
         #   networks2.append(network2)
        return networks2
