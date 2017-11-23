import torch
import random
from functools import reduce 
from operator import add

dtype = torch.FloatTensor


class Optimizer():
	""" Class that implements genetic algorithm for MLP optimization """

	def __init__(self, retain=0.4, random_select=0.1, mutate_chance=0.2):
		"""
			Create an Optimizer
		Args:	
			retain (float): Percentage of population to retain after
                each generation
            random_select(float):     

			


		"""

		self.mutate_chance = mutate_chance
		self.random_select = random_select
		self.retain = retain

	def create_population(self, count=10, total_params):

		#create a population of random weights
		#total param 2X3+3X2
		population = []
		for i in range(0, count):
			w = torch.randn(total_params).type(dtype)

			population.append(w)

		return population

	@staticmethod
	def fitness(param):

		return param['distance_raced']+param['distance_from_start'] \
		  -0.5*(param["distance_from_start"]/param['current_lap_time'])-param['damage']

	def avg_fitness(fitness_arr):

		return reduce(add, (x for x in fitness_arr),0)/(len(fitness_arr))


	
	def mutate(self,parent,sigma = 3):
		#Randomly mutate off-springs(crossover) or some parent
		total_mutations = np.random.randint(1,len(parent))

		for i in range(total_mutations):
			ind_ = random.randint(0, len(parent))
			#mu = 0
			parent[ind_] = parent[ind_]+ np.random.normal(0, sigma)

		return parent





	def evolve(self, population):
		#Evolve population

		graded = [(individual, self.fitness(individual)) for individual in population]

		graded = [x[0] for x in sorted(graded, key = lambda x: x[1], reverse=True )]

		# Get the number that we want to keep in next gen
		retain_length = int(len(graded)*self.retain)


		#take strong contenders for parents
		parents = graded[:retain_length]

		#To add diversity add some weak contender as well
		for individual in graded[retain_length:]:
			if self.random_select > random.random()
				parents.append(individual)


		# no of places left to fill in with offsprings
		selected_parents = len(parents)
		desired_length = len(graded) - len(selected_parents)

		children = []

		# Add children which are mutated
		while len(children)< desired_length:
			p = random.randint(0,selected_parents-1)

			#get a random parent
			p_ = parents[p]

			#mutate
			if self.mutate_chancec > random.random()
				p_ = mutate(p_)

			children.append(p_)

		parents.extend(children)

		return parents
















#N = batch_size
N, D_in, H, D_out = 64, 1000, 100, 3

#create random data
x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)

#randomly intialize weights
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

for t in range(50):
	#Forward pass

	h = x.mm(w1)
	h = h.clamp(min=0)
	y_pred = h.mm(w2)

	#Compute Loss
	loss = (y_pred-y).pow(2).sum()
	print(t, loss)


