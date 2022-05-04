"""
Test the performance of the best genome produced by evolve-feedforward.py.
"""

from __future__ import print_function
import os
import pickle
import neat
import visualize
from data_parser import create_passengers_objects, read_test_data

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# load the winner
with open('./winner','rb') as f:
    c = pickle.load(f)

# print('Loaded genome:')
# print(c)

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path)


node_names = {
        -1: 'is_1st_class', -2: 'is_2nd_class', -3: 'is_3rd_class', 
        -4: 'Sex', -5: 'Age', -6: 'SibSp', -7: "Parch",
        0: "Survived"}



net = neat.nn.FeedForwardNetwork.create(c, config)
passengers = create_passengers_objects(read_test_data())
fitness = 0

for passenger in passengers:
    input = passenger.actuator()
    guess = net.activate(input)
    guess_fitness = passenger.fitness(guess[0])
    fitness += guess_fitness

print()
print("Fitness:", round(fitness / len(passengers), 4))
print()


visualize.draw_net(config, c, view=True, node_names=node_names,filename="winner-enabled-pruned.gv", show_disabled=False, prune_unused=True)
