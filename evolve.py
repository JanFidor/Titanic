import os
import pickle
import neat
from data_parser import create_passengers_objects, read_training_data

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

generations = 25

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitness = 0
    passengers = create_passengers_objects(read_training_data())
    
    for passenger in passengers:
        # print("ok")
        input = passenger.actuator()
        guess = net.activate(input)
        # print(guess)
        guess_fitness = passenger.fitness(guess[0])
        fitness += guess_fitness
    
    return fitness


def main():
    # Load the config file, which is assumed to live in the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(6, eval_genome)
    winner = pop.run(pe.evaluate,generations)

    # Save the winner.
    with open('winner', 'wb') as f:
        pickle.dump(winner, f)

    # Show winning neural network
    print(winner)

    # node_names = {
    #     -1: 'collision_distance_left', -2: 'collision_distance_forward', -3: 'collision_distance_right', 
    #     -4: 'food_distance_left', -5: 'food_distance_forward', -6: 'food_distance_right', 
    #     0: 'move_left', 1: 'move_forward', 2: 'move_right'}
    # visualize.draw_net(config, winner, view=True, node_names=node_names,filename="winner-enabled-pruned.gv", show_disabled=False, prune_unused=True)


    # winning_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # pygame.init()
    # sim = SnakeUI(15, 15, 20, ai_interface, 30, winning_net)

    # ########################
    # # Simulate the performance of the loaded network
    # ########################    
    # sim.gameLoop()

    # print(sim.game._snake.length)
    


if __name__ == '__main__':
    main()

