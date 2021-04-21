# DarwinBird

Set your parameters and watch as a population of AI learn how to play Flappy Bird using neural networks trained with a genetic algorithm.

## About

This game is a recreation of the classic mobile game [Flappy Bird](https://flappybird.io/), except in this version a group of AI will learn how to play the game for you. The objective of the original game was to keep your bird airborne for as long as possible without colliding with any of the pipe obstacles that cross the screen from right to left, gaining a point for each pipe passed. In this game, you will watch as many bird agents play the game simultaneously and try to achieve the highest score possible while iteratively improving themselves over many generations.

Each bird agent can see its own position, as well as the position of the next pipe obstacle, and feeds this information as input to its own [neural network](https://en.wikipedia.org/wiki/Artificial_neural_network), which will output whether the bird should flap its wings or not at any moment. When the game first begins, the weights and biases of the neural networks in all of the bird agents will be completely random and it is unlikely that any single bird will perform well. After every bird in a single generation of bird agents fails, we will use a [genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) to hopefully improve the performance of the next generation of birds. Genetic algorithms are a class of reinforcment learning algorithms that are like a naive simulation of the evolutionary process, aiming to iteratively improve a population using the concepts of:

  * **Selection:** a portion of the existing population is selected, based on some fitness metric, to carry their chromosomes to the next generation.
  * **Crossover:** the chromosomes of the selected individuals are mixed together to create children that will make up the next generation.
  * **Mutation:** with some (usually small) probability, randomly edit some chromosomes of each child.

While there are many ways to implement these concepts, the basic idea is to use selection to move towards better solutions, then use crossover to combine the best chromosomes of these solutions in the hope of making even better solutions and finally use mutation to introduce entirely new chromosomes that can help a population to escape from a local maximum and sometimes drastically improve performance. In this game, the "chromosomes" that will be selected, crossed and mutated are the weights and biases of the bird agents' neural networks and the fitness metric will be the distance (in pixels) that the bird is able to travel.

The rate at which the bird agents improve will depend on the parameters for the genetic algorithm that you can set at the beginning of the game, and the random initialization of the first generation of birds. If the parameters are set right and you're not *too* unlucky with your initial population, you should see the birds improving noticeably over 5-10 generations (there will be the ocassional downturn due to the stochastic nature of the genetic algorithm and the randomly generated pipe obstacles), and within 25-50 generations there is a good chance you will have a bird that can't be beaten. If your birds are not improving as fast as you would like, restart your game and maybe choose some different parameter values (*hint: the default values seem to work pretty well*) .


## How to Play

To play, use python3 to run DarwinBird.py. When you first start the game you will be at a start menu where you can change the mutation probability and population size as well as choose to pre-train your birds for a number of turns before animating (it is a much faster process). There is also an option to display plots that track the performance of the birds over each generation that is toggled on by default.

Once the game starts, some basic metrics about each generation will be printed to the console. After any pre-training turns are done and the game is being animated on the display, you can use the **Space Bar** to pause and un-pause the game at any time. While paused, there will be a pause menu displayed with a few options including the option to restart, which will send you back to the start menu of the game.

## Requirements

* Python 3.x
* Pygame
* Numpy
* Pandas
* Matplotlib
## To-Do

* Add an option in the pause menu to simulate a certain number of generations at any point.
* Replace the existing plots with more useful/interesting metrics.
* Add buttons to allow the player to select the n and step_size values of the mutation function.
