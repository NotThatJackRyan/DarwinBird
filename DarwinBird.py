import math
import numpy as np
import random
import pygame
from pygame.locals import *
import pandas as pd
import matplotlib.pyplot as plt
import sys

FRAME_ANIMATION_WIDTH = 3      # ]
FRAME_BIRD_DROP_HEIGHT = 2     # ] - pixels per frame
FRAME_BIRD_JUMP_HEIGHT = 5     # ]
BIRD_JUMP_STEPS = 25
WIN_WIDTH = 600
WIN_HEIGHT = 512
PIPE_WIDTH = 60
BIRD_RADIUS = 16
BIRD_X = 50


# Defines a perceptron layer to be used in the feed forward neural network of
# a Bird object.
class PerceptronLayer(object):

    def __init__(self, W=None, m=None, n=None):
        """
        Initialize the perceptron layer, either by taking in a matrix
        representation of its weights and biases as W, or by generating random
        weights in the shape of an (m)x(n) matrix.

        :param W: a matrix representation of weights and biases
        :param m: the number of perceptron outputs (rows)
        :param n: the number of inputs (columns)
        """

        self.W = None
        self.bias = None

        if W is not None:
            # Use given weight matrix
            self.W = W[:, :-1]
            self.bias = W[:, 1]
        else:
            # Random weights via Xavier Initialization
            self.W = np.random.normal(
                loc=0.0,
                scale=np.sqrt(2 / (n + m)),
                size=(m, n)
            )
            # Bias will also be initialized with a normal distribution,
            # instead centered about 1.0.
            self.bias = np.random.normal(
                loc=1.0,
                scale=np.sqrt(2 / (m + 1)),
                size=m
            )

        self.m, self.n = np.shape(self.W)

    def forward(self, inputs):
        """
        Calculates a forward pass through the perceptron layer.

        :param inputs: the input values to the layer (a vector of length n)
        :return: the dot product of the input and the layer weights, plus the
                 layer bias (vector of length m).
        """
        return np.dot(self.W, inputs) + self.bias


# Represents a "bird" agent.
class Bird(object):

    def __init__(self, hidden_layer, output_layer, color):
        """
        Initializes a new bird. The hidden layer must have n = 4, output layer
        n = hidden layer m and output layer m = 1.

        :param hidden_layer: PerceptronLayer object
        :param output_layer: PerceptronLayer object
        :param color: RGB values of the birds color (3-tuple of Int [0, 255])
        """
        self.distance = 0
        self.color = color
        self.y = 0
        self.steps_to_jump = 0
        self.alive = True

        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

    def flap(self, target_x, target_y1, target_y2):
        """
        Based on some information about the incoming pipe pair, the bird decides
        to jump by feeding the values forward through its neural network and
        interpreting its binary output.

        :param target_x: the x-value of the end of the next pipe pair (Int)
        :param target_y1: the y-value of the top pipe in the next pair (Int)
        :param target_y2: the y-value of the bottom pipe in the next pair (Int)
        :return: True if the output is less than 0.5, else False (Boolean)
        """

        hidden_output = self.hidden_layer.forward((target_x, target_y1, target_y2, self.y))
        output = self.output_layer.forward(hidden_output)
        output = 1 / (1 + np.exp(np.float128(-output)))

        if output < 0.5:
            return True
        else:
            return False


# Step 1 of the genetic algorithm.
def natural_selection(population):
    """
    Selects a bird randomly from a distribution of the population that has
    probabilities proportional to the distance achieved by each bird.

    :param population: a list of Bird objects
    :return: a randomly selected Bird object
    """

    selector = random.random()
    total_fitness = sum(bird.distance for bird in population)
    cum_sum = 0

    for bird in population:
        cum_sum += bird.distance
        if cum_sum / total_fitness >= selector:
            return bird


# Step 2 of the genetic algorithm.
def crossover(p1, p2):
    """
    Given two parent birds, returns a child bird whose genetic makeup (weight
    and bias values of its hidden layer and output layer) is a mix of the patents'.
    The specific crossover method is Two-point crossover.

    :param p1: parent 1 (Bird object)
    :param p2: parent 2 (Bird object)
    :return: child (bird object)
    """
    # Get the weights and biases of the parent 1's hidden layer
    p1_hidden_W = p1.hidden_layer.W
    p1_hidden_b = p1.hidden_layer.bias.reshape(len(p1.hidden_layer.bias), 1)
    p1_hidden = np.append(p1_hidden_W, p1_hidden_b, axis=1)
    hidden_shape = np.shape(p1_hidden)  # also save the shape of the hidden layer
    p1_hidden = np.ndarray.flatten(p1_hidden)  # flatten the matrix for crossover

    # Repeat for parent 2
    p2_hidden_W = p2.hidden_layer.W
    p2_hidden_b = p2.hidden_layer.bias.reshape(len(p2.hidden_layer.bias), 1)
    p2_hidden = np.ndarray.flatten(np.append(p2_hidden_W, p2_hidden_b, axis=1))

    # Get weights and biases for parent 1's output layer
    p1_output_W = p1.output_layer.W
    p1_output_b = p1.output_layer.bias.reshape(len(p1.output_layer.bias), 1)
    p1_output = np.append(p1_output_W, p1_output_b, axis=1)
    output_shape = np.shape(p1_output)  # save shape of output layer as well
    p1_output = np.ndarray.flatten(p1_output)

    # Repeat for parent 2
    p2_output_W = p2.output_layer.W
    p2_output_b = p2.output_layer.bias.reshape(len(p2.output_layer.bias), 1)
    p2_output = np.ndarray.flatten(np.append(p2_output_W, p2_output_b, axis=1))

    # Combine the flattened arrays of parent weights into one long array
    p1_dna = np.concatenate((p1_hidden, p1_output))
    p2_dna = np.concatenate((p2_hidden, p2_output))

    # Randomly select the indices of th 2 crossover points
    idx1 = random.randrange(0, len(p1_dna))
    idx2 = random.randrange(idx1, len(p1_dna))

    # Two-point crossover generates two children but we only want one of those
    # children. Here we flip a coin to select either the child with parent 1's
    # dna on the outside or parent 2's dna on the outside.
    if random.random() > 0.5:
        child1_dna = np.concatenate((p1_dna[0:idx1], p2_dna[idx1:idx2], p1_dna[idx2:]))
        child1_hidden = child1_dna[0:len(p1_hidden)].reshape(hidden_shape)
        child1_output = child1_dna[len(p1_hidden):].reshape(output_shape)
        return Bird(PerceptronLayer(W=child1_hidden),
                    PerceptronLayer(W=child1_output),
                    color=p1.color)
    else:
        child2_dna = np.concatenate((p2_dna[0:idx1], p1_dna[idx1:idx2], p2_dna[idx2:]))
        child2_hidden = child2_dna[0:len(p1_hidden)].reshape(hidden_shape)
        child2_output = child2_dna[len(p1_hidden):].reshape(output_shape)
        return Bird(PerceptronLayer(W=child2_hidden),
                    PerceptronLayer(W=child2_output),
                    color=p1.color)


# Step 3 of the genetic algorithm
def mutation(child, p, n, step_size):
    """
    Given a child (Bird object), mutate n randomly selected chromosomes
    (weights/biases) with probability p. A mutation of a chromosome
    is done by adding a random normal value centered about 0 with a standard
    deviation equal to the step_size.

    :param child: a Bird object to be mutated
    :param p: probability of mutation (Float [0.0, 1])
    :param n: number of mutations (Int [0, # of chromosomes])
    :param step_size: standard deviation of the mutation step (Float (0.0, Inf))
    :return: the (possibly) mutated child.
    """

    def rnorm():
        """
        generates random normal variable centered at 0 with a standard
        deviation equal to the step_size parameter.

        :return: random normal variable (Float)
        """
        return np.random.normal(loc=0.0, scale=step_size)

    # Mutate n different points
    for i in range(n):
        # Only perform mutation i with probability p
        if random.random() <= p:

            # Randomly select which part (which layer and whether to change
            # a weight or bias) of the child to mutate.
            part_selector = random.randrange(1, 5, 1)

            # Hidden Layer Weights
            if part_selector == 1:
                m, n = np.shape(child.hidden_layer.W)
                m_selector = random.randrange(0, m)
                n_selector = random.randrange(0, n)
                child.hidden_layer.W[m_selector][n_selector] += rnorm()

            # Hidden Layer Bias
            elif part_selector == 2:
                selector = random.randrange(0, len(child.hidden_layer.bias))
                child.hidden_layer.bias[selector] += rnorm()

            # Output Layer Weights
            elif part_selector == 3:
                m, n = np.shape(child.output_layer.W)
                m_selector = random.randrange(0, m)
                n_selector = random.randrange(0, n)
                child.output_layer.W[m_selector][n_selector] += rnorm()

            # Output Layer Bias
            else:
                selector = random.randrange(0, len(child.output_layer.bias))
                child.output_layer.bias[selector] += rnorm()

    return child


# Represents a pipe pair obstacle that birds need to pass through.
class PipePair:

    def __init__(self):

        """
        Creates a pipe pair at the very end of the display surface with a gap
        centered around a point that has a randomly selected y-value no closer
        to the top/bottom of the window than 100 pixels.
        """
        self.center_x = WIN_WIDTH + PIPE_WIDTH / 2
        self.center_y = random.randrange(100, WIN_HEIGHT - 100)

        self.x1 = 0
        self.x2 = 0
        self.y_top = 0
        self.y_bot = 0

        self.color = (10, 70, 135)
        self.score_counted = False

        self.calc_coords()

    def draw_pipes(self, surface):
        """
        Draws the pipe pair to the display surface of the game.

        :param surface: a Pygame Surface object
        """

        top_pipe = pygame.Rect(self.x1, 0, self.x2-self.x1, self.y_top)
        bot_pipe = pygame.Rect(self.x1, self.y_bot, self.x2-self.x1, WIN_HEIGHT)

        pygame.draw.rect(surface, self.color, top_pipe)
        pygame.draw.rect(surface, self.color, bot_pipe)

    def calc_coords(self):
        """
        Uses the coordinates of the centering point to calculate the vertices
        necessary to represent the pipe pair on screen. Makes it simple to
        change the position of the pipe pair by only changing the coordinates
        of the centering point.
        """
        self.x1 = self.center_x - PIPE_WIDTH / 2
        self.x2 = self.center_x + PIPE_WIDTH / 2
        self.y_top = self.center_y - int(5 * BIRD_RADIUS)
        self.y_bot = self.center_y + int(5 * BIRD_RADIUS)


def check_collision(bird, pipe):
    """
    Checks for a collision between the circular bird and the rectangular pipes
    in the pipe pair. Assumes pipes always approach the bird from right to left.
    :param bird: a Bird object
    :param pipe: a PipePair object
    :return: True if the circular surface of the bird intersects with any
    point within the rectangular surfaces of the pipe pairs, else return false.
    """

    # Short Circuit while the pipes are too far away for any possible collision
    if pipe.x1 > BIRD_X + BIRD_RADIUS + 3:
        return False

    # Check if bird hits the side of a pipe
    if pipe.x1 >= BIRD_X and (bird.y < pipe.y_top or bird.y > pipe.y_bot):
        return True

    # Check if bird hits the inside edge of a pipe.
    if pipe.x1 <= BIRD_X <= pipe.x2:
        if bird.y + BIRD_RADIUS > pipe.y_bot:
            return True
        if bird.y - BIRD_RADIUS < pipe.y_top:
            return True

    # Bird center as numpy array
    c = np.array((BIRD_X, bird.y))
    # All remaining collision scenarios involve one of the inner 4 points of
    # the pipe pair
    # calculate if the bird touches any of the pipe corners by checking if the
    # euclidean distance (L2 norm) is less than BIRD_RADIUS
    if np.linalg.norm(c - np.array((pipe.x1, pipe.y_top))) < BIRD_RADIUS:
        return True
    if np.linalg.norm(c - np.array((pipe.x2, pipe.y_top))) < BIRD_RADIUS:
        return True
    if np.linalg.norm(c - np.array((pipe.x1, pipe.y_bot))) < BIRD_RADIUS:
        return True
    if np.linalg.norm(c - np.array((pipe.x2, pipe.y_bot))) < BIRD_RADIUS:
        return True

    # If the function exits here, the bird is somewhere in between
    # the pieces of the pipe pair
    return False


def get_frame_jump_height(jump_step):
    """
    Calculates how much a bird should rise at each stage of its jump.

    :param jump_step: the current stage (1-25) of a birds jump process (Int)
    :return: number of pixels the bird should rise for this current step (Float)
    """
    frac_jump_done = jump_step / float(BIRD_JUMP_STEPS)
    return (1 - math.cos(frac_jump_done * math.pi)) * FRAME_BIRD_JUMP_HEIGHT


def random_color():
    """
    Generates a random RGB representation of a color.

    :return: the RGB representation of a random color (3 tuple of Int [0, 255])
    """
    return [random.randrange(0, 255) for i in range(3)]


def score_graph(df, graph):
    """
    Updates the graph of the max scores over the generations.

    :param df: pandas data frame of statistics tracked during the game.
    :param graph: a matplotlib subplot.
    """
    graph.clear()
    graph.plot(df["Generation"], df["Score"])
    graph.set_title("Best Score by Generation")
    graph.set_xlabel("Generation")
    graph.set_ylabel("Score")


def distance_graph(df, graph):
    """
    Updates the graph describing the distances of the birds in each generation.

    :param df: pandas data frame of statistics tracked during the game.
    :param graph: a matplotlib subplot.
    """
    graph.clear()
    graph.plot(df["Generation"], df["Max Distance"], label="Max Distance")
    graph.plot(df["Generation"], df["Mean Distance"], label="Mean Distance")
    graph.set_title("Distance by Generation")
    graph.set_xlabel("Generation")
    graph.set_ylabel("Distance")
    graph.legend(loc="upper left")


# A class defining a rectangular menu button.
class MenuButton:

    def __init__(self, x, y, width, height, text):
        """
        Initialize the button.

        :param x: x-value of top left corner (Int)
        :param y: y-value of top left corner (Int)
        :param width: width of button (Int)
        :param height: height of button (Int)
        :param text: text displayed on button surface (Rendered Pygame Font object)
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.default_color = (10, 70, 135)
        self.highlight_color = (83, 116, 155)

    def mouse_over(self, mouse):
        """
        Detects if the mouse cursor is over the button.

        :param mouse: coordinates of the mouse (tuple of Int)
        :return: True if the cursor is over the button, else false.
        """
        xrange = self.x <= mouse[0] <= self.x + self.width
        yrange = self.y <= mouse[1] <= self.y + self.height
        return xrange and yrange

    def animate(self, surface, mouse):
        """
        Draws the button on the menu surface. Will be a different color
        depending on if the mouse cursor is over the button or not.

        :param surface: a pygame Surface object.
        :param mouse: coordinates of the mouse (tuple of Int)
        :return: None
        """
        if self.mouse_over(mouse):
            pygame.draw.rect(surface, self.highlight_color, self.rect)
        else:
            pygame.draw.rect(surface, self.default_color, self.rect)
        surface.blit(self.text,
                     (self.x + self.width/2 - self.text.get_width()/2,
                      self.y + self.height/2 - self.text.get_height()/2))


# A class defining a button used to increment parameter values with a plus and
# minus button.
class IncrementButton:

    def __init__(self, x, y, width, height, label, value):
        """
        Initialize the button.

        :param x: x-value of top left corner (Int)
        :param y: y-value of top left corner (Int)
        :param width: width of button (Int)
        :param height: height of button (Int)
        :param label: text displayed above button surface (Rendered Pygame Font object)
        :param value: the initial value to be displayed on the button (Int or Float)
        """

        pygame.font.init()
        self.value_font = pygame.font.SysFont(None, 32)

        self.label = label
        self.value = value
        self.value_text = self.value_font.render(str(self.value), True, (0, 0, 0))
        self.plus = self.value_font.render("+", True, (0, 0, 0))
        self.minus = self.value_font.render("-", True, (0, 0, 0))

        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.default_color = (10, 70, 135)
        self.highlight_color = (83, 116, 155)

        self.minus_box = pygame.Rect(self.x, self.y,
                                     self.width/4, self.height)
        self.plus_box = pygame.Rect(self.x + 3*self.width/4,
                                    self.y, self.width/4, self.height)
        self.mid_box = pygame.Rect(self.x + self.width/4, self.y,
                                   self.width/2, self.height)

    def over_minus(self, mouse):
        """
        Detects when the mouse cursor os over the minus button.

        :param mouse: coordinates of the mouse (tuple of Int)
        :return: True if the cursor is over the minus button, else false.
        """
        xrange = self.x <= mouse[0] <= self.x + self.width/4
        yrange = self.y <= mouse[1] <= self.y + self.height
        return xrange and yrange

    def over_plus(self, mouse):
        """
        Detects when the mouse cursor is over the plus button.

        :param mouse: coordinates of the mouse (tuple of Int)
        :return: rue if the cursor is over the plus button, else false.
        """
        xrange = self.x + 3*self.x/4 <= mouse[0] <= self.x + self.width
        yrange = self.y <= mouse[1] <= self.y + self.height
        return xrange and yrange

    def increment(self, step):
        """
        Increments the value stored and displayed by the button by the given
        step amount.

        :param step: the amount the value should be incremented by (or decremented
        when negative)
        :return: None
        """
        self.value += step
        # Rounds to avoid displaying floating point errors
        self.value = round(self.value, 2)
        self.value_text = self.value_font.render(str(self.value), True, (0, 0, 0))

    def animate(self, surface, mouse):
        """
        Draws the button on the menu surface. Will be a different color
        depending on if the mouse cursor is over one of the buttons or not.

        :param surface: a pygame Surface object.
        :param mouse: coordinates of the mouse (tuple of Int)
        :return: None
        """
        surface.blit(self.label,
                     (self.x + self.width/2 - self.label.get_width()/2,
                      self.y - self.height/2 + self.label.get_height()))
        pygame.draw.rect(surface, self.default_color, self.mid_box)
        surface.blit(self.value_text,
                     (self.x + self.width/2 - self.value_text.get_width()/2,
                      self.y + self.height/2 - self.value_text.get_height()/2))
        if self.over_minus(mouse):
            pygame.draw.rect(surface, self.highlight_color, self.minus_box)
        else:
            pygame.draw.rect(surface, self.default_color, self.minus_box)
        if self.over_plus(mouse):
            pygame.draw.rect(surface, self.highlight_color, self.plus_box)
        else:
            pygame.draw.rect(surface, self.default_color, self.plus_box)

        surface.blit(self.minus, (self.x + self.width/8 - self.minus.get_width()/2,
                                  self.y + self.height/2 - self.minus.get_height()/2))
        surface.blit(self.plus, (self.x + 7*self.width/8 - self.plus.get_width()/2,
                                 self.y + self.height/2 - self.plus.get_height()/2))


def start_menu():
    """
    Contains the logic for the start menu of the game. I put it into a method
    for the sake of cleanliness. When the menu is exited by starting the game,
    the parameter values chosen by the user will be returned.

    :return: pop_size: the initial population size of each genration of birds (Int > 0)
    :return: p_mut: probability of a mutation taking place (Float [0.0, 1.0])
    :return: sim_len: number of generations to pre-train/simulate before animating (Int > 0)
    :return: show_plots: the users choice whether or not to display plots about the
             performance of each generation (Boolean)
    """

    in_menu = True
    pygame.init()
    menu_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption('Darwin Bird')

    # Default values for toggle-able variables
    pop_size = 100
    p_mut = 0.15
    sim_len = 0
    show_plots = True

    # Render the text for the buttons.
    button_font = pygame.font.SysFont(None, 32)
    label_font = pygame.font.SysFont(None, 16)
    start_text = button_font.render("Start", True, (0, 0, 0))
    pop_text = label_font.render("Population Size", True, (0, 0, 0))
    mut_text = label_font.render("Mutation Probability", True, (0, 0, 0))
    sim_text = label_font.render("Pre-Training Turns", True, (0, 0, 0))
    plot_text = button_font.render("Show Plots: True", True, (0, 0, 0))
    quit_text = button_font.render("Quit", True, (0, 0, 0))

    # Position variables for the menu buttons
    b_width = 200
    b_height = 50
    b_dist = 20
    b_x = WIN_WIDTH/2 - b_width/2
    start_y = 50
    pop_y = start_y + b_height + b_dist
    mut_y = pop_y + b_height + b_dist
    sim_y = mut_y + b_height + b_dist
    plot_y = sim_y + b_height + b_dist
    quit_y = plot_y + b_height + b_dist

    # Initialize all of the necessary buttons
    start_button = MenuButton(b_x, start_y, b_width, b_height, start_text)
    pop_button = IncrementButton(b_x, pop_y, b_width, b_height, pop_text, pop_size)
    mut_button = IncrementButton(b_x, mut_y, b_width, b_height, mut_text, p_mut)
    sim_button = IncrementButton(b_x, sim_y, b_width, b_height, sim_text, sim_len)
    plot_button = MenuButton(b_x, plot_y, b_width, b_height, plot_text)
    quit_button = MenuButton(b_x, quit_y, b_width, b_height, quit_text)

    # The loop that contains the logic for the menu
    while in_menu:

        menu_surface.fill((255, 255, 255))
        mouse = pygame.mouse.get_pos()

        # Display the different buttons
        start_button.animate(menu_surface, mouse)
        pop_button.animate(menu_surface, mouse)
        mut_button.animate(menu_surface, mouse)
        sim_button.animate(menu_surface, mouse)
        plot_button.animate(menu_surface, mouse)
        quit_button.animate(menu_surface, mouse)

        # Button functionality
        for e in pygame.event.get():

            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if e.type == pygame.MOUSEBUTTONDOWN:

                # Start Button
                if start_button.mouse_over(mouse):
                    in_menu = False

                # Quit Button
                if quit_button.mouse_over(mouse):
                    pygame.quit()

                # Population button
                if pop_button.over_plus(mouse):
                    pop_button.increment(10)
                    pop_size += 10
                if pop_button.over_minus(mouse) and pop_size >= 20:
                    pop_button.increment(-10)
                    pop_size -= 10

                # Mutation button
                if mut_button.over_plus(mouse) and p_mut < 1.0:
                    mut_button.increment(0.01)
                    p_mut += 0.01
                    p_mut = round(p_mut, 2)
                if mut_button.over_minus(mouse) and p_mut > 0.0:
                    mut_button.increment(-0.01)
                    p_mut -= 0.01
                    p_mut = round(p_mut, 2)

                # Simulation Button
                if sim_button.over_plus(mouse):
                    if sim_len < 50:
                        sim_button.increment(5)
                        sim_len += 5
                    elif sim_len < 100:
                        sim_button.increment(10)
                        sim_len += 10
                    else:
                        sim_button.increment(50)
                        sim_len += 50
                if sim_button.over_minus(mouse) and sim_len > 0:
                    if sim_len <= 50:
                        sim_button.increment(-5)
                        sim_len -= 5
                    elif sim_len <= 100:
                        sim_button.increment(-10)
                        sim_len -= 10
                    else:
                        sim_button.increment(-50)
                        sim_len -= 50

                # Plot Button
                if plot_button.mouse_over(mouse):
                    if show_plots:
                        plot_button.text = button_font.render(
                            "Show Plots: False", True, (0, 0, 0))
                        show_plots = False
                    else:
                        plot_button.text = button_font.render(
                            "Show Plots: True", True, (0, 0, 0))
                        show_plots = True

        pygame.display.update()

    return pop_size, p_mut, sim_len, show_plots


def pause_menu():
    """
    When called, this function will display the pause menu, which will remain
    until the user decides to resume, restart or quit the game.
    
    :return: True if the restart option was selected, else False. (Boolean)
    """

    pause_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption('Darwin Bird')

    button_font = pygame.font.SysFont(None, 32)
    resume_text = button_font.render("Resume", True, (0, 0, 0))
    restart_text = button_font.render("Restart", True, (0, 0, 0))
    quit_text = button_font.render("Quit", True, (0, 0, 0))

    b_width = 200
    b_height = 50
    b_dist = 50
    b_x = WIN_WIDTH / 2 - b_width / 2
    resume_y = 200
    restart_y = resume_y + b_height + b_dist
    quit_y = restart_y + b_height + b_dist

    resume_button = MenuButton(b_x, resume_y, b_width, b_height, resume_text)
    restart_button = MenuButton(b_x, restart_y, b_width, b_height, restart_text)
    quit_button = MenuButton(b_x, quit_y, b_width, b_height, quit_text)

    while True:

        pause_surface.fill((255, 255, 255))
        mouse = pygame.mouse.get_pos()

        # Display the buttons
        resume_button.animate(pause_surface, mouse)
        restart_button.animate(pause_surface, mouse)
        quit_button.animate(pause_surface, mouse)

        # Button functionality
        for e in pygame.event.get():

            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

            # Pause key (Space bar)
            if e.type == KEYUP and e.key == K_SPACE:
                return False

            if e.type == pygame.MOUSEBUTTONDOWN:

                # Resume Button
                if resume_button.mouse_over(mouse):
                    return False

                # Restart Button
                if restart_button.mouse_over(mouse):
                    return True

                # Quit Button
                if quit_button.mouse_over(mouse):
                    pygame.quit()

        pygame.display.update()


# Game loop
while True:

    restart = False

    bird_stats_df = None
    fig = None
    graph1 = None
    graph2 = None
    pop_size, p_mut, sim_length, plot = start_menu()
    if plot:
        bird_stats_df = pd.DataFrame(columns=["Generation",
                                              "Score",
                                              "Max Distance",
                                              "Mean Distance"])
        fig = plt.figure()
        graph1 = fig.add_subplot(211)
        graph2 = fig.add_subplot(212, sharex=graph1)
    gen = 1
    best_score = 0
    # Create first generation of Birds
    bird_colors = [random_color() for i in range(pop_size)]
    birds = []
    for col in bird_colors:
        hidden_layer = PerceptronLayer(m=6, n=4)
        output_layer = PerceptronLayer(m=1, n=6)
        new_bird = Bird(hidden_layer, output_layer, col)
        new_bird.y = int(WIN_HEIGHT / 2 - BIRD_RADIUS)
        birds.append(new_bird)

    # Generation Loop (each loop represents one generation in the game)
    while not restart:
        pygame.init()
        display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption('Darwin Bird')
        gen_font = pygame.font.SysFont(None, 32, bold=True)
        display_font = pygame.font.SysFont(None, 24, bold=True)

        # Display a standby screen while simulating/pre-training generations
        if sim_length > 0:
            display_surface.fill((255, 255, 255))
            simulation_text1 = display_font.render(
                "Simulating " + str(sim_length) +
                " more generations before animating.",
                True, (0, 0, 0))
            text_x = WIN_WIDTH/2 - simulation_text1.get_width()/2
            text_y = WIN_HEIGHT/2 - simulation_text1.get_height()/2
            display_surface.blit(simulation_text1, (text_x, text_y))
            simulation_text2 = display_font.render(
                "Press [SPACEBAR] to begin animation now.",
                True, (0, 0, 0))
            text_x = WIN_WIDTH/2 - simulation_text2.get_width()/2
            text_y = WIN_HEIGHT - 2*simulation_text2.get_height()
            display_surface.blit(simulation_text2, (text_x, text_y))
            pygame.display.update()

        gen_score = 0
        alive_count = len(birds)
        pipes = [PipePair()]
        # Frame by frame Loop
        while alive_count > 0:

            # Check for keyboard input.
            if sim_length < 1:
                for e in pygame.event.get():
                    if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                        pygame.quit()
                        sys.exit(0)
                    elif e.type == KEYUP and e.key in (K_PAUSE, K_p, K_SPACE):
                        restart = pause_menu()
                        if restart and plot:
                            plt.close(fig)
                display_surface.fill((255, 255, 255))
            else:
                for e in pygame.event.get():
                    if e.type == KEYUP and e.key == K_SPACE:
                        sim_length = 0

            if restart:
                # This will return the program to the start menu.
                break
            # Calculations for pipe pairs between frames.
            # Iterate over a copy of pipes because we are
            # removing items from the list.
            for pipe in pipes[:]:

                # Move the pipe pair to the left and update its coordinates.
                pipe.center_x -= FRAME_ANIMATION_WIDTH
                pipe.calc_coords()
                # Count the score for the generation if at least one
                # bird makes it past.
                if pipe.x2 < BIRD_X - BIRD_RADIUS and not pipe.score_counted:
                    gen_score += 1
                    best_score = max(gen_score, best_score)
                    pipe.score_counted = True
                # Remove a pipe when it is no longer on screen.
                if pipe.x2 < 0:
                    pipes.remove(pipe)
                    continue
                if sim_length < 1:
                    pipe.draw_pipes(display_surface)

            # Add a new pipe when the last pipe crosses halfway
            # through the screen.
            # This method of adding new pipes is not robust to changes in the
            # WIN_WIDTH constant.
            if len(pipes) == 1 and pipes[0].x1 == WIN_WIDTH / 2:
                pipes.append(PipePair())
            elif len(pipes) > 1 and pipes[1].x1 == WIN_WIDTH / 2:
                pipes.append(PipePair())

            # Calculations for birds between frames
            for bird in birds:
                if bird.alive:
                    # Calculate bird y position
                    if bird.steps_to_jump > 0:
                        bird.y -= int(
                            get_frame_jump_height(BIRD_JUMP_STEPS - bird.steps_to_jump))
                        bird.steps_to_jump -= 1
                    # Birds do not start falling until the first pipe appears
                    elif len(pipes) > 0:
                        bird.y += FRAME_BIRD_JUMP_HEIGHT
                    # Check if bird is still in bounds (center within display)
                    if bird.y < 0 or bird.y > WIN_HEIGHT:
                        bird.alive = False
                        alive_count -= 1
                        continue
                    # There should only ever be 2 pipes to iterate over so this
                    # is a simpler solution than keeping track of the next
                    # incoming pipe pair with some pointer or queue.
                    for pipe in pipes:
                        if pipe.x2 + BIRD_RADIUS >= BIRD_X:
                            collision = check_collision(bird, pipe)
                            if collision:
                                bird.alive = False
                                alive_count -= 1
                                break
                    # No need to calculate the rest if the bird is dead
                    if not bird.alive:
                        continue
                    # Bird can't double jump
                    if bird.steps_to_jump < 1:
                        # Check if bird wants to flap
                        if len(pipes) > 0 and pipes[0].x2 + BIRD_RADIUS >= BIRD_X:
                            if bird.flap(pipes[0].x2, pipes[0].y_top, pipes[0].y_bot):
                                bird.steps_to_jump = BIRD_JUMP_STEPS
                        elif len(pipes) > 1:
                            if bird.flap(pipes[1].x2, pipes[1].y_top, pipes[1].y_bot):
                                bird.steps_to_jump = BIRD_JUMP_STEPS
                    if sim_length < 1:
                        pygame.draw.circle(display_surface, bird.color,
                                           (BIRD_X, bird.y), BIRD_RADIUS)
                    # Bird distance measured in how many pixels it traverses
                    # (imagining the bird as moving and pipes as stationary)
                    bird.distance += FRAME_ANIMATION_WIDTH

            # Display text on screen about the state of the game, only when not
            # pre-training/simulating
            if sim_length < 1:
                gen_surface = gen_font.render(
                    "Generation: " + str(gen), True, (0, 0, 0))
                gen_x = WIN_WIDTH/2 - gen_surface.get_width()/2
                display_surface.blit(gen_surface, (gen_x, BIRD_RADIUS))
                score_color = (0, 0, 0)
                # Make the color of the score text green with new high score.
                if gen_score == best_score and best_score != 0:
                    score_color = (0, 255, 0)
                gen_score_surface = display_font.render(
                    "Generation Score: " + str(gen_score), True, score_color)
                display_surface.blit(
                    gen_score_surface, (BIRD_X, WIN_HEIGHT - 32))
                best_score_surface = display_font.render(
                    "Best Score: " + str(best_score), True, score_color)
                display_surface.blit(
                    best_score_surface, (WIN_WIDTH - 160, WIN_HEIGHT - 32))
                birds_left_surface = display_font.render(
                    "Birds Left: " + str(alive_count), True, (0, 0, 0))
                birds_left_x = WIN_WIDTH / 2 - birds_left_surface.get_width() / 2
                display_surface.blit(
                    birds_left_surface, (birds_left_x, 3*BIRD_RADIUS))
                pygame.display.update()
            # Prints some information about the state
            # of the game to the console.
            print("\r| Generation: " + str(gen) +
                  "| Birds Left: " + str(alive_count) +
                  "| Score: " + str(gen_score) +
                  "| Best Score: " + str(best_score),
                  end="", flush=True)

        # Generation is complete, now for calculations between generations

        # Updates the plots every generation.
        if plot:
            gen_max_dist = max(bird.distance for bird in birds)
            gen_mean_dist = np.mean([bird.distance for bird in birds])
            bird_stats_df.loc[gen - 1] = [gen,
                                          gen_score,
                                          gen_max_dist,
                                          gen_mean_dist]
            score_graph(bird_stats_df, graph1)
            distance_graph(bird_stats_df, graph2)
            fig.tight_layout()
            plt.style.use("fivethirtyeight")
            plt.pause(0.00001)
        # -- Genetic Algorithm Steps -- #
        # The pool of possible parents will be the 20% best performing birds of
        # the previous generation. These Birds will also automatically join the
        # next generation
        parent_birds = sorted(birds, key=lambda x: x.distance,
                              reverse=True)[:int(0.2 * pop_size)]
        # Fill the remaining 80% of the population with offspring from the
        # selected parents
        new_birds = []
        for i in range(pop_size - len(parent_birds)):
            # Sample 2 parents with replacement
            parent1 = natural_selection(parent_birds)
            parent2 = natural_selection(parent_birds)
            # Mix the DNA of the parents to create a child
            new_bird = crossover(parent1, parent2)
            new_bird.y = int(WIN_HEIGHT/2 - BIRD_RADIUS)
            new_bird.color = random_color()
            # Mutate the child with probability p before adding it to the
            # population. Number of mutations and step size are currently
            # fixed at values that I found to work well, but I plan to add
            # these to the parameters that the user can modify from the
            # start menu.
            new_birds.append(mutation(new_bird, p_mut, 5, 0.10))
        # Reset the distance and living status of returning birds
        for bird in parent_birds:
            bird.distance = 0
            bird.alive = True
        birds = new_birds + parent_birds
        gen += 1
        if sim_length > 0:
            sim_length -= 1

