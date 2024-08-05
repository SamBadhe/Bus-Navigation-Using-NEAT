import pygame   # Import the Pygame library for creating the game
import os       # Import the os module for operating system functionalities
import math     # Import the math module for mathematical operations
import sys      # Import the sys module for system-specific functions
import neat     # Import the NEAT library for NeuroEvolution of Augmenting Topologies

SCREEN_WIDTH = 1244  # Set the screen width
SCREEN_HEIGHT = 1016  # Set the screen height

# Initialize the Pygame display
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Load the track image
TRACK = pygame.image.load(os.path.join("Assets", "track.png"))

# Create Class Bus
class Bus(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load(os.path.join("Assets", "bus.png"))  # Load the bus image
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(490, 820))  # Set the initial position of the bus
        self.vel_vector = pygame.math.Vector2(0.8, 0)  # Set the initial velocity vector
        self.angle = 0  # Initialize the angle of the bus
        self.rotation_vel = 5  # Set the rotation velocity
        self.direction = 0  # Initialize the direction of the bus
        self.alive = True  # Set the initial state of the bus to alive
        self.radars = []  # Initialize the list for radar data

    def update(self):
        self.radars.clear()  # Clear the radar data list
        self.drive()  # Call the drive method
        self.rotate()  # Call the rotate method
        for radar_angle in (-60, -30, 0, 30, 60):  # Loop through radar angles
            self.radar(radar_angle)  # Call the radar method for each angle
        self.collision()  # Call the collision method
        self.data()  # Call the data method

    def drive(self):
        self.rect.center += self.vel_vector * 6  # Update the position based on the velocity vector

    def collision(self):
        length = 40  # Set the length for collision points
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]  # Calculate the right collision point
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                                int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]  # Calculate the left collision point

        # Die on Collision
        if SCREEN.get_at(collision_point_right) == pygame.Color(2, 105, 31, 255) \
                or SCREEN.get_at(collision_point_left) == pygame.Color(2, 105, 31, 255):
            self.alive = False  # Set the bus state to not alive on collision

        # Draw Collision Points
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_right, 4)  # Draw the right collision point
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_left, 4)  # Draw the left collision point

    def rotate(self):
        if self.direction == 1:  # If the direction is right
            self.angle -= self.rotation_vel  # Decrease the angle
            self.vel_vector.rotate_ip(self.rotation_vel)  # Rotate the velocity vector
        if self.direction == -1:  # If the direction is left
            self.angle += self.rotation_vel  # Increase the angle
            self.vel_vector.rotate_ip(-self.rotation_vel)  # Rotate the velocity vector

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)  # Rotate the bus image
        self.rect = self.image.get_rect(center=self.rect.center)  # Update the rectangle position

    def radar(self, radar_angle):
        length = 0  # Initialize radar length
        x = int(self.rect.center[0])  # Get the x-coordinate of the bus
        y = int(self.rect.center[1])  # Get the y-coordinate of the bus

        while not SCREEN.get_at((x, y)) == pygame.Color(2, 105, 31, 255) and length < 200:  # Continue until an obstacle is detected or maximum length is reached
            length += 1  # Increment length
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)  # Calculate x-coordinate based on radar angle
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)  # Calculate y-coordinate based on radar angle

        # Draw Radar
        pygame.draw.line(SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)  # Draw radar line
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 3)  # Draw radar point

        dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2) + math.pow(self.rect.center[1] - y, 2)))  # Calculate distance to obstacle

        self.radars.append([radar_angle, dist])  # Append radar data to the list

    def data(self):
        input = [0, 0, 0, 0, 0]  # Initialize input list
        for i, radar in enumerate(self.radars):  # Loop through radar data
            input[i] = int(radar[1])  # Set input value to radar distance
        return input  # Return the input list


def remove(index):  # Function to remove a bus from the simulation
    buses.pop(index)  # Remove the bus from the list
    ge.pop(index)  # Remove the genome from the list
    nets.pop(index)  # Remove the neural network from the list


def eval_genomes(genomes, config):  # Function to evaluate the fitness of each genome in a population
    global buses, ge, nets  # Access global variables

    buses = []  # Initialize the list of buses
    ge = []  # Initialize the list of genomes
    nets = []  # Initialize the list of neural networks

    for genome_id, genome in genomes:  # Loop through genomes in the population
        buses.append(pygame.sprite.GroupSingle(Bus()))  # Add a bus to the list
        ge.append(genome)  # Add the current genome to the list
        net = neat.nn.FeedForwardNetwork.create(genome, config)  # Create a feedforward neural network for the current genome
        nets.append(net)  # Add the neural network to the list
        genome.fitness = 0  # Initialize the fitness of the current genome to 0

    run = True  # Set the flag for the main game loop to True
    while run:  # Start the main game loop
        for event in pygame.event.get():  # Iterate over the events in the Pygame event queue
            if event.type == pygame.QUIT:  # Check if the event is quitting the game
                pygame.quit()  # Quit Pygame
                sys.exit()  # Exit the program

        SCREEN.blit(TRACK, (0, 0))  # Draw the track on the screen

        if len(buses) == 0:  # Check if there are no buses remaining
            break  # Exit the game loop if there are no buses

        for i, bus in enumerate(buses):  # Iterate over the buses
            ge[i].fitness += 1  # Increase the fitness of the current bus's genome
            if not bus.sprite.alive:  # Check if the bus is not alive
                remove(i)  # Remove the bus from the simulation

        for i, bus in enumerate(buses):  # Iterate over the buses again
            output = nets[i].activate(bus.sprite.data())  # Get the output from the neural network for the current bus
            if output[0] > 0.7:  # Check if the output for direction is greater than 0.7
                bus.sprite.direction = 1  # Set the direction of the bus to right
            if output[1] > 0.7:  # Check if the output for direction is greater than 0.7
                bus.sprite.direction = -1  # Set the direction of the bus to left
            if output[0] <= 0.7 and output[1] <= 0.7:  # Check if both outputs are less than or equal to 0.7
                bus.sprite.direction = 0  # Set the direction of the bus to straight

        # Update
        for bus in buses:  # Iterate over the buses
            bus.draw(SCREEN)  # Draw the bus on the screen
            bus.update()  # Update the bus's state
        pygame.display.update()  # Update the Pygame display


# Setup NEAT Neural Network
def run(config_path):
    global pop  # Access the global population variable
    config = neat.config.Config(  # Create a NEAT configuration using the specified path
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)  # Create a NEAT population

    pop.add_reporter(neat.StdOutReporter(True))  # Add a standard output reporter for printing information
    stats = neat.StatisticsReporter()  # Create a statistics reporter
    pop.add_reporter(stats)  # Add the statistics reporter to the population

    pop.run(eval_genomes, 50)  # Run the NEAT algorithm for 50 generations

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)  # Get the directory of the current script
    config_path = os.path.join(local_dir, 'config.txt')  # Join the directory and the config file name
    run(config_path)  # Run the NEAT algorithm with the specified configuration
