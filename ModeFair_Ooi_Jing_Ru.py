import random
from copy import deepcopy
import math

# initialize data
# depot (latitude, longitude)
depot = (4.4184, 114.0932)
# customers (id, latitude, longitude, demand)
customers = [
    (1, 4.3555, 113.9777, 5),
    (2, 4.3976, 114.0049, 8),
    (3, 4.3163, 114.0764, 3),
    (4, 4.3184, 113.9932, 6),
    (5, 4.4024, 113.9896, 5),
    (6, 4.4142, 114.0127, 8),
    (7, 4.4804, 114.0734, 3),
    (8, 4.3818, 114.2034, 6),
    (9, 4.4935, 114.1828, 5),
    (10, 4.4932, 114.1322, 8),
]
# vehicles 
vehicles = [
    {'type': 'A', 'capacity': 25, 'cost': 1.2},
    {'type': 'B', 'capacity': 30, 'cost': 1.5},
]


# calculate vehicle travel distance given latitude & longitude of two locations
def calculate_distance(loc1, loc2):
    loc1x, loc1y = loc1
    loc2x, loc2y = loc2
    return 100 * math.sqrt((loc2y - loc1y)**2 + (loc2x - loc1x)**2)


# initialize population of first generation
def first_generation(customers, vehicle_types, population_size):
    population = []
    for _ in range(population_size):
        chromosome = []  # each chromosome is a solution
        shuffled_customers = customers
        random.shuffle(shuffled_customers)   
        curr_customer = 0

        assigned_customers = []
        
        # loop through each vehicle type in the list, ensure all customers are assigned
        while curr_customer < len(shuffled_customers):
            for vehicle_type in vehicle_types:  
                route = []
                total_demand = 0
                for i in range(curr_customer, len(shuffled_customers)):
                    # ensure total demand per vehicle <= capacity of the vehicle
                    if shuffled_customers[i] in assigned_customers:
                        break
                    if total_demand + shuffled_customers[i][3] <= vehicle_type['capacity']:
                        route.append(shuffled_customers[i])
                        total_demand += shuffled_customers[i][3]
                        assigned_customers.append(shuffled_customers[i])
                    else:
                        curr_customer = i
                        break
                else:
                    # to indicate that all customers are assigned (without breaking the for loop)
                    curr_customer = len(shuffled_customers)  
                if route:
                    # if route is not empty then append the newly generated chromosome
                    chromosome.append({'vehicle': vehicle_type, 'route': route})
                # break if all customers are assigned
                if curr_customer >= len(shuffled_customers):
                    break
        population.append(chromosome)
    return population


# calculate total distance of each chromosome
def calculate_total_distance(route):
    total_distance = 0
    # start from depot
    previous_location = depot
    # calculate & add distance between consecutive customers in the route
    for customer in route:
        customer_location = (customer[1], customer[2])
        total_distance += calculate_distance(previous_location, customer_location)
        previous_location = customer_location
    # end route at depot
    total_distance += calculate_distance(previous_location, depot)
    return total_distance


# calculate fitness value of each chromosome
def calculate_fitness(chromosome):
    total_cost = 0
    for vehicle_route in chromosome:
        distance_travelled = 0
        vehicle = vehicle_route['vehicle']
        route = vehicle_route['route']
        # skip if route is empty
        if not route:
            continue
        # calculate distance of route
        distance_travelled += calculate_total_distance(route)
        total_cost += distance_travelled * vehicle['cost']   # cost in RM
    return total_cost


# crossover two parent chromosomes to generate new two new chromosomes
def crossover(parent1, parent2):
    # ensure the split produces two valid parts of the parent
    split_point = random.randint(1, len(parent1) - 2) 
    # deepcopy to produce clone of first half of each parent
    child1_part1 = deepcopy(parent1[:split_point])
    child2_part1 = deepcopy(parent2[:split_point])
    # ensure there are no duplicate customers
    child1_part2 = [customer for customer in parent2 if customer not in child1_part1]
    child2_part2 = [customer for customer in parent1 if customer not in child2_part1]
    
    child1 = child1_part1 + child1_part2
    child2 = child2_part1 + child2_part2
    
    return child1, child2


# random mutation of chromosome
def mutate(chromosome, mutation_rate = 0.05):
    mutated_chromosome  = deepcopy(chromosome)  # create a clone to avoid modifying the original
    
    for vehicle_route in mutated_chromosome:
        if random.random() < mutation_rate:  # check if mutation should occur based on random value
            route = vehicle_route['route']
            if route and len(route) >= 2:
                # swap the customers at indices i and j selected randomly
                i, j = random.sample(range(len(route)), 2)
                route[i], route[j] = route[j], route[i]
    return mutated_chromosome


# tournament selection
def tournament_selection(population, tournament_size):
    selected = []
    population_size = len(population)
    
    # conduct tournament selection as many times as the original population size
    for _ in range(population_size):
        # reinitialize best index and best fitness after each iteration
        # to ensure equal probability of each chromosome being selected
        best_index = -1
        best_fitness = float('inf')
        
        # tournament selection by comparing fitness values
        for _ in range(tournament_size):
            index = random.randint(0, population_size - 1)  # Randomly select an individual
            fitness = calculate_fitness(population[index])
            
            if fitness < best_fitness:
                best_fitness = fitness
                best_index = index
        
        selected.append(deepcopy(population[best_index]))  # DeepCopy the selected winner chromosome
    
    return selected


# create new population
def create_new_population(population, tournament_size, mutation_rate):
    new_population = []
    selected_population = tournament_selection(population, tournament_size)
    while(len(new_population) < len(population)):
        parent1, parent2 = random.sample(selected_population, 2)
        # carry out crossover and mutation
        child1, child2 = crossover(parent1, parent2)
        mutated_child1 = mutate(child1, mutation_rate)
        mutated_child2 = mutate(child2, mutation_rate)
        # add children to new population, ensure new population size is not bigger than original population size
        new_population.append(mutated_child1)
        if len(new_population) < len(population):
            new_population.append(mutated_child2)
    return new_population


# genetic algorithm function
def genetic_algorithm(customers, vehicles, population_size = 50, generations = 100, tournament_size = 3, mutation_rate = 0.05):
    num_customers = len(customers)
    while True:
        # generate first population
        population = first_generation(customers, vehicles, population_size)
        # create a new population for each generation
        for generation in range(generations):
            new_population = create_new_population(population, tournament_size, mutation_rate)
            population = new_population
        best_chromosome = min(population, key = calculate_fitness)
        # only accept best chromosome solution if solution is valid
        if check_valid_solution(best_chromosome, num_customers):
            break    
    # after getting the wanted newest population
    best_cost = calculate_fitness(best_chromosome)
    return best_chromosome, best_cost


# check if all customers are assigned exactly once
def check_valid_solution(solution, num_customers):
    assigned_customers = set()
    for vehicle_route in solution:
        for customer in vehicle_route['route']:
            if customer[0] in assigned_customers:
                return False
            assigned_customers.add(customer[0])
    return len(assigned_customers) == num_customers

# main function starts here to run genetic algorithm
def main():
    best_solution, best_cost = genetic_algorithm(customers, vehicles)  
    
    # calculate total distance of the solution
    distance_travelled = 0
    for vehicle_route in best_solution:
        vehicle = vehicle_route['vehicle']
        route = vehicle_route['route']
        # skip if route is empty
        if not route:
            continue
        # calculate distance of route
        distance_travelled += calculate_total_distance(route)
    print(f"Total Distance = {distance_travelled} km")
    print(f"Total Cost = RM {best_cost:.2f}\n")

    # print out route of each vehicle
    for index, vehicle_route in enumerate(best_solution):
        print(f"Vehicle {index + 1} (Type {vehicle_route['vehicle']['type']}):")
        if not vehicle_route['route']:
            print("No route assigned")
            continue
        round_trip_distance = calculate_total_distance(vehicle_route['route'])
        route_cost = round_trip_distance * vehicle_route['vehicle']['cost']
        total_demand = sum([customer[3] for customer in vehicle_route['route']])
        print(f"Round Trip Distance: {round_trip_distance:.3f} km, Cost: RM {route_cost:.2f}, Demand: {total_demand}")

        print(f"Depot -> ", end="")
        prev_loc = depot
        for i in range(len(vehicle_route['route'])):
            curr_customer = vehicle_route['route'][i][0]
            curr_loc = (vehicle_route['route'][i][1], vehicle_route['route'][i][2])
            print(f"C{curr_customer} ({calculate_distance(prev_loc, curr_loc):.3f} km) -> ", end="")
            prev_loc = curr_loc
        print(f"Depot ({calculate_distance(prev_loc, depot):.3f} km)\n")
    


if __name__ == "__main__":
    main()


