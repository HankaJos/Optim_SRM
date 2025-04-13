import json
import subprocess
import os
import datetime
import argparse
import random
import copy
from data import DataGenerator
from data import RICFileHandler

class Simulation:
    """
    Manages simulation runs and score calculations.
    """

    def __init__(self, script_path,
                 input_name=os.path.join("tmp","new.ric"),
                 output_name=os.path.join("tmp","output_vals.txt"),
                 last_data_name="last_data.txt",
                 path_to_venv_py=os.path.join("openMotor",".venv","Scripts","python.exe")):
        self.script_path = script_path
        # Ensure directory exists
        os.makedirs("tmp", exist_ok=True)
        self.input_name = input_name
        self.output_name = output_name
        self.last_data_name = last_data_name
        self.path_to_venv_py = path_to_venv_py

    def run(self, input_data, limits="Default"):
        """Runs the external simulation script and processes the output."""

        # uprava cesty
        RICFileHandler.create_ric_input(input_data, self.input_name)
        #args = [self.path_to_venv_py, self.script_path, "-o", self.last_data_name, "-h", self.input_name]
        args = [self.path_to_venv_py, self.script_path, "-h", self.input_name]

        result = subprocess.run(args, capture_output=True, text=True)
        print("Simulation Output:", result.stdout)
        print("Simulation Errors:", result.stderr)

        return DataGenerator.create_data(None, self.input_name, self.output_name, limits)

    @staticmethod
    def calculate_score(parametr: float,
                        limit_max: float,
                        limit_min: float,
                        limit_factor: float) -> float:
        """
        Calculates a score based on the given parameter and limits.

        Args:
            parametr (float): The value being evaluated.
            limit_max (float): The upper limit of the acceptable range.
            limit_min (float): The lower limit of the acceptable range.
            limit_factor (float): A scaling factor used in score calculation.

        Returns:
            float: The calculated score. Returns 0 if the parameter is within limits,
                   otherwise returns a scaled difference from the target range.

        Example:
            >>> ScoreCalculator.calculate_score(15, 20, 10, 1.5)
            0  # Since 15 is within the range [10, 20]

            >>> ScoreCalculator.calculate_score(25, 20, 10, 1.5)
            1.5  # Score calculated based on deviation from target
        """
        if limit_max >= parametr >= limit_min:
            return 0  # No penalty if within limits

        target = (limit_min + limit_max) / 2

        if limit_max == limit_min == 0:
            return abs(parametr) * limit_factor * 10000

        return abs(parametr - target) * limit_factor / target

    @staticmethod
    def calculate_total_score(data: dict) -> float:
        """
        Calculates the total score based on given data and limit constraints.

        Args:
            data (dict): A nested dictionary containing information about different 
                         data types and their scoring limits.

        Returns:
            float: The total calculated score.
        """
        total_score = 0

        for key_type, limits in data['info']['limits'].items():
            for key_limits, properties in limits.items():
                if key_limits == 'grains':
                    for grain in data[key_type][key_limits]:
                        for param, limits in properties['properties'].items():
                            score = Simulation.calculate_score(
                                grain['properties'][param],
                                limits['max'], limits['min'], limits['factor']
                            )
                            total_score += score
                else:
                    for param, limits in properties.items():
                        score = Simulation.calculate_score(
                            data[key_type][key_limits][param],
                            limits['max'], limits['min'], limits['factor']
                        )
                        total_score += score

        # data['info']['total_score'] = total_score
        print(total_score, " TOTAL SCORE")
        return total_score


class Population:
    """
    Manages the evolutionary population, including initialization, selection, breeding, and mutation.
    """

    def __init__(self, simulation: Simulation, data, n_populations, time):
        self.simulation = simulation
        self.n = n_populations
        self.time = time
        self.generations = self.initialize_population(data, n_populations)

    def initialize_population(self, data, n) -> dict:
        """
        Initializes a population by creating `n`-populated 1st generation based on input data.

        Args:
            data (dict): The base data structure to copy and modify.
            n (int): The number of individuals in the population.

        Returns:
            dict: A dictionary representing the initialized population.
        """

        generation = {}
        generation[0] = data

        # Ensure output directory exists
        os.makedirs(os.path.join(self.time,f"inicializace_{SEED}"), exist_ok=True)

        # save initial
        file_path = os.path.join(self.time, f"inicializace_{SEED}", f"iter_{0}_score_{data['info']['total_score']}.json")
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)


        for i in range(1, n+1):
            # Deep copy the data template
            cdata = copy.deepcopy(data)

            # Modify `cdata` based on random properties
            for key_type, limits in cdata['info']['limits']['input'].items():
                if key_type == 'grains':
                    for prop, limit in limits['properties'].items():
                        for grain in cdata['input'][key_type]:
                            grain['properties'][prop] = self.random_property_value(limit)
                else:
                    for prop, limit in limits.items():
                        cdata['input'][key_type][prop] = self.random_property_value(limit)

            # Run the simulation and calculate scores
            cdata = self.simulation.run(cdata['input'], cdata['info']['limits'])
            cdata['info']['id'] = i
            cdata['info']['total_score'] = Simulation.calculate_total_score(cdata)
            generation[i] = cdata

            # save it
            file_path = os.path.join(self.time, f"inicializace_{SEED}", f"iter_{i}_score_{cdata['info']['total_score']}.json")
            with open(file_path, "w") as file:
                json.dump(cdata, file, indent=4)

        return generation

    @staticmethod
    def random_property_value(limits: dict):
        if limits['factor'] > 0 and abs(limits['max'] - limits['min']) > 0:
            return round(random.uniform(limits['min'], limits['max']), 4)
        return limits['max']

    def select_top(self, perc_threshold: float):
        """
        Select the top X% members of the population based on total_score.

        Args:
            perc_threshold (float): The percentage of top members to select.

        Returns:
            list: A sorted list of the top selected members.
        """
        # Ensure at least one member is selected
        n_to_select = max(
            1, round(len(self.generations) * perc_threshold / 100))

        sorted_generations = sorted(
            self.generations.values(), key=lambda g: g['info']['total_score'])
        return sorted_generations[:n_to_select]

    def breed(self, selected, mut_prob, mut_prop_prob, mut_strength, n_size_new_pop):
        """
        Creates a new generation through crossover and mutation.

        Args:
            selected (list): List of selected parents.
            mut_prob (float): Probability of mutation occurring.
            mut_prop_prob (float): Probability of each property being mutated.
            mut_strength (float): Strength of mutation.
            n_size_new_pop (int): Desired size of the new population.

        Returns:
            dict: A dictionary representing the new population.
        """

        # Initialize new population with selected individuals
        new_population = {i: individual for i,
                          individual in enumerate(selected)}

        while len(new_population) < n_size_new_pop:
            parents = random.sample(selected, 2)
            new_population[len(new_population)] = self.crossover(
                parents[0], parents[1])

        # Apply mutations
        return self.mutate(new_population, mut_prob, mut_prop_prob, mut_strength)

    @staticmethod
    def crossover(parent1, parent2) -> dict:
        """
        Performs crossover between two parent dictionaries to create an offspring.

        Args:
            parent1 (dict): The first parent.
            parent2 (dict): The second parent.

        Returns:
            dict: The resulting offspring after crossover.
        """

        # Create a deep copy of parent2 to serve as the base for the offspring
        child = copy.deepcopy(parent2)

        # Perform crossover on input data
        for key_type, input_data in parent1['input'].items():
            if key_type == 'grains':
                for grain, pgrain in zip(input_data, child['input'][key_type]):
                    for prop in grain['properties']:
                        if random.random() > 0.5:
                            pgrain['properties'][prop] = grain['properties'][prop]
            else:
                for prop in input_data:
                    if random.random() > 0.5:
                        child['input'][key_type][prop] = input_data[prop]

        # Reset total_score since offspring needs evaluation
        child['info']['total_score'] = None

        return child

    def mutate(self, generation, mut_prob, mut_prop_prob, mut_strength):
        new_generation = {}
        for i, gen in enumerate(generation.values()):
            if mut_prob > random.randint(0, 99):
                for key_type, limits in gen['info']['limits']['input'].items():
                    if key_type == 'grains':
                        for grain in gen['input'][key_type]:
                            for prop, limits in limits['properties'].items():
                                if self.should_mutate(mut_prop_prob, limits):
                                    grain['properties'][prop] = self.apply_mutation(
                                        grain['properties'][prop], limits, mut_strength
                                    )
                    else:
                        for prop, limits in limits.items():
                            if self.should_mutate(mut_prop_prob, limits):
                                gen['input'][key_type][prop] = self.apply_mutation(
                                    gen['input'][key_type][prop], limits, mut_strength
                                )

            gen = self.simulation.run(gen['input'], gen['info']['limits'])
            gen['info']['total_score'] = Simulation.calculate_total_score(gen)
            new_generation[i] = gen
        return new_generation

    @staticmethod
    def should_mutate(mut_prop_prob, limits):
        return limits['factor'] > 0 and abs(limits['max'] - limits['min']) > 0 and mut_prop_prob > random.randint(0, 99)

    @staticmethod
    def apply_mutation(value, limits, strength):
        if not isinstance(limits['min'], (int, float)) or not isinstance(limits['max'], (int, float)):
            raise ValueError("Limits must be numeric values.")
        value += value * strength * random.choice([-1, 1])
        return min(max(value, limits['min']), limits['max'])


def run_gen_alg(population: Population, evo_threshold: float, max_same_results: int):
    '''Function that runs the genetical algorithms and finds optimal solution

    Parameters:
        population (Population): initial population

    Returns:
        Individual (dict) with lowest fitess sore.
    '''

    # Ensure directory exists
    os.makedirs(os.path.join(population.time, f"postup_{SEED}"), exist_ok=True)

    # Find the individual (`p`) with the lowest total_score
    lowest_p = min(population.generations.values(),
                   key=lambda p: p['info']['total_score'])

    # Extract lowest score for filename
    prev_lowest = lowest_p['info']['total_score']

    with open(os.path.join(population.time, f"postup_{SEED}", f"iter_{0}_score_{prev_lowest}.json"), "w") as file:
        json.dump(lowest_p, file, indent=4)

    i, num_of_same_result = 1, 0

    while prev_lowest > evo_threshold and num_of_same_result < max_same_results:
        # selection
        selected = population.select_top(50)
        # crossover and mutation
        new_generation = population.breed(
            selected, mut_prob=20, mut_prop_prob=30, mut_strength=0.1, n_size_new_pop=population.n)
        population.generations = new_generation

        # evaluation
        lowest_score = min(p['info']['total_score']
                           for p in population.generations.values())

        if lowest_score >= prev_lowest:
            num_of_same_result += 1
        else:
            num_of_same_result = 0
            prev_lowest = lowest_score

        best_gen = next(g for g in population.generations.values()
                        if g['info']['total_score'] == lowest_score)

        with open(os.path.join(population.time, f"postup_{SEED}",f"iter_{i}_score_{lowest_score}.json"), "w") as file:
            json.dump(best_gen, file, indent=4)
        i += 1

    # doiterovat slozku postup, jestli neni neco mensi?
    return next(g for g in population.generations.values() if g['info']['total_score'] == lowest_score)

def parse_args():
    parser = argparse.ArgumentParser(description="Genetic Algorithm Runner")
    parser.add_argument("--script-path", type=str, default=os.path.join("openMotor", "main.py"),
                        help="Path to the simulation script.")
    parser.add_argument("--input-file", type=str, default="data1.json",
                        help="Path to the input JSON or RIC file.")
    parser.add_argument("--limits", type=str, default="Default",
                        help="Limits for evaluating motor (Default, RIClike or path).")
    parser.add_argument("--n-populations", type=int, default=10,
                        help="Number of individuals in the population.")
    parser.add_argument("--evo-threshold", type=int, default=8,
                        help="Evolution threshold for early stopping.")
    parser.add_argument("--max-same-results", type=int, default=5,
                        help="Max number of times same result can occur before stopping.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument("--naive-tol", type=int, default=12,
                        help="Tolerance for RIC input file limits.")

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    # Set or generate random seed
    SEED = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    random.seed(SEED)

    timestamp_dir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(timestamp_dir)


    # Initialize simulation and population
    simulation = Simulation(args.script_path)

    if ".json" in args.input_file:
        with open(args.input_file, "r") as file:
            initial_data = json.load(file)
    else: # ric file in input
        initial_data = DataGenerator.create_data(None, args.input_file,
                                                 limits=args.limits, naive_tol=args.naive_tol)
    initial_data = simulation.run(initial_data['input'], initial_data['info']['limits'])
    initial_data['info']['id'] = 0
    initial_data['info']['total_score'] = Simulation.calculate_total_score(initial_data)

    population = Population(simulation, initial_data, args.n_populations, timestamp_dir)

    # Run the genetic algorithm
    best_gen = run_gen_alg(
        population,
        evo_threshold=args.evo_threshold,
        max_same_results=args.max_same_results
    )

    # Save final results
    pathJSON = os.path.join(timestamp_dir, f"result_{SEED}.json")
    with open(pathJSON, "w") as file:
        json.dump(best_gen, file, indent=4)

    pathRIC = os.path.join(timestamp_dir, f'GEN_RESULT_{SEED}.ric')
    RICFileHandler.create_ric_result(pathJSON=pathJSON, pathRIC=pathRIC)