import os
import datetime
import argparse
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

class TransformSimulation:
    """
    Applies a parameterized transform to simulation data to match experiment data.
    """

    def __init__(self, experiment_series, sim_series):
        self.experiment = np.array(experiment_series)  # (t, thrust_exp)
        self.sim = np.array(sim_series)                # (t, thrust_sim)

    def apply_transform(self, params):
        """
        Example transform: scale, shift in time, vertical offset
        params = {"scale": float, "time_shift": float, "offset": float}
        """
        t_sim, y_sim = self.sim[:,0], self.sim[:,1]
        # shift time
        t_sim_shifted = t_sim + params.get("time_shift", 0.0)
        # scale amplitude
        y_transformed = y_sim * params.get("scale", 1.0)
        # add offset
        y_transformed += params.get("offset", 0.0)
        return np.column_stack([t_sim_shifted, y_transformed])

    def score(self, params):
        """Compute error metric between transformed sim and experiment."""
        transformed = self.apply_transform(params)
        # interpolate to experimental time grid
        t_exp, y_exp = self.experiment[:,0], self.experiment[:,1]
        y_interp = np.interp(t_exp, transformed[:,0], transformed[:,1], left=0, right=0)
        # RMSE
        rmse = np.sqrt(np.mean((y_exp - y_interp)**2))
        return rmse

class Population:
    def __init__(self, simulation, n_populations):
        self.simulation = simulation
        self.n = n_populations
        self.generations = self.initialize_population(n_populations)

    def initialize_population(self, n):
        gen = {}
        for i in range(n):
            params = {
                "scale": random.uniform(0.5, 2.0),
                "time_shift": random.uniform(-0.5, 0.5),
                "offset": random.uniform(-5, 5),
            }
            score = self.simulation.score(params)
            gen[i] = {"params": params, "score": score}
        return gen

    def select_top(self, perc=50):
        k = max(1, round(len(self.generations) * perc/100))
        return sorted(self.generations.values(), key=lambda g: g["score"])[:k]

    def crossover(self, p1, p2):
        child = copy.deepcopy(p1)
        for k in child["params"]:
            if random.random() < 0.5:
                child["params"][k] = p2["params"][k]
        child["score"] = self.simulation.score(child["params"])
        return child

    def mutate(self, ind, strength=0.1):
        for k,v in ind["params"].items():
            if random.random() < 0.3:  # 30% chance per param
                ind["params"][k] = v + random.uniform(-strength, strength)*v
        ind["score"] = self.simulation.score(ind["params"])
        return ind

    def breed(self, selected, pop_size):
        new_pop = []
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(selected, 2)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            new_pop.append(child)
        self.generations = {i:c for i,c in enumerate(new_pop)}

def run_gen_alg(population, evo_threshold=0.01, max_iter=500):
    best = min(population.generations.values(), key=lambda g: g["score"])
    for i in range(max_iter):
        if best["score"] <= evo_threshold:
            break
        selected = population.select_top(50)
        population.breed(selected, population.n)
        current_best = min(population.generations.values(), key=lambda g: g["score"])
        if current_best["score"] < best["score"]:
            best = current_best
        print(f"Iter {i}, Best Score={best['score']:.4f}, Params={best['params']}")
    return best

def parse_args():
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Transforming Simulation to Match Experiment")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Path to CSV file with experimental data (time,thrust).")
    parser.add_argument("--simulation", type=str, required=True,
                        help="Path to CSV file with simulation data (time,thrust).")
    parser.add_argument("--n-populations", type=int, default=20,
                        help="Number of individuals in the population.")
    parser.add_argument("--max-iter", type=int, default=50,
                        help="Maximum number of GA iterations.")
    parser.add_argument("--evo-threshold", type=float, default=1e-3,
                        help="Threshold score for early stopping.")
    parser.add_argument("--max-same-results", type=int, default=10,
                        help="Stop if no improvement after this many iterations.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument('--debug', action='store_false', help="Random data as example")
    return parser.parse_args()

def load_csv(path):
    """
    Load a CSV file as numpy array with two columns [time, thrust].
    CSV is expected to have no header, just two columns.
    """
    return np.loadtxt(path, delimiter=",", dtype=float)


def plot_results(experiment, sim, best, sim_runner):
    """
    Plot experiment, baseline simulation, and best transformed simulation.

    Args:
        experiment (np.ndarray): Array of [t, thrust] for experiment
        sim (np.ndarray): Array of [t, thrust] for baseline simulation
        best (dict): Best individual from GA with keys "params" and "score"
        sim_runner (TransformSimulation): The simulation runner used
    """
    # Transform sim using best parameters
    transformed = sim_runner.apply_transform(best["params"])

    plt.figure(figsize=(10,6))
    plt.plot(experiment[:,0], experiment[:,1], label="Experiment", linewidth=2)
    plt.plot(sim[:,0], sim[:,1], label="Simulation (raw)", linestyle="--")
    plt.plot(transformed[:,0], transformed[:,1], label=f"Transformed (score={best['score']:.3f})")

    plt.xlabel("Time [s]")
    plt.ylabel("Thrust [N]")
    plt.title("Experiment vs Simulation vs Transformed Simulation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def example():
    # Example fake data
    t = np.linspace(0,5,100)
    experiment = np.column_stack([t, np.sin(t)])
    sim = np.column_stack([t, 0.8*np.sin(t-0.2)+0.1])

    sim_runner = TransformSimulation(experiment, sim)
    population = Population(sim_runner, n_populations=20)

    best = run_gen_alg(population)
    print("Final Best:", best)

    # Plot results
    plot_results(experiment, sim, best, sim_runner)

def main(args):
    # Set random seed
    SEED = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    random.seed(SEED)
    np.random.seed(SEED)

    # Create output directory
    timestamp_dir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(timestamp_dir, exist_ok=True)

    # Load data
    experiment = load_csv(args.experiment)
    sim = load_csv(args.simulation)

    # Setup and run GA
    sim_runner = TransformSimulation(experiment, sim)
    population = Population(sim_runner, n_populations=args.n_populations)
    best = run_gen_alg(population,
                       evo_threshold=args.evo_threshold,
                       max_same_results=args.max_same_results,
                       max_iter=args.max_iter)

    # Save results
    outpath = os.path.join(timestamp_dir, f"best_result_{SEED}.json")
    with open(outpath, "w") as f:
        import json
        json.dump(best, f, indent=4)
    print(f"Best result saved to {outpath}")

    # Plot
    plot_results(experiment, sim, best, sim_runner)

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        example()
    else:
        main(args)
