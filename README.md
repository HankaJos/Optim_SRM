# Optim_SRM
A Python-based genetic algorithm system for optimizing motor parameters using OpenMotor simulations. Or simply a program for SRM design optimization.

This project uses evolutionary strategies (selection, crossover, mutation) to find optimal motor designs. The simulation is driven by `.ric` input files and external Python simulation scripts.

---

## 🧠 Features

- Genetic Algorithm with customizable selection, mutation, and evolution thresholds
- Compatible with `.ric` or `.json` inputs
- Automatically creates timestamped directories for each run
- Outputs `.json` and `.ric` files for the best results
- Command-line interface using `argparse`
- Supports seeding for reproducibility

---

## 🛠 Requirements

- Python 3.8+
- openMotor simulation environment and `.venv` path in it should be set up following the openMotor installation instructions

You may install requirements like:

```bash
pip install -r requirements.txt
