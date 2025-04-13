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
```

## 🚦 Usage

###  Run the optimizer via command line:
```bash
(.venv) python gen.py [OPTIONS]
```

###  Available CLI Arguments:

| Argument              | Description                                                             | Default             |
|-----------------------|-------------------------------------------------------------------------|---------------------|
| `--script-path`       | Path to the simulation script to run                                    | `openMotor/main.py` |
| `--input-file`        | Input motor data file (`.json` or `.ric`)                               | `data.json`         |
| `--limits`            | Limit type: `'Default'`, `'RIClike'`, or a custom JSON path             | `'Default'`         |
| `--n-populations`     | Number of individuals per generation                                    | `10`                |
| `--evo-threshold`     | Stop if score improves less than this threshold                         | `0`                 |
| `--max-same-results`  | Stop if best score doesn't improve for this many generations            | `5`                 |
| `--seed`              | Random seed (generated if not provided)                                 | `None`              |
| `--naive-tol`         | Tolerance when interpreting `.ric` inputs into parameter limits         | `12`                |


###  Example Run
Run the script with a .ric input file and custom settings:
```bash
(.venv) python gen.py\
  --input-file motor.ric \
  --script-path openMotor/main.py \
  --n-populations 20 \
  --evo-threshold 5 \
  --max-same-results 3 \
  --seed 12345 \
  --limits Default
```
If you want to use my example .json input file (or your manually crafted), you can use that instead:

```bash
(.venv) python gen.py --input-file optimized_design.json
```

## 🗃 Output Structure
Each execution creates a timestamped folder like:

```php-template
2025-02-25_16-03-12/
├── inicializace_<SEED>/
│   ├── iter_0_score_10.23.json
│   ├── iter_1_score_9.56.json
│   └── ...
├── postup_<SEED>/
│   ├── iter_1_score_8.88.json
│   ├── iter_2_score_7.34.json
│   └── ...
├── result_<SEED>.json
└── GEN_RESULT_<SEED>.ric
```

## 📚 Project Structure
```php-template
.
├── gen.py/                    # Main CLI runner
│   ├── Simulation             # Handles running the simulation script, collecting output and output evaluation
│   └── Population             # Performs initialization, selection, mutation and breeing of population
├── data.py/
│   ├── DataGenerator          # Handles .ric → JSON conversions
│   └── RICFileHandler         # Reads/writes .ric files
├── README.md
├── openMotor                  # simulation interface
└── data.json                  # example input data
```

## 📈 Scoring System
Each individual's "fitness" (score) is calculated based on how close it is to given parameter limits:
- Inside bounds = no penalty

- Outside bounds = scaled penalty based on a factor

- The total score = sum of all parameter scores

- The algorithm tries to minimize this total score over generations.

## 👤 Author
Developed by Hana Josifkova.
