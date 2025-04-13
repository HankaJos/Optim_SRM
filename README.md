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

### 🔧 Run the optimizer via command line:
```bash
(.venv) python gen.py [OPTIONS]
```

### ⚙️ Available CLI Arguments:

### 🧬 Example Run
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
├── run.py/                    # Main CLI runner
│   ├── Simulation             # Handles simulations in openMotor and data evaluation
│   └── Population             # Performs selection, mutation and breeing
├── data.py/
│   ├── DataGenerator          # Handles .ric → JSON conversions
│   └── RICFileHandler         # Reads/writes .ric files
├── README.md
└── requirements.txt           # (Optional, if using external packages)
```
