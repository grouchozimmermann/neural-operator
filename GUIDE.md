# Running My Neural Network Models

Hey there! 👋 This is a quick guide to help you navigate and run the code for my thesis project. It's a bit rough around the edges, but this should help you get started.

## How to Run Each Model

To run any of the models/cases, look for the corresponding `main_*.py` file. For example:

- `main_pendulum.py` - For running the pendulum test case
- `main_FOWT.py` - For running the final FOWT case
- etc.

Each main file handles:
- Loading the appropriate data
- Setting up the neural network model
- Training the model
- Displaying performance graphs

Just run it with Python:

```bash
python main_pendulum.py
```

## Project Structure

### Model Implementation
- **Individual model files**: Each model has its own file named after the model (e.g., `EDeepONet.py`, `CoupledDeepONet.py`)
- **`layers.py`**: Contains the smaller, reusable components that make up the models

### Data
- **`data/`**: Contains the datasets
- **`datageneration/`**: Contains scripts for generating/preprocessing data

### Configuration
- **`config.py`**: Contains settings for all models (learning rates, batch sizes, etc.)

## ⚠️ Important Notes ⚠️

I'll be honest - things got a bit messy towards the end of the project. Here are some things to watch out for:

1. **Hardcoded values**: Some parameters might be hardcoded in the model files rather than using the config file. If you're getting unexpected results, check the implementation code.

2. **Learning rate schedulers**: Some models might have custom learning rate schedulers implemented directly in the code rather than in the config.

3. **Gradient clipping**: Similar to above, some gradient clipping values might be hardcoded.

If you want the absolute best results, I recommend skimming through the implementation code to catch any hardcoded variables or special cases I might have added during late-night coding sessions.

## Example Workflow

1. Check `config.py` to see the default settings
2. Run the model: `python main_pendulum.py` 
3. If results seem off, look at the actual implementation in `EDeepONet.py` for any hardcoded tricks
4. Adjust parameters as needed

Good luck! And sorry about the messiness.
