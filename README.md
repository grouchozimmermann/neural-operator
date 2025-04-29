# DeepONet Implementation

A Python implementation of the DeepONet architecture for operator learning. This project includes utilities for training, testing, and visualizing the performance of the model.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Config](#config)
  - [Data Structure](#data-structure)
  - [Setting Up](#setting-up)
  - [Training the Model](#training-the-model)
  - [Evaluating and Visualizing Results](#evaluating-and-visualizing-results)
- [Architecture Overview](#architecture-overview)

## Introduction

The DeepONet architecture is a neural network framework designed to learn nonlinear operators efficiently. This project aims to provide a flexible implementation of DeepONet. Goal of the implementation is to enable me to experiment and analyze the efficiency of the architecture, specifically for a master thesis.

## Installation

### Prerequisites

- Python 3.12.6 or later

### Setting Up the Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository-name.git
   cd your-repository-name

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate` (I think)

3. Install required dependencies:
    ```bash
    pip install -r requirements.txt

## Usage

### Config
All the settings/hyperparameters/parameters for running the model is found in `config.py`. Not sure if this is a smart way of doing it. 

### Data structure
The data used for the model need to adhere to the following pattern: $Dim(data) = (Number of samples, spatial points)$. This will be updated as the model (hopefully) becomes more complex. 

### Setting up
The DeepONet class and its associated layers/components are defined in the `DeepONet.py` and `layers.py` files. These components are orchestrated and executed in main.py. The DeepONet class found `DeepONet.py` is simply a wrapper for the actual model with certain convience functions attached. 
Ensure that you provide your training and testing datasets in a compatible format before running the model. Use the `.set_training_data()` and `.set_testing_data()` methods to specify the data.

### Training the model
To train the DeepONet model, call the .train() method of the DeepONet class.

### Evaluating and Visualizing Results
The DeepONet contains a few utility functions for plotting and visualizing results such as `.plotLoss()`, `.plotPrediction()` and `.plotGradientNorms()`.

## Architecture Overview

The project is organized into the following components:

- **`DeepONet.py`**: This file contains the `DeepONet` class, which is the main implementation of the DeepONet architecture. It includes:
  - Methods for training the model, such as `.train()`.
  - Utility functions like `.plotErrors()` for visualizing errors.
  - Setters for specifying training and testing data.

- **`layers.py`**: Defines the core network architecture used in the DeepONet implementation, including:
  - **FeedForward Network**: A utility module for constructing general-purpose neural network layers.
  - **DeepONet Network**: The actual DeepONet architechture

- **`main.py`**: Serves as the entry point of the project. This script instantiates the `DeepONet` class, sets up the training and testing data, and runs the model.


