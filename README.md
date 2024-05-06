# Video Machine Learning Framework for Spatiotemporal Analysis in Urban Complex System


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

This GitHub repository presents a comprehensive video machine learning framework designed for spatiotemporal analysis in urban complex systems. By decomposing the space-time cube into a video-like structure, the framework leverages state-of-the-art machine learning models such as ConvLSTM, PredRNN, PredRNN-V2, and E3D-LSTM for effective spatiotemporal analysis. Compared to traditional regression-based approaches, which struggle with heterogeneous geospatial data, this framework offers a novel analytical tool tailored specifically for complex urban environments. The repository contains code and resources for leveraging machine learning techniques on video data to study urban complex systems.

<h3>Data:</h3>
Real-life data collected from AoT and remote sensing data product & Simulated Dataset

<h3>Models:</h3>
ConvLSTM, PredRNN, PredRNN-v2, E3D-LSTM

## Installation

To install and set up the project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/cybergis/video-ML-urban-complex-sys.git

## Usage

To use the project, follow these steps:

1. Prepare your video data and ensure it is properly formatted and annotated if required.
2. Use the provided scripts and modules to preprocess the data, train machine learning models, and analyze the results.
3. Customize the project as needed for specific applications or use cases, adjusting parameters, models, and visualization techniques accordingly.

### Start Script

- The `start.bash` file serves as a convenient entry point for running experiments.
- Users can modify the `start.bash` file to specify various experiment configurations.
- It provides a computing cluster-friendly way to execute multiple experiments and compare output results efficiently.

### Experiment Configuration

- The `run.py` script allows users to run experiments with customizable configurations.
- Users can specify experiment parameters such as dataset name, model name, learning rate, and experiment name using command-line flags.
- For example, to run PredRNN with a learning rate of 5e-3 on the real dataset and name the experiment "vis", users can use the following command: `python run.py --dataset-name real --model-name predrnn -lr 5e-3 --experiment-name vis`

