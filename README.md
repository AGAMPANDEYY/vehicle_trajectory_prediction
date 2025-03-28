# vehicle_trajectory_prediction
This project is part of CEN 300 course subject at the Civil Engineering department IIT Roorkee for Pre-Final year students.


![pipeline](https://github.com/AGAMPANDEYY/vehicle_trajectory_prediction/blob/main/media/traj-pred.png)

# Vehicle Trajectory Prediction

## Introduction  
This repository contains a project focused on predicting vehicle trajectories using advanced machine learning techniques. The goal is to accurately forecast the future path of vehicles based on historical data, which can be crucial for autonomous driving systems, traffic management, and safety applications.

## Table of Contents  
- [Introduction](#introduction)  
- [Getting Started](#getting-started)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Guiding Code Snippets](#guiding-code-snippets)  
- [Contributing](#contributing)  
- [License](#license)  

## Getting Started  
To begin working with this project, follow these steps:

### Clone the Repository  
Clone this repository to your local machine using Git.

```bash
git clone https://github.com/AGAMPANDEYY/vehicle_trajectory_prediction.git
```

### Navigate to the Project Directory:

``` bash
cd vehicle_trajectory_prediction
```

### Project Structure
The project is organized as follows:

data: Contains datasets used for training and testing.

models: Includes the machine learning models implemented for trajectory prediction.

utils: Utility functions for data preprocessing and visualization.

main.py: The entry point for running the prediction pipeline.

Installation
To install the required dependencies, run:

```bash
pip install -r requirements.txt
```
Ensure you have Python and pip installed on your system.

### Usage
Running the Prediction Model
Prepare Data: Ensure your dataset is in the data directory.

Train the Model: Run the training script.

``` bash
python main.py --mode train
```

Make Predictions: Use the trained model to predict trajectories.

```bash
python main.py --mode predict
```

### Example Use Case
For a more detailed example, consider the following Python snippet:

``` python
import numpy as np
from models import TrajectoryPredictor

# Load data
data = np.load('data/vehicle_trajectories.npy')

# Initialize the predictor
predictor = TrajectoryPredictor()

# Train the model
predictor.train(data)

# Make predictions
predictions = predictor.predict(data)
``` 
### Guiding Code Snippets
Data Preprocessing
``` python
import pandas as pd

# Load data
df = pd.read_csv('data/raw_data.csv')

# Clean and preprocess data
df = df.dropna()  # Remove missing values
df = df[['x', 'y', 'speed']]  # Select relevant columns

# Save preprocessed data
df.to_csv('data/preprocessed_data.csv', index=False)
Model Training
python
from sklearn.model_selection import train_test_split
from models import TrajectoryPredictor

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Initialize and train the model
predictor = TrajectoryPredictor()
predictor.train(train_data)

# Evaluate the model
accuracy = predictor.evaluate(test_data)
print(f"Model Accuracy: {accuracy}")
```

### Contributing

We welcome contributions from the community! To contribute, please follow these steps:

#### Steps to Contribute  

1. **Fork the Repository**  
   Click the **Fork** button on the top right of this repository to create your own copy.  

2. **Clone Your Fork**  
   Clone your forked repository to your local machine:  
   ```bash
   git clone https://github.com/your-username/repository-name.git
   cd repository-name
   ```
3. **Create a New Branch**
   Create a branch for your feature or fix:

``` bash
git checkout -b feature-branch-name
```
4. **Commit, Push and open a PR!**

### License
This project is licensed under the MIT License. See LICENSE for details.

Feel free to adjust this template based on specific details from your repository. Ensure that you include accurate information about the project structure, dependencies, and usage guidelines.
