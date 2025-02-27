# Handwritten Digit Prediction - Classification Analysis

This project involves the classification of handwritten digits using the digits dataset. The dataset consists of 8x8 pixel images of digits, with each image stored as an 8x8 array of grayscale values. The goal is to predict the digit represented by each image.

## Project Description

The digits dataset comprises 8x8 pixel images representing handwritten digits. Each image is stored as an 8x8 array of grayscale values, allowing us to visualize the initial four images. Additionally, the dataset's target attribute holds the corresponding digit for each image, indicating what each image signifies.

## Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
