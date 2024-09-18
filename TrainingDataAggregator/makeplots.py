import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import glob
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('input', help="Input csv file or directory")
parser.add_argument('output', help="Output directory")

args = parser.parse_args()

def makeplot(csvpath, plotpath):
    df = pd.read_csv(csvpath)

    timestamp = df['Time Stamp'].values.reshape(-1, 1)  # reshape to make it a 2D array for sklearn
    d1v = df['Droplet 1 Volume'].values

    # Step 3: Fit linear regression model
    model = LinearRegression()
    model.fit(timestamp, d1v)
    r_squared = model.score(timestamp, d1v)

    # Get the slope and intercept of the regression line
    slope = model.coef_[0]
    intercept = model.intercept_

    equation = f"y = {slope:.2f}x + {intercept:.2f}"
    r_squared_text = f"R² = {r_squared:.2f}"
    # Get the slope and intercept of the regression line
    slope = model.coef_[0]
    intercept = model.intercept_
    # Step 4: Predict y values based on the model
    y_pred = model.predict(timestamp)

    # Step 5: Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(timestamp, d1v, color='blue', label='Data points')

    # Add text annotations for the equation and R-squared value
    plt.text(
        timestamp.min(), d1v.max(), f'{equation}\n{r_squared_text}', 
        color='black', fontsize=12, 
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
    )
    plt.plot(timestamp, y_pred, color='red', linewidth=2, label='Linear regression')
    plt.title('Scatter Plot with Linear Regression')
    plt.xlabel('Time Stamp')
    plt.ylabel('Droplet 1 Volume')
    plt.legend()
    plt.grid(True)
    plt.savefig(plotpath)
    
csvfiles = []
if(os.path.isdir(args.input)):
    csvfiles = glob.glob(os.path.join(args.input, '*.csv'))
else:
    csvfiles = [ args.input ]

if not os.path.exists(args.output):
    os.makedirs(args.output)

print(f'Writing {len(csvfiles)} files to {args.output}')    
for csvpath in csvfiles:
    plotpath = os.path.join(args.output, Path(csvpath).stem + '.png')
    makeplot(csvpath, plotpath)
    print(Path(csvpath).stem)
