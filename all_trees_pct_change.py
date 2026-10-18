import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
df = pd.read_csv('/home/loglegion/Documents/BSE508/yield_model_results/with_2024_data/summary/all_tree_metrics.csv')

# Group the data by 'Trees' and calculate the average R2 value for each group
avg_r2_by_trees = df.groupby('trees')['r2'].mean().reset_index()

# Calculate the standard error of the R2 values for each group
std_err_r2_by_trees = df.groupby('trees')['r2'].std() / np.sqrt(len(df.groupby('trees')['r2']))

# Calculate the percent change in R2 values
avg_r2_by_trees['R2_Pct_Change'] = (avg_r2_by_trees['r2'] - avg_r2_by_trees['r2'].shift(1)) / avg_r2_by_trees['r2'].shift(1) * 100

# Create a scatter plot of R2_Pct_Change vs Trees with error bars
plt.errorbar(avg_r2_by_trees['R2_Pct_Change'], avg_r2_by_trees['trees'], yerr=std_err_r2_by_trees, fmt='o')

# Add a title and labels
plt.title('Percent Change in R² vs Trees')
plt.xlabel('R² Percent Change')
plt.ylabel('Trees')

# Show the plot
plt.show()
plt.savefig()
