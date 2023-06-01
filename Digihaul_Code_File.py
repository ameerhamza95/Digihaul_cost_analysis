# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:35:15 2023

@author: HAMZA
"""

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


def process_dataset(sheet_names, dataset_path):
    """
    Process the dataset and return the resulting DataFrame.

    Args:
        sheet_names (list): List of sheet names to process.
        dataset_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Resulting DataFrame.
    """
    dfs = []  # List to store the resulting DataFrames

    # Iterate over each sheet and assign week number
    for week_num, sheet_name in enumerate(sheet_names, start=15):
        # Read the sheet and store as a DataFrame
        df = pd.read_excel(dataset_path, sheet_name=sheet_name, header=3)

        # Drop rows and columns with NaN values
        df = df.dropna(axis=0, how='all')  # Drop rows with all NaN values
        df = df.dropna(axis=1, how='all')  # Drop columns with all NaN values

        # Drop columns with 'Unnamed' in the name
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Reset the index of the DataFrame
        df = df.reset_index(drop=True)

        # Split the DataFrame into multiple DataFrames based on column name repetition
        num_dataframes = len(df.columns) // 8  # Number of DataFrames to create
        sheet_dfs = []  # List to store the DataFrames for the current sheet

        for i in range(num_dataframes):
            start_col = i * 8
            end_col = (i + 1) * 8
            curr_df = df.iloc[:, start_col:end_col].copy()
            curr_df.columns = curr_df.columns.str.replace(r'\.\d$', '', 
                                                          regex=True)
            sheet_dfs.append(curr_df)

        # Concatenate the DataFrames for the current sheet vertically
        sheet_df = pd.concat(sheet_dfs, axis=0)
        sheet_df['WeekNumber'] = week_num  # Add the week number column
        dfs.append(sheet_df)

    # Concatenate the DataFrames from all sheets vertically
    result_df = pd.concat(dfs, axis=0)

    # Reset the index of the resulting DataFrame
    result_df = result_df.reset_index(drop=True)
    
    # Check for NaN values in the resulting DataFrame
    print("NaN Values:")
    print(result_df.isnull().sum())

    # Check for duplicate values in the resulting DataFrame
    print("\nDuplicate Values:")
    print(result_df.duplicated().sum())

    return result_df


def process_dataset_2(dataset_path):
    """
    Process Dataset 2 and return the resulting DataFrame.

    Args:
        dataset_path (str): Path to the Dataset 2 file.

    Returns:
        pd.DataFrame: Resulting DataFrame.
    """
    # Read Dataset 2
    df = pd.read_excel(dataset_path)

    # Drop rows and columns with NaN values
    df = df.dropna(axis=0, how='all')  # Drop rows with any NaN value
    df = df.dropna(axis=1, how='any')  # Drop columns with any NaN value

    # Sort the rows based on 'WeekNumber' column
    df = df.sort_values('WeekNumber')

    # Reset the index of the DataFrame
    df = df.reset_index(drop=True)

    return df


# Calling process_dataset
sheet_names = ['Apr-21 Week1', 'Apr-21 Week2', 'Apr-21 Week3', 
               'Apr-21 Week4', 'May-21 Week1', 'May-21 Week2', 
               'May-21 Week3']
dataset_path = 'Task 1 - Dataset 1.xlsx'
df_competitor = process_dataset(sheet_names, dataset_path)
print(df_competitor)

# Calling process_dataset_1
dataset_path = 'Task 1 - Dataset 2.xlsx'
df_digihaul = process_dataset_2(dataset_path)
df_digihaul

""" Exploratory Data Analysis (EDA): """
    
# Summary statistics for competitor rates
competitor_rates_stats = df_competitor[['Rate (24h)', 
                                        'Rate (48h)', 
                                        'Rate (72h)', 
                                        'Rate (96h)']].describe()
print("Competitor Rates Summary Statistics:")
print(competitor_rates_stats)

# Summary statistics for Digihaul costs
digihaul_costs_stats = df_digihaul['Average of Cost'].describe()
print("\nDigihaul Costs Summary Statistics:")
print(digihaul_costs_stats)

# Histogram of competitor rates
fig, axes = plt.subplots(figsize=(10, 6), nrows=2, ncols=2)
rates_columns = ['Rate (24h)', 'Rate (48h)', 'Rate (72h)', 
                 'Rate (96h)']

for i, ax in enumerate(axes.flatten()):
    ax.hist(df_competitor[rates_columns[i]], bins=10, 
            edgecolor='black')
    ax.set_xlabel('Rates')
    ax.set_ylabel('Frequency')
    ax.set_title(rates_columns[i])
    ax.xaxis.set_label_coords(0.5, -0.15)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.title.set_position([0.5, 1.05])

plt.suptitle('Distribution of Competitor Rates', 
             y=1.05, fontsize=20)
plt.tight_layout()
plt.show()

# Histogram of Digihaul costs
plt.figure(figsize=(8, 6))
df_digihaul['Average of Cost'].hist(bins=10, 
                                    edgecolor='black')
plt.xlabel('Costs')
plt.ylabel('Frequency')
plt.title('Distribution of Digihaul Costs', 
          fontsize=20)
plt.show()

# Box plot of competitor rates
plt.figure(figsize=(10, 6))
df_competitor[['Rate (24h)', 'Rate (48h)', 
               'Rate (72h)', 'Rate (96h)']].boxplot()
plt.xlabel('Rate Types')
plt.ylabel('Rates')
plt.title('Box Plot of Competitor Rates', fontsize=20)
plt.show()

# Box plot of Digihaul costs
plt.figure(figsize=(8, 6))
df_digihaul['Average of Cost'].plot(kind='box')
plt.ylabel('Costs')
plt.title('Box Plot of Digihaul Costs', fontsize=20)
plt.show()

# Identify outliers in competitor rates using Tukey's fences
q1 = df_competitor[['Rate (24h)', 'Rate (48h)', 
                    'Rate (72h)', 'Rate (96h)']].quantile(0.25)
q3 = df_competitor[['Rate (24h)', 'Rate (48h)', 
                    'Rate (72h)', 'Rate (96h)']].quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
outliers_competitor_rates = \
df_competitor[((df_competitor[['Rate (24h)', 'Rate (48h)', 
                               'Rate (72h)', 'Rate (96h)']] < lower_fence) |
                                          (df_competitor[['Rate (24h)', 
                                                          'Rate (48h)', 
                                                          'Rate (72h)', 
                                                          'Rate (96h)']] > upper_fence)).any(axis=1)]
print("\nOutliers in Competitor Rates:")
print(outliers_competitor_rates)

# Identify outliers in Digihaul costs using Tukey's fences
q1 = df_digihaul['Average of Cost'].quantile(0.25)
q3 = df_digihaul['Average of Cost'].quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
outliers_digihaul_costs = \
df_digihaul[(df_digihaul['Average of Cost'] < lower_fence) | \
            (df_digihaul['Average of Cost'] > upper_fence)]
print("\nOutliers in Digihaul Costs:")
print(outliers_digihaul_costs)

""" Compare Competitor's Rates and Digihaul's Costs: """

# Merge competitor DataFrame and Digihaul DataFrame
merged_df = pd.merge(df_competitor, df_digihaul, \
                     left_on=['Lane (Area)', 'WeekNumber'], 
                     right_on=['LaneArea', 'WeekNumber'], how='inner')

# Calculate the difference between competitor rates and Digihaul costs
merged_df['Cost-Rate_Difference'] = \
merged_df['Average of Cost'] - merged_df['Rate (48h)']

# Analyze the differences
difference_stats = merged_df['Cost-Rate_Difference'].describe()
print("Difference between Digihaul Costs and Competitor Rates:")
print(difference_stats)

# Histogram of rate-cost differences
plt.figure(figsize=(8, 6))
merged_df['Cost-Rate_Difference'].hist(bins=10, edgecolor='black')
plt.xlabel('Cost-Rate Difference')
plt.ylabel('Frequency')
plt.title('Distribution of Cost-Rate Differences', fontsize=20)
plt.show()

# Histogram of rate-cost differences
plt.figure(figsize=(8, 6))
merged_df['Rate (48h)'].hist(bins=10, edgecolor='black', 
                             alpha=0.5, label='Rate (48h)')
merged_df['Average of Cost'].hist(bins=10, edgecolor='black', 
                                  alpha=0.5, label='Average of Cost')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of Competitor Rates and Digihaul Costs', 
          fontsize=20)
plt.legend()
plt.show()

""" Identify Factors Influencing Competitor's Pricing: """

# Scatter plot of competitor rates vs. Total Loads
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_competitor, x='Total Loads', 
                y='Rate (48h)')
plt.xlabel('Total Loads')
plt.ylabel('Rate (48h)')
plt.title('Competitor Rates vs. Total Loads', fontsize=20)
plt.show()

# Correlation analysis of competitor rates and other variables
correlation_matrix = df_competitor[['Total Loads', 'Rate (24h)', 
                                    'Rate (48h)', 'Rate (72h)', 
                                    'Rate (96h)']].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Regression analysis of competitor rates and Total Loads
sns.lmplot(data=df_competitor, x='Total Loads', y='Rate (48h)', 
           scatter_kws={'alpha': 0.5})
plt.xlabel('Total Loads')
plt.ylabel('Rate (48h)')
plt.title('Regression Analysis: Competitor Rates and Total Loads', 
          fontsize=20)
plt.show()

# Analyze trends in competitor rates based on Postcode 1 and Postcode 2
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_competitor, x='Postcode 1', 
            y='Rate (48h)', hue='Postcode 2')
plt.xlabel('Postcode 1')
plt.ylabel('Rate (48h)')
plt.title('Competitor Rates Variation across Regions', 
          fontsize=20)
plt.legend(title='Postcode 2', bbox_to_anchor=(1, 1), ncol=3)  # Modify ncol parameter
plt.show()

# Analyze trends in competitor rates over time (WeekNumber) for different lead times
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_competitor, x='WeekNumber', 
             y='Rate (24h)', label='24hr')
sns.lineplot(data=df_competitor, x='WeekNumber', 
             y='Rate (48h)', label='48hr')
sns.lineplot(data=df_competitor, x='WeekNumber', 
             y='Rate (72h)', label='72hr')
sns.lineplot(data=df_competitor, x='WeekNumber', 
             y='Rate (96h)', label='96hr')
plt.xlabel('WeekNumber')
plt.ylabel('Rate')
plt.title('Competitor Rates Variation over Time', fontsize=20)
plt.legend(title='Lead Time')
plt.ylim(bottom=300, top=600)  # Modify y-axis limits
plt.show()

""" Visualize Trends and Insights: """

# Line chart: Digihaul Costs vs. Competitor Rates (48h) over time
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_competitor, x='WeekNumber', 
             y='Rate (48h)', label='Competitor Rates (48h)')
sns.lineplot(data=df_digihaul, x='WeekNumber', 
             y='Average of Cost', label='Digihaul Costs')
plt.xlabel('WeekNumber')
plt.ylabel('Rate/Cost')
plt.title('Comparison of Digihaul Costs and Competitor Rates over Time', 
          fontsize=20)
plt.legend()
plt.show()

# Bar plot: Mean Competitor Rates (48h) by Postcode 1 and Postcode 2
plt.figure(figsize=(12, 6))
sns.barplot(data=df_competitor, x='Postcode 1', 
            y='Rate (48h)', hue='Postcode 2')
plt.xlabel('Postcode 1')
plt.ylabel('Mean Rate (48h)')
plt.title('Mean Competitor Rates Variation across Regions', 
          fontsize=20)
plt.legend(title='Postcode 2', bbox_to_anchor=(1, 1), ncol=3)
plt.show()

# Aggregate the data by Lane/Area and WeekNumber, taking the mean of Rate_Cost_Difference
pivot_df = merged_df.groupby(['Lane (Area)', 'WeekNumber'])\
['Cost-Rate_Difference'].mean().reset_index()

# Create the pivot table for the heatmap
pivot_table = pivot_df.pivot(index='Lane (Area)', 
                             columns='WeekNumber', 
                             values='Cost-Rate_Difference')

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='coolwarm', 
            annot=True, fmt='.2f', cbar=True)
plt.xlabel('WeekNumber')
plt.ylabel('Lane/Area')
plt.title('Difference between Digihaul Costs and Competitor \nRates (48h) by Lane/Area and WeekNumber', 
          fontsize=20)
plt.show()

# Notable Insights
print("Notable Insights:")
# Identify lanes/areas where Digihaul's costs are significantly different from competitor rates
significant_differences = merged_df[abs(merged_df['Cost-Rate_Difference']) > 50]
print("Lanes/Areas with Significant Differences:")
print(significant_differences[['Lane (Area)', 
                               'WeekNumber', 'Cost-Rate_Difference']])
SD = significant_differences[['Lane (Area)', 
                              'WeekNumber', 'Cost-Rate_Difference']]
SD.to_excel('significant_differences.xlsx')

    
