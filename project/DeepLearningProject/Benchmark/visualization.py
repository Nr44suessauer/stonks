"""
visualization.py - Visualization for benchmark results

This module contains functions for visualizing benchmark results 
with tables and charts.
"""

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np

def visualize_results(results, models, tasks):
    if not results:
        print("No results available for visualization.")
        return
    df = pd.DataFrame(results)
    task_names = list(set([task['name'] for task in tasks]))
    plt.figure(figsize=(12, 6))
    x = np.arange(len(task_names))
    width = 0.35
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        times = []
        for task_name in task_names:
            task_data = model_data[model_data['Task'] == task_name]['Generation Time (s)'].values
            if len(task_data) > 0:
                times.append(task_data[0])
            else:
                times.append(0)
        plt.bar(x + i*width, times, width, label=model)
    plt.xlabel('Task')
    plt.ylabel('Generation Time (s)')
    plt.title('Comparison of Generation Times')
    plt.xticks(x + width/2, task_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(12, 6))
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        tokens_per_sec = []
        for task_name in task_names:
            task_data = model_data[model_data['Task'] == task_name]['Tokens per Second'].values
            if len(task_data) > 0:
                tokens_per_sec.append(task_data[0])
            else:
                tokens_per_sec.append(0)
        plt.bar(x + i*width, tokens_per_sec, width, label=model)
    plt.xlabel('Task')
    plt.ylabel('Tokens per Second')
    plt.title('Comparison of Tokens per Second')
    plt.xticks(x + width/2, task_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    if len(df) > 0:
        summary = df.groupby('Model').agg({
            'Generation Time (s)': 'mean',
            'Tokens Generated': 'mean',
            'Tokens per Second': 'mean'
        }).reset_index()
        print("\n📊 Summary of Performance Metrics:")
        display(summary)
    else:
        print("\n⚠️ No data available for summary.")
    # No output of responses in the notebook anymore