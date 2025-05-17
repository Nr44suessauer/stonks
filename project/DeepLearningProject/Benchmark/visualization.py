"""
visualization.py - Visualisierung für Benchmark-Ergebnisse

Dieses Modul enthält Funktionen zur Visualisierung von Benchmark-Ergebnissen 
mit Tabellen und Grafiken.
"""

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np

def visualize_results(results, models, tasks):
    if not results:
        print("Keine Ergebnisse zum Visualisieren vorhanden.")
        return
    df = pd.DataFrame(results)
    task_names = list(set([task['name'] for task in tasks]))
    plt.figure(figsize=(12, 6))
    x = np.arange(len(task_names))
    width = 0.35
    for i, model in enumerate(models):
        model_data = df[df['Modell'] == model]
        times = []
        for task_name in task_names:
            task_data = model_data[model_data['Aufgabe'] == task_name]['Generierungszeit (s)'].values
            if len(task_data) > 0:
                times.append(task_data[0])
            else:
                times.append(0)
        plt.bar(x + i*width, times, width, label=model)
    plt.xlabel('Aufgabe')
    plt.ylabel('Generierungszeit (s)')
    plt.title('Vergleich der Generierungszeiten')
    plt.xticks(x + width/2, task_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(12, 6))
    for i, model in enumerate(models):
        model_data = df[df['Modell'] == model]
        tokens_per_sec = []
        for task_name in task_names:
            task_data = model_data[model_data['Aufgabe'] == task_name]['Tokens pro Sekunde'].values
            if len(task_data) > 0:
                tokens_per_sec.append(task_data[0])
            else:
                tokens_per_sec.append(0)
        plt.bar(x + i*width, tokens_per_sec, width, label=model)
    plt.xlabel('Aufgabe')
    plt.ylabel('Tokens pro Sekunde')
    plt.title('Vergleich der Tokens pro Sekunde')
    plt.xticks(x + width/2, task_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    if len(df) > 0:
        summary = df.groupby('Modell').agg({
            'Generierungszeit (s)': 'mean',
            'Tokens generiert': 'mean',
            'Tokens pro Sekunde': 'mean'
        }).reset_index()
        print("\n📊 Zusammenfassung der Performance-Metriken:")
        display(summary)
    else:
        print("\n⚠️ Keine Daten für die Zusammenfassung verfügbar.")
    # Keine Ausgabe der Antworten im Notebook mehr