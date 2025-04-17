import os
import matplotlib.pyplot as plt
import numpy as np
import csv

from utils.performance_metrics import calculate_costs

def load_results_from_csv(csv_path):
    results = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in results:
                    results[key] = []
                try:
                    results[key].append(float(value))
                except:
                    results[key].append(value)
    return results

def plot_metric(metric, values1, values2, model_names, save_path):
    fig, ax = plt.subplots(figsize=(12, 8))

    max_len = max(len(values1), len(values2))
    values1 += [np.nan] * (max_len - len(values1))
    values2 += [np.nan] * (max_len - len(values2))
    epochs = np.arange(max_len)

    # Color and style customization
    color1 = '#FFD700'  # Gold for DQN
    color2 = '#00CED1'  # Dark Turquoise for MAPPO

    # Plotting
    ax.plot(epochs, values1,  color=color1, linewidth=3, label=model_names[0])
    ax.plot(epochs, values2,  color=color2, linewidth=3, label=model_names[1])

    ax.set_title(f'{metric.capitalize()} Comparison', fontsize=20)
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel(metric.capitalize(), fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)

    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'comparison_{metric}.png'), dpi=300)
    plt.close()

def plot_comparison_metrics(results1, results2, model_names, save_path='plots'):
    os.makedirs(save_path, exist_ok=True)

    all_keys = set(results1.keys()).intersection(set(results2.keys()))
    skip_keys = {'params', 'agent_type', 'epoch', 'n_agents', 'estimated_cost'}

    for metric in all_keys:
        if metric in skip_keys:
            continue

        try:
            values1 = list(map(float, results1[metric]))
            values2 = list(map(float, results2[metric]))
        except ValueError:
            continue

        plot_metric(metric, values1, values2, model_names, save_path)

def plot_selected_metrics(results1, results2, model_names, metrics, save_path='plots'):
    os.makedirs(save_path, exist_ok=True)

    for metric in metrics:
        if metric in results1 and metric in results2:
            try:
                values1 = list(map(float, results1[metric]))
                values2 = list(map(float, results2[metric]))
                plot_metric(metric, values1, values2, model_names, save_path)
            except ValueError:
                continue

def plot_training_costs(save_path='plots'):
    os.makedirs(save_path, exist_ok=True)
    training_csv = "../synthetic_dataset.csv"
    data = load_results_from_csv(training_csv)
    comp, trans, delay = calculate_costs(data)

    dummy_label = ["Training", "Training"]
    plot_metric("Computation Cost", comp, comp, dummy_label, save_path)
    plot_metric("Transmission Cost", trans, trans, dummy_label, save_path)
    plot_metric("Delay Penalty", delay, delay, dummy_label, save_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare two model CSV logs and visualize training data costs")
    parser.add_argument('--model1', type=str, help='Path to first model CSV')
    parser.add_argument('--model2', type=str, help='Path to second model CSV')
    parser.add_argument('--name1', type=str, default='Model1', help='Label for first model')
    parser.add_argument('--name2', type=str, default='Model2', help='Label for second model')
    args = parser.parse_args()

    results1 = load_results_from_csv(args.model1)
    results2 = load_results_from_csv(args.model2)

    plot_comparison_metrics(results1, results2, [args.name1, args.name2])

    performance_metrics = [
        'reward', 'inference_time', 'training_time',
        'loss', 'steps_per_episode'
    ]
    plot_selected_metrics(results1, results2, [args.name1, args.name2], performance_metrics)

    plot_training_costs()

if __name__ == '__main__':
    main()
