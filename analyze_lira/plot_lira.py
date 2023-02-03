from prv_accountant.dpsgd import DPSGDAccountant
from opacus.accountants.rdp import RDPAccountant
import math
import argparse
import csv
import pickle
import os
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot_curve(x, y, xlabel, ylabel, ax, label, color, style, title=None):
    ax.plot([0, 1], [0, 1], 'k-', lw=1.0)
    ax.plot(x, y, lw=2, label=label, color=color, linestyle=style)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set(aspect=1, xscale='log', yscale='log')
    if title is not None:
        ax.title.set_text(title)


def compute_attack_advantage(fpr, tpr):
    return max(tpr - fpr)


def compute_bound(fpr, epsilon, delta):
    return min(math.exp(2 * epsilon) * fpr + (1 + math.exp(epsilon)) * delta,
               1-math.exp(-2*epsilon)*(1-(1+math.exp(epsilon))*delta-fpr))


def compute_tight_prv_bound(fpr, cached = False):
    # params for eps = 1, S=10 and Head with R-50
    total_steps = 398
    sample_rate = 0.5
    noise_multiplier = 29.0625

    print(f"Computing PRV tight bounds with total_steps {total_steps}, sample_rate {sample_rate} and noise_multiplier {noise_multiplier}.")
    # PRV
    if not cached:
        prv_accountant = DPSGDAccountant(
            noise_multiplier=noise_multiplier,
            sampling_probability=sample_rate,
            eps_error=1e-5,
            delta_error=1e-11,
            max_steps=total_steps
        )
    all_bounds = []
    for i, delta in enumerate([1/1000, 1/2000, 1/5000, 1e-4, 1e-5, 1e-6]):
        if cached:
            eps_upper = [0.864171699946296, 0.9426598672691959, 1.0394010035538905,
                     1.1081376646228707, 1.3151737688955718, 1.4977559753135155][i]
        else:
            eps_low, eps_estimate, eps_upper = prv_accountant.compute_epsilon(num_steps=total_steps, delta=delta)

        print(f"δ={delta} results in ϵ={eps_upper}")

        bound = []
        for i in fpr:
            bound.append(compute_bound(i, eps_upper, delta))
        bound = np.hstack(bound)
        all_bounds.append(bound)

    # get tighest bound from all computed PRV bounds
    tighest_bound = np.amin(all_bounds, axis=0)

    return tighest_bound


def compute_rdp_bound(fpr):
    epsilon = 1
    delta = 1/1000
    bound = []
    for i in fpr:
        bound.append(compute_bound(i, epsilon, delta))
    bound = np.hstack(bound)
    return bound


def compute_rd_bound_multiple_delta(fpr):
    # params for eps = 1, S=10 and Head with R-50
    total_steps = 398
    sample_rate = 0.5
    noise_multiplier = 29.0625

    print(
        f"Computing RDP bounds with multiple delta and total_steps {total_steps}, sample_rate {sample_rate} and noise_multiplier {noise_multiplier}.")

    rdp_accountant = RDPAccountant()
    rdp_accountant.history.append((noise_multiplier, sample_rate, total_steps))
    all_bounds = []
    for delta in [1/1000, 1/2000, 1/5000, 1e-4, 1e-5, 1e-6]:
        eps, alpha = rdp_accountant.get_privacy_spent(delta=delta)
        print(f"δ={delta} results in ϵ={eps}")
        bound = []
        for i in fpr:
            bound.append(compute_bound(i, eps, delta))
        bound = np.hstack(bound)
        all_bounds.append(bound)

    # get tighest bound from all computed PRV bounds
    tighest_bound = np.amin(all_bounds, axis=0)

    return tighest_bound


def plot_roc_curve(y_true_list, y_score_list, legend_list, color_list, style_list, fpr_points, save_path, title,
                   shot_list, epsilon_list, flip_legend=False, plot_bound=False, plot_rdp_bound=False):
    assert len(y_true_list) == len(y_score_list)
    assert len(legend_list) == len(y_true_list)
    tpr_at_fpr_results = []
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for y_true, y_score, legend, color, style, shot, epsilon in zip(y_true_list, y_score_list, legend_list,
                                                                    color_list, style_list, shot_list, epsilon_list):
        # get the AUC
        auc = roc_auc_score(y_true=y_true, y_score=y_score)
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
        attack_advantage = compute_attack_advantage(fpr, tpr)
        tpr_at_fpr = {
            'legend': legend,
            'values': [],
            'auc': auc,
            'shot': shot,
            'epsilon': epsilon,
            'adv': attack_advantage
        }
        for fpr_point in fpr_points:
            tpr_at_fpr['values'].append(np.interp(x=fpr_point, xp=fpr, fp=tpr))
        tpr_at_fpr_results.append(tpr_at_fpr)

        if plot_bound and shot == '10' and legend == "Head: S=10":  # should only happen in the case of epsilon == '1' and shot == '10' and Head
            if plot_rdp_bound:
                rdp_bound = compute_rdp_bound(fpr)
                plot_curve(x=fpr, y=rdp_bound, xlabel='FPR', ylabel='TPR', ax=ax, color='C4', style=':',
                           label=r"UB (RDP $δ=$1e-3): $S=10$", title=None)
                rdp_tighest_bound = compute_rd_bound_multiple_delta(fpr)
                plot_curve(x=fpr, y=rdp_tighest_bound, xlabel='FPR', ylabel='TPR', ax=ax, color='blue', style=':',
                           label=r"UB (RDP multiple $δ$): $S=10$", title=None)

            tighest_bound = compute_tight_prv_bound(fpr, cached=True)
            plot_curve(x=fpr, y=tighest_bound, xlabel='FPR', ylabel='TPR', ax=ax, color='red', style=':',
                       label=r"UB (PRV multiple $δ$): $S=10$" if plot_rdp_bound else r"Upper Bound: $S=10$", title=None)

        if shot == "10" or not plot_rdp_bound:
            # plot the roc curve
            plot_curve(x=fpr, y=tpr, xlabel='FPR', ylabel='TPR', ax=ax, label='{0:}, TPR={1:1.3f}'.format(
                legend, tpr_at_fpr['values'][0]), color=color, style=style, title=title)

    if flip_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[::-1], labels=labels[::-1], loc='lower right', fontsize=9.3)
    else:
        plt.legend(loc='lower right', fontsize=9.3)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    return tpr_at_fpr_results


class CsvWriter:
    def __init__(self, file_path, header):
        self.file = open(file_path, 'w', encoding='UTF8', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(header)

    def __del__(self):
        self.file.close()

    def write_row(self, row):
        self.writer.writerow(row)


OUT_DIR = '.'
fpr_points = [1e-3, 1e-2, 1e-1]

colors1 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

colors2 = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

line_styles = [
    '-', '--'
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None, help="Path to in_indices, stat, and output score files.")
    args = parser.parse_args()

    csv_writer = CsvWriter(
        file_path=os.path.join(OUT_DIR, 'mia_tpr_fpr_results.csv'),
        header=[
            'epsilon', 'shot', 	'0.1-Head',	'0.1-FiLM', '1-Head', '1-FiLM', '10-Head', '10-FiLM', 'AUC-Head',
            'AUC-FiLM', 'Adv-Head', 'Adv-FiLM'
        ]
    )

    # plot by epsilon
    for shot in ['10', '25', '50', '100']:
        title = "S = {}".format(shot)
        y_true_list = []
        y_score_list = []
        legend_list = []
        color_list = ['#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e', '#2ca02c', '#2ca02c', '#d62728',
                      '#d62728', '#9467bd', '#9467bd']
        style_list = ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--']
        epsilon_list = ['1', '1', '2', '2', '4', '4', '8', '8', 'inf', 'inf']
        shot_list = [shot] * len(epsilon_list)
        for epsilon in ['1', '2', '4', '8', 'inf']:
            for config in ['none', 'film']:
                with open(os.path.join(args.data_path, 'scores_{}_{}_{}.pkl'.format(config, shot, epsilon)), "rb") as f:
                    result = pickle.load(f)
                    y_true_list.append(result['y_true'])
                    y_score_list.append(result['scores'])
                legend_list.append('{}:$ϵ$={}'.format('Head' if config == 'none' else 'FiLM',
                                                       '$\infty$' if epsilon == 'inf' else epsilon))

        tpr_at_fpr_results = plot_roc_curve(y_true_list, y_score_list, legend_list, color_list, style_list,
                                            fpr_points, os.path.join(OUT_DIR, 'roc_shot_{}.pdf'.format(shot)),
                                            title, shot_list, epsilon_list, flip_legend=True)

        for i, legend in enumerate(legend_list):
            print('{} - fpr, tpr @ {}'.format(title, legend))
            print('auc = {0:1.3f}'.format(tpr_at_fpr_results[i]['auc']))
            for fpr, tpr in zip(fpr_points, tpr_at_fpr_results[i]['values']):
                print('{0:1.3f}, {1:1.3f}'.format(fpr, tpr))

    # plot by shot
    for epsilon in ['1', '2', '4', '8', 'inf']:
        title = '$ϵ$={}'.format('$\infty$' if epsilon == 'inf' else epsilon)
        y_true_list = []
        y_score_list = []
        legend_list = []
        color_list = ['#8c564b', '#8c564b', '#808080', '#808080', 'orange', 'orange', '#0000FF', '#0000FF']
        style_list = ['-', '--', '-', '--', '-', '--', '-', '--']
        shot_list = ['10', '10', '25', '25', '50', '50', '100', '100']
        epsilon_list = [epsilon] * len(shot_list)
        for shot in ['10', '25', '50', '100']:
            for config in ['none', 'film']:
                with open(os.path.join(args.data_path, 'scores_{}_{}_{}.pkl'.format(config, shot, epsilon)), "rb") as f:
                    result = pickle.load(f)
                    y_true_list.append(result['y_true'])
                    y_score_list.append(result['scores'])
                legend_list.append('{}: S={}'.format('Head' if config == 'none' else 'FiLM', shot))

        if epsilon == '1':
            tpr_at_fpr_results = plot_roc_curve(y_true_list, y_score_list, legend_list, color_list, style_list,
                                                fpr_points, os.path.join(OUT_DIR, 'roc_eps_{}_both_bounds.pdf'.format(epsilon)),
                                                title, shot_list, epsilon_list, flip_legend=False,
                                                plot_bound=True, plot_rdp_bound=True)

        tpr_at_fpr_results = plot_roc_curve(y_true_list, y_score_list, legend_list, color_list, style_list,
                                            fpr_points, os.path.join(OUT_DIR, 'roc_eps_{}.pdf'.format(epsilon)),
                                            title, shot_list, epsilon_list, flip_legend=False,
                                            plot_bound=True if epsilon == '1' else False)

        for i, legend in enumerate(legend_list):
            print('{} - fpr, tpr @ {}'.format(title, legend))
            print('auc = {0:1.3f}'.format(tpr_at_fpr_results[i]['auc']))
            print('adv = {0:1.3f}'.format(tpr_at_fpr_results[i]['adv']))
            for fpr, tpr in zip(fpr_points, tpr_at_fpr_results[i]['values']):
                print('{0:1.3f}, {1:1.3f}'.format(fpr, tpr))

        for i, legend in enumerate(legend_list):
            if i % 2 == 1:
                csv_writer.write_row(
                    [
                        "{0:}".format(tpr_at_fpr_results[i]['epsilon']),
                        "{0:}".format(tpr_at_fpr_results[i]['shot']),
                        "{0:2.2f}".format(tpr_at_fpr_results[i - 1]['values'][0] * 100.0),
                        "{0:2.2f}".format(tpr_at_fpr_results[i]['values'][0] * 100.0),
                        "{0:2.2f}".format(tpr_at_fpr_results[i - 1]['values'][1] * 100.0),
                        "{0:2.2f}".format(tpr_at_fpr_results[i]['values'][1] * 100.0),
                        "{0:2.2f}".format(tpr_at_fpr_results[i - 1]['values'][2] * 100.0),
                        "{0:2.2f}".format(tpr_at_fpr_results[i]['values'][2] * 100.0),
                        "{0:1.3f}".format(tpr_at_fpr_results[i - 1]['auc']),
                        "{0:1.3f}".format(tpr_at_fpr_results[i]['auc']),
                        "{0:1.3f}".format(tpr_at_fpr_results[i - 1]['adv']),
                        "{0:1.3f}".format(tpr_at_fpr_results[i]['adv']),
                    ]
                )


if __name__ == '__main__':
    main()
