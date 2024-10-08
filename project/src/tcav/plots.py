import matplotlib.pyplot as plt
import pandas as pd
from captum.concept._utils.common import concepts_to_str
import numpy as np
from matplotlib.container import BarContainer
from scipy.stats import ttest_ind


def plot_tcav_sets(experimental_sets, tcav_scores, max_sets=5, score_type="sign_count", plt_name="plot"):
    max_sets = np.min([len(experimental_sets), max_sets])
    experimental_sets = experimental_sets[:max_sets]
    fig, ax = plt.subplots(1, len(experimental_sets), figsize=(10, 3))

    for exp_idx, exp in enumerate(experimental_sets):
        key = concepts_to_str(exp)
        tcav_scores_exp = tcav_scores[key]
        concepts = [concept.name for concept in exp]
        layers = []
        scores = []
        for layer, score in tcav_scores_exp.items():
            layers.append(layer)
            scores.append([np.float(val) for val in score[score_type]])
        df = pd.DataFrame(data=scores, index=layers, columns=concepts)
        _ax = (ax[exp_idx] if len(experimental_sets) > 1 else ax)
        df.plot.bar(ax=_ax, rot=90, title=f"Set {exp_idx + 1}")
        _ax.set_ylim(bottom=0.0, top=1.0)

    plt.tight_layout()
    plt.savefig(f"{plt_name}.svg")
    plt.close()


def assemble_scores(tcav_scores, experimental_sets, idx, score_layer, score_type):
    score_list = []
    for concepts in experimental_sets:
        score_list.append(tcav_scores["-".join([str(c.id) for c in concepts])][score_layer][score_type][idx])
    score_list = np.array(score_list)
    return score_list


def plot_tcav_scores(all_experimental_sets, all_tcav_scores, score_type="sign_count", alpha=0.05, only_significant=True,
                     with_error=True, plt_name="plot", file_type="svg"):
    tcav_score0 = next(iter(all_tcav_scores[0].values()))
    layers = list(tcav_score0.keys())
    scores_mean = {layer: [] for layer in layers}
    scores_std = {layer: [] for layer in layers}
    stars = {layer: [] for layer in layers}
    concept_short_names = []
    for experimental_sets, tcav_scores in zip(all_experimental_sets, all_tcav_scores):
        concept_short_codes = [int(str(concept.id)[:5]) for concept in experimental_sets[0]]
        concept_short_names_next = [concept.name[:-4] for concept in experimental_sets[0]]
        concept_short_names.append(concept_short_names_next[0])

        for layer in layers:
            pos_scores = assemble_scores(tcav_scores, experimental_sets, 0, layer, score_type)
            neg_scores = assemble_scores(tcav_scores, experimental_sets, 1, layer, score_type)
            # t-test
            _, pval = ttest_ind(pos_scores, neg_scores)
            # Bonferroni correction
            m = 2
            alpha = alpha / m
            # Get non-significant and significant scores
            not_significant = np.mean(pos_scores) < np.mean(neg_scores) or pval >= alpha
            stars[layer].append(not_significant)
            if not_significant and only_significant:
                scores_mean[layer].append(0.0)
                scores_std[layer].append(0.0)
            else:
                scores_mean[layer].append(np.mean(pos_scores))
                scores_std[layer].append(np.std(pos_scores))
    scores_mean_df = pd.DataFrame(data=scores_mean, index=concept_short_names)
    scores_std_df = pd.DataFrame(data=scores_std, index=concept_short_names)

    fig = plt.figure(figsize=(10, 3))
    ax = plt.subplot()
    if with_error:
        scores_mean_df.plot.bar(ax=ax, rot=0, yerr=scores_std_df)
    else:
        scores_mean_df.plot.bar(ax=ax, rot=0)

    containers = []
    for container in ax.containers:
        if isinstance(container, BarContainer):
            containers.append(container)
    for container, star in zip(containers, stars.values()):
        labels = []
        for s_idx, s in enumerate(star):
            if s:
                labels.append("*")
                current_color = container.patches[s_idx].get_facecolor()
                new_color = current_color
                # new_color = [c for c in current_color[:-1]]
                # new_color.append(0.25)
                container.patches[s_idx].set_facecolor(new_color)
            else:
                labels.append("")
        ax.bar_label(container, labels=labels, size=18, color=container.patches[0].get_facecolor())
    ax.set_ylim(bottom=0.0, top=1.0)

    ax.set_ylabel("TCAV score")
    ax.legend(loc="upper right", title="Network layer")

    plt.tight_layout()
    plt.savefig(f"{plt_name}.{file_type}")
    plt.close()


def plot_tcav_scores_rel(experimental_sets, tcav_scores, score_type="sign_count", alpha=0.05, only_significant=True,
                         with_error=True, plt_name="plot", file_type="svg"):
    tcav_score0 = next(iter(tcav_scores.values()))
    layers = list(tcav_score0.keys())
    concept_short_codes = [int(str(concept.id)[:5]) for concept in experimental_sets[0]]
    concept_short_names = [concept.name[:-4] for concept in experimental_sets[0]]
    scores_mean = {layer: [] for layer in layers}
    scores_std = {layer: [] for layer in layers}
    stars = {layer: [] for layer in layers}

    for concept_idx, concept_short_code in enumerate(concept_short_codes):
        for layer in layers:
            pos_scores = assemble_scores(tcav_scores, experimental_sets, concept_idx, layer, score_type)
            neg_scores = 1 - pos_scores
            # t-test
            _, pval = ttest_ind(pos_scores, neg_scores)
            # Bonferroni correction
            m = 2
            alpha = alpha / m
            # Get non-significant and significant scores
            not_significant = np.mean(pos_scores) < np.mean(neg_scores) or pval >= alpha
            stars[layer].append(not_significant)
            if not_significant and only_significant:
                scores_mean[layer].append(0.0)
                scores_std[layer].append(0.0)
            else:
                scores_mean[layer].append(np.mean(pos_scores))
                scores_std[layer].append(np.std(pos_scores))
    scores_mean_df = pd.DataFrame(data=scores_mean, index=concept_short_names)
    scores_std_df = pd.DataFrame(data=scores_std, index=concept_short_names)

    fig = plt.figure(figsize=(10, 3))
    ax = plt.subplot()
    if with_error:
        scores_mean_df.plot.bar(ax=ax, rot=0, yerr=scores_std_df)
    else:
        scores_mean_df.plot.bar(ax=ax, rot=0)

    containers = []
    for container in ax.containers:
        if isinstance(container, BarContainer):
            containers.append(container)
    for container, star in zip(containers, stars.values()):
        labels = []
        for s_idx, s in enumerate(star):
            if s:
                labels.append("*")
                current_color = container.patches[s_idx].get_facecolor()
                new_color = current_color
                # new_color = [c for c in current_color[:-1]]
                # new_color.append(0.25)
                container.patches[s_idx].set_facecolor(new_color)
            else:
                labels.append("")
        ax.bar_label(container, labels=labels, size=18, color=container.patches[0].get_facecolor())
    ax.set_ylim(bottom=0.0, top=1.0)

    ax.set_ylabel("TCAV score")
    ax.legend(loc="upper right", title="Network layer")

    plt.tight_layout()
    plt.savefig(f"{plt_name}.{file_type}")
    plt.close()


def plot_tcav_accuracies(experimental_sets, stats, plt_baseline=True, plt_name="plot", file_type="svg"):
    accs = []
    layers = []
    concept_short_names = [concept.name[:-4] for concept in experimental_sets[0]]
    fig_legends = [", ".join(concept_short_names)]
    for exp_idx, exp in enumerate(stats.values()):
        accs.append([])
        layers.append([])
        for exp_layer in exp.values():
            task_name = ", ".join([concept.name for concept in exp_layer["concepts"]])
            accs[exp_idx].append(exp_layer["stats"]["accs"])
            layers[exp_idx].append(exp_layer["layer"])
    acc_mean = np.mean(accs, axis=0)
    acc_std = np.std(accs, axis=0)
    layers = layers[0]

    fig = plt.figure(figsize=(10, 3))
    ax = plt.subplot()
    plt.errorbar(layers, acc_mean, yerr=acc_std, fmt='-o')
    ax.set_ylim(bottom=0.0, top=1.0)
    plt.legend(fig_legends)
    if plt_baseline:
        baseline_acc = 1 / len(concept_short_names)
        baseline_acc = np.ones(acc_mean.shape) * baseline_acc
        plt.plot(layers, baseline_acc, color="gray", alpha=0.2)
    ax.set_ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(f"{plt_name}.{file_type}")
    plt.close()


def plot_tcav_all_accuracies(all_experimental_sets, all_stats, plt_baseline=False, plt_name="plot", file_type="svg"):
    fig = plt.figure(figsize=(10, 3))
    ax = plt.subplot()
    fig_legends = []
    for experimental_sets, stats in zip(all_experimental_sets, all_stats):
        accs = []
        layers = []
        concept_short_names = [concept.name[:-4] for concept in experimental_sets[0]]
        fig_legends.append(", ".join(concept_short_names))
        for exp_idx, exp in enumerate(stats.values()):
            accs.append([])
            layers.append([])
            for exp_layer in exp.values():
                task_name = ", ".join([concept.name for concept in exp_layer["concepts"]])
                accs[exp_idx].append(exp_layer["stats"]["accs"])
                layers[exp_idx].append(exp_layer["layer"])
        acc_mean = np.mean(accs, axis=0)
        acc_std = np.std(accs, axis=0)
        layers = layers[0]
        plt.errorbar(layers, acc_mean, yerr=acc_std, fmt='-o')
        if plt_baseline:
            baseline_acc = 1 / len(concept_short_names)
            baseline_acc = np.ones(acc_mean.shape) * baseline_acc
            plt.plot(layers, baseline_acc, color="gray", alpha=0.2)
            plt_baseline = False
    ax.set_ylabel("Accuracy")
    plt.legend(fig_legends, loc="upper right", title="Concepts")
    plt.tight_layout()
    plt.savefig(f"{plt_name}.{file_type}")
    plt.close()
