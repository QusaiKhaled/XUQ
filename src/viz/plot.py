"""Plotting helpers: membership bands, MI bar chart, target density."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_mi_scores(mi_series, title="Feature Importance via Mutual Information", figsize=(8, 6)):
    """Horizontal bar chart of mutual information scores (ascending)."""
    if not isinstance(mi_series, pd.Series):
        mi_series = pd.Series(mi_series)
    if mi_series.index.name is None and hasattr(mi_series.index, "tolist"):
        pass
    mi_sorted = mi_series.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=figsize)
    mi_sorted.plot(kind="barh", color="skyblue", ax=ax)
    ax.set_xlabel("Mutual Information Score", fontsize=12)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    return fig, ax


def plot_target_density(y, xlabel="Energy Consumption", figsize=(6, 4)):
    """Probability density of target variable."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(y, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title("Target Distribution")
    plt.tight_layout()
    return fig, ax


def plot_membership_functions(
    model,
    feature_names=None,
    n_points=200,
    selected_rules=None,
    figsize_per_subplot=(4, 3),
):
    """
    Plot lower/upper membership functions for each feature and (optionally) selected rules.
    model must have c1, c2, stds, n_rules, n_features.
    """
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(model.n_features)]
    if selected_rules is None:
        selected_rules = list(range(min(4, model.n_rules)))
    n_feat = model.n_features
    fig, axes = plt.subplots(1, n_feat, figsize=(figsize_per_subplot[0] * n_feat, figsize_per_subplot[1]))
    if n_feat == 1:
        axes = [axes]
    for f in range(n_feat):
        ax = axes[f]
        x_min = min(model.c1[:, f].min(), model.centers[:, f].min()) - 0.5
        x_max = max(model.c2[:, f].max(), model.centers[:, f].max()) + 0.5
        x = np.linspace(x_min, x_max, n_points)
        for k in selected_rules:
            if k >= model.n_rules:
                continue
            c1, c2 = model.c1[k, f], model.c2[k, f]
            sigma = model.stds[k, f]
            c_mid = (c1 + c2) / 2
            mu_lower = np.where(
                x <= c_mid,
                np.exp(-0.5 * ((x - c2) / sigma) ** 2),
                np.exp(-0.5 * ((x - c1) / sigma) ** 2),
            )
            mu_upper = np.where(
                x < c1,
                np.exp(-0.5 * ((x - c1) / sigma) ** 2),
                np.where(x <= c2, 1.0, np.exp(-0.5 * ((x - c2) / sigma) ** 2)),
            )
            ax.fill_between(x, mu_lower, mu_upper, alpha=0.3)
            ax.plot(x, mu_lower, color="blue", lw=1)
            ax.plot(x, mu_upper, color="red", lw=1)
        ax.set_xlabel(feature_names[f] if f < len(feature_names) else f"x{f}")
        ax.set_ylabel("Membership")
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.3)
    plt.suptitle("IT2 Membership Functions (lower=blue, upper=red, band=uncertainty)")
    plt.tight_layout()
    return fig, axes
