"""Matplotlib figures. Captions are separate; cells have no dots or markers."""
from collections import defaultdict
import colorsys
import itertools
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from data import feature_key

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.facecolor": "white"})


def save_figure(fig, directory, name, caption, captions):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    captions.append(f"{name}: {caption}")


def inclusion_plot(records, labels, config, directory, name, captions):
    if not records:
        return
    originals = [set(record["original"]) for record in records]
    expanded = [{f for group in record["groups"] for f in group["features"]} for record in records]
    features = set.union(*expanded)
    loci = list(config["columns"]["ranges"])
    # Colors identify exact block memberships, including blocks spanning loci.
    maps = [{f: tuple(sorted(group["features"], key=feature_key))
             for group in record["groups"] for f in group["features"]} for record in records]
    signatures = {}
    for feature in features:
        signatures[feature] = tuple(mapping.get(feature, ()) for mapping in maps)
    block_keys = sorted({members for mapping in maps for members in mapping.values() if len(members) > 1})
    colors = {members: np.array(colorsys.hsv_to_rgb((i * .61803398875 + .53) % 1, .70, .64))
              for i, members in enumerate(block_keys)}
    gray = np.array([.57, .62, .65])
    ordered = sorted(features, key=lambda f: (loci.index(feature_key(f)[0]), signatures[f], feature_key(f)[1]))
    per_page = (len(ordered) if config["plots"].get("complete_top_bins", False)
                and all(r["rank"] == 1 for r in records) else config["plots"]["rows_per_page"])
    pages = math.ceil(len(ordered) / per_page)
    for page in range(pages):
        names = ordered[page * per_page:(page + 1) * per_page]
        fig, axes = plt.subplots(1, 2, figsize=(16, max(4, 0.29 * len(names) + 2.3)), sharey=True)
        for ax, sets, processed in zip(axes, (originals, expanded), (False, True)):
            rgb = np.ones((len(names), len(records), 3))
            for i, feature in enumerate(names):
                for j, selected in enumerate(sets):
                    if feature in selected:
                        color = colors.get(maps[j][feature], gray)
                        rgb[i, j] = color if feature in originals[j] else 0.30 * color + 0.70
            ax.imshow(rgb, aspect="auto", interpolation="nearest")
            ticks = list(range(len(labels))) if len(labels) <= 12 else sorted(set([0, *range(4, len(labels), 5)]))
            ax.set_xticks(ticks, [labels[i] for i in ticks], rotation=90 if len(labels) > 12 else 0)
            ax.set_xlabel("Processed bins" if processed else "Original bins")
            ax.set_yticks(range(len(names)), [n.removeprefix("MM_").replace("_", " ") for n in names])
        axes[0].set_ylabel("Amino acid mismatch position")
        fig.legend(handles=[Patch(facecolor="#236c80", label="Original position (dark)"),
                            Patch(facecolor="#bdd3d9", label="Added position (light)"),
                            Patch(facecolor=gray, label="Ungrouped original (gray)")],
                   loc="upper center", ncol=1, frameon=False, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 1 - 1.25 / fig.get_figheight()))
        suffix = f"_page{page + 1:02d}" if pages > 1 else ""
        save_figure(fig, directory, name + suffix,
                    "Feature inclusion before (left) and after (right) processing, with identical rows and column order. "
                    "Dark cells are original features; matching light cells are added alternatives. Each hue identifies an exact "
                    "correlated block, including interlocus members in different locus sections. Gray means ungrouped original. "
                    "White means absent. Colors are local to this comparison, not HLA loci or risk. "
                    "A filled block shows feature inclusion, not independent risk effects. "
                    f"Scheme: {records[0]['scheme']}; page {page + 1}/{pages}.", captions)


def threshold_plot(consistency, risk, config, directory, captions):
    cv = consistency.loc[consistency["comparison"] == "across_cv"]
    if cv.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    selected = risk.loc[(risk["rank"] == 1) & (risk["variant"] == "processed_fixed") & (risk["dataset"] == "test")]
    thresholds = config["correlation"]["thresholds"]
    if len(thresholds) == 1:
        plt.close(fig)
        scopes = config["correlation"].get("scopes", ["within"])
        names = ["original"] + [("any_" if scope == "any" else "") + "r" + str(float(thresholds[0])).replace(".", "p") for scope in scopes]
        labels = ["Original"] + ["Within-locus" if scope == "within" else "Any-locus" for scope in scopes]
        top = risk.loc[(risk["rank"] == 1) & (risk["dataset"] == "test") & risk.variant.isin(["original", "processed_fixed"])]
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        for ax, measure in zip(axes, ["jaccard", "HR", "Adj NoAg HR", "Adj HR"]):
            table = cv if measure == "jaccard" else top
            values = []
            for name in names:
                group = table.loc[table.scheme == name]
                valid = measure == "jaccard" or group[measure + " status"].eq("ok").all()
                values.append(group[measure].mean() if valid else float("nan"))
            bars = ax.bar(labels, values, color=["#929FA6", "#238A9D", "#BE7540"][:len(names)])
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=14)
            ax.set_ylabel("Mean shared-position fraction" if measure == "jaccard" else "Mean held-out " + measure)
            ax.tick_params(axis="x", labelrotation=25)
            ax.margins(y=.20)
        fig.tight_layout()
        save_figure(fig, directory, "figure4_threshold_comparison",
                    f"Original, within-locus and any-locus top-bin comparisons at Pearson r > {thresholds[0]:g}. "
                    "Left: mean pairwise Jaccard across cross-validation folds. Remaining panels: arithmetic means of "
                    "held-out unadjusted, clinical-adjusted and clinical-plus-antigen-adjusted hazard ratios, not pooled "
                    "estimates. Original bin thresholds are unchanged. See top_bin_summary.csv for estimate counts and "
                    "risk_comparison.csv.gz for fold-specific intervals and statuses. These descriptive comparisons do not "
                    "demonstrate clinical benefit or formal equivalence.", captions)
        return
    for col, label, ax in (("jaccard", "Mean shared-position fraction", axes[0]), ("positions", "Mean positions per top bin", axes[1])):
        values = []
        for r in thresholds:
            name = "r" + str(float(r)).replace(".", "p")
            values.append(cv.loc[cv.scheme == name, col].mean())
        ax.plot(thresholds, values, color="#166678", label="Processed")
        ax.axhline(cv.loc[cv.scheme == "original", col].mean(), color="#333333", linestyle="--", label="Original")
        ax.set_ylabel(label)
        ax.set_xlabel("Pearson correlation cutoff")
    for metric, color in (("HR", "#333333"), ("Adj HR", "#166678"), ("Adj NoAg HR", "#b95b2a")):
        values = [selected.loc[selected.scheme == "r" + str(float(r)).replace(".", "p"), metric + " delta"].mean() for r in thresholds]
        axes[2].plot(thresholds, values, color=color, label=metric)
    axes[2].axhline(0, color="#aaaaaa", linewidth=1)
    axes[2].set_ylabel("Mean change in held-out hazard ratio")
    axes[2].set_xlabel("Pearson correlation cutoff")
    axes[0].legend(frameon=False, fontsize=12)
    axes[2].legend(frameon=False, fontsize=12)
    fig.tight_layout()
    save_figure(fig, directory, "figure4_threshold_comparison",
                "Cutoffs begin at r > 0.10. Left: mean pairwise Jaccard similarity of top bins across validation folds "
                "within each imputation and seed. Center: mean top-bin position count. Right: mean paired change in test-set "
                "hazard ratio, processed minus original, using the original bin threshold. Adj HR adjusts for clinical and "
                "antigen covariates; Adj NoAg HR adjusts for clinical covariates. FIBERS uses product fitness with training "
                "residuals. These are descriptive fold averages, not pooled independent effect estimates. Failed/unrequested "
                "Cox fits are missing and are reported in the metrics table.", captions)


def interlocus_figures(summary, config, directory, captions):
    cutoff = config["plots"]["interlocus_threshold"]
    for (imp, left, right), pairs in summary.groupby(["imputation", "locus1", "locus2"], sort=False):
        retained = pairs.loc[(pairs.mean_r > cutoff) & (pairs.folds_available == len(config["folds"]))].copy()
        if retained.empty:
            continue
        maximum = config["plots"]["interlocus_max_positions"]
        left_features = retained.groupby("feature1").mean_r.max().nlargest(maximum).index.tolist()
        right_features = retained.groupby("feature2").mean_r.max().nlargest(maximum).index.tolist()
        left_features.sort(key=feature_key)
        right_features.sort(key=feature_key)
        selected = pairs.loc[pairs.feature1.isin(left_features) & pairs.feature2.isin(right_features)]
        means = selected.pivot(index="feature1", columns="feature2", values="mean_r").reindex(index=left_features, columns=right_features)
        counts = selected.pivot(index="feature1", columns="feature2", values="folds_above_plot_cutoff").reindex_like(means)
        shown = means.to_numpy().copy()
        shown[(shown <= cutoff) | ~np.isfinite(shown)] = np.nan
        fig, ax = plt.subplots(figsize=(max(7, .7 * len(right_features) + 2), max(5, .55 * len(left_features) + 2)))
        im = ax.imshow(shown, cmap="YlGnBu", vmin=cutoff, vmax=1, aspect="auto")
        for i, j in itertools.product(range(len(left_features)), range(len(right_features))):
            if np.isfinite(shown[i, j]):
                ax.text(j, i, f"{shown[i,j]:.2f}\n{int(counts.iloc[i,j])}", ha="center", va="center",
                        fontsize=14, color="white" if shown[i,j] > .75 else "black")
        ax.set_xticks(range(len(right_features)), [feature_key(f)[1] for f in right_features], rotation=90)
        ax.set_yticks(range(len(left_features)), [feature_key(f)[1] for f in left_features])
        ax.set_xlabel(f"{right} amino acid position")
        ax.set_ylabel(f"{left} amino acid position")
        fig.colorbar(im, ax=ax, label="Mean training-fold Pearson r")
        fig.tight_layout()
        save_figure(fig, directory, f"figure5_imp{imp}_{left}_{right}",
                    f"Imputation {imp}: {left} versus {right}. Cells show the mean Pearson coefficient across all configured "
                    f"training folds, with the number of folds exceeding r > {cutoff:g} below. Means at or below {cutoff:g} "
                    "are hidden. Axes are individual positions, not collapsed blocks. Only pairs measurable in every fold "
                    f"are displayed; at most {maximum} positions per axis are selected by their strongest qualifying mean "
                    "relationship. The full tables contain all within- and between-locus pairs, including negative values. "
                    "These correlations are descriptive and do not establish clinical relevance or genetic linkage disequilibrium.", captions)
