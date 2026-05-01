import argparse
import os

import pandas as pd


LOCUS_RANGE_DICT = {
    'A': [1, 182],
    'B': [1, 182],
    'C': [1, 182],
    'DRB1': [6, 94],
    'DRB345': [6, 94],
    'DQA1': [6, 94],
    'DQB1': [6, 95],
    'DPA1': [6, 94],
    'DPB1': [6, 94],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export rare-filter percentages and audit tables for CV training folds and/or a non-CV dataset."
    )
    parser.add_argument(
        "--cv-datafolder",
        dest="cv_datafolder",
        type=str,
        default=None,
        help="Folder containing CV train/test CSVs such as NewImp_1_CV_1_Train.csv.",
    )
    parser.add_argument(
        "--noncv-datafile",
        dest="noncv_datafile",
        type=str,
        default=None,
        help="Single non-CV CSV to audit, for example NewImp_1.csv.",
    )
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        type=str,
        default=None,
        help="Optional directory for audit outputs. Defaults to the CV folder / non-CV file folder.",
    )
    parser.add_argument(
        "--ra",
        dest="rare_filter",
        type=float,
        default=0.1,
        help="Rare-filter threshold used in training. Default is 0.1.",
    )
    parser.add_argument(
        "--loci-list",
        dest="loci_list",
        type=str,
        default="A,B,C,DRB1,DRB345,DQA1,DQB1",
        help="Comma-separated loci list used in the run.",
    )
    return parser.parse_args()


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def build_mm_feature_list(loci_list):
    mm_feature_list = []
    for locus in loci_list:
        if locus not in LOCUS_RANGE_DICT:
            raise ValueError(f"Unknown locus: {locus}")
        start, end = LOCUS_RANGE_DICT[locus]
        for position in range(start, end + 1):
            mm_feature_list.append(f"MM_{locus}_{position}")
    return mm_feature_list


def export_audit_for_dataframe(df, dataset_label, save_dir, loci_list, rare_filter):
    ensure_directory(save_dir)

    mm_feature_list = build_mm_feature_list(loci_list)
    present_mm_features = [feature for feature in mm_feature_list if feature in df.columns]
    missing_mm_features = [feature for feature in mm_feature_list if feature not in df.columns]

    percentages = df.loc[:, present_mm_features].apply(lambda column: (column > 0).mean())

    if rare_filter > 0.0:
        columns_to_remove = percentages[percentages < rare_filter].index.tolist()
        filter_rule = f"frequency < {rare_filter}"
        filter_threshold = rare_filter
    else:
        columns_to_remove = percentages[percentages == 0.0].index.tolist()
        filter_rule = "frequency == 0.0"
        filter_threshold = 0.0

    percentages_df = (
        percentages.rename("nonzero_fraction")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values(by="feature")
    )
    percentages_df["nonzero_percent"] = percentages_df["nonzero_fraction"] * 100.0
    percentages_df.to_csv(
        os.path.join(save_dir, f"{dataset_label}_rare_filter_percentages.csv"),
        index=False,
    )

    frequency_df = percentages_df.copy()
    frequency_df["filter_threshold"] = filter_threshold
    frequency_df["filter_rule"] = filter_rule
    frequency_df["removed_by_filter"] = frequency_df["feature"].isin(columns_to_remove)
    frequency_df["kept_for_training"] = ~frequency_df["removed_by_filter"]
    frequency_df = frequency_df.sort_values(
        by=["removed_by_filter", "nonzero_fraction", "feature"],
        ascending=[False, True, True],
    )
    frequency_df.to_csv(
        os.path.join(save_dir, f"{dataset_label}_rare_filter_feature_frequencies.csv"),
        index=False,
    )

    kept_features = [feature for feature in present_mm_features if feature not in columns_to_remove]
    count_list = []
    total_count = 0
    for locus in loci_list:
        count = sum(feature.startswith(f"MM_{locus}_") for feature in kept_features)
        total_count += count
        count_list.append(f"{locus}:{count}")

    with open(os.path.join(save_dir, f"{dataset_label}_post_filter_counts.txt"), "w") as handle:
        handle.write(f"RareFilterThreshold:{filter_threshold}\n")
        handle.write(f"RareFilterRule:{filter_rule}\n")
        handle.write(f"RemovedFeatures:{len(columns_to_remove)}\n")
        handle.write(f"MissingMMFeatures:{len(missing_mm_features)}\n")
        for item in count_list:
            handle.write(f"{item}\n")
        handle.write(f"Total:{total_count}\n")

    print(
        f"Saved audit for {dataset_label}: "
        f"{dataset_label}_rare_filter_percentages.csv, "
        f"{dataset_label}_rare_filter_feature_frequencies.csv, "
        f"{dataset_label}_post_filter_counts.txt"
    )


def export_cv_audits(cv_datafolder, save_dir, loci_list, rare_filter):
    train_files = sorted(
        filename
        for filename in os.listdir(cv_datafolder)
        if filename.endswith("_Train.csv") and os.path.isfile(os.path.join(cv_datafolder, filename))
    )
    if len(train_files) == 0:
        raise FileNotFoundError(f"No *_Train.csv files found in {cv_datafolder}")

    target_dir = save_dir if save_dir is not None else cv_datafolder
    for filename in train_files:
        dataset_label = os.path.splitext(filename)[0]
        df = pd.read_csv(os.path.join(cv_datafolder, filename))
        export_audit_for_dataframe(df, dataset_label, target_dir, loci_list, rare_filter)


def export_noncv_audit(noncv_datafile, save_dir, loci_list, rare_filter):
    if not os.path.exists(noncv_datafile):
        raise FileNotFoundError(noncv_datafile)

    target_dir = save_dir if save_dir is not None else os.path.dirname(noncv_datafile) or "."
    dataset_label = os.path.splitext(os.path.basename(noncv_datafile))[0]
    df = pd.read_csv(noncv_datafile)
    export_audit_for_dataframe(df, dataset_label, target_dir, loci_list, rare_filter)


def main():
    args = parse_args()

    if args.cv_datafolder is None and args.noncv_datafile is None:
        raise ValueError("Provide at least one of --cv-datafolder or --noncv-datafile.")

    loci_list = [item.strip() for item in args.loci_list.split(",") if item.strip() != ""]

    if args.cv_datafolder is not None:
        export_cv_audits(args.cv_datafolder, args.save_dir, loci_list, args.rare_filter)

    if args.noncv_datafile is not None:
        export_noncv_audit(args.noncv_datafile, args.save_dir, loci_list, args.rare_filter)


if __name__ == "__main__":
    main()
