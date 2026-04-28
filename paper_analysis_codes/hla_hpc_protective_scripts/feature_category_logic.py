def mode_title(mode_name):
    if mode_name == "default":
        return "Original"
    return mode_name.replace("_", " ").title()


def _sort_feature_names(feature_names, counts_by_mode, sort_key_fn):
    return sorted(feature_names, key=lambda feature_name: sort_key_fn(feature_name, counts_by_mode))


def categorize_three_mode_features(counts_by_mode, mode_order, core_shared_min, min_feature_count):
    reference_mode, compare_mode_a, compare_mode_b = mode_order
    reference_title = mode_title(reference_mode)
    compare_a_title = mode_title(compare_mode_a)
    compare_b_title = mode_title(compare_mode_b)

    section_specs = [
        {
            "name": "Core shared",
            "sort_key": lambda feature_name, counts: (
                -(
                    counts[reference_mode].get(feature_name, 0)
                    + counts[compare_mode_a].get(feature_name, 0)
                    + counts[compare_mode_b].get(feature_name, 0)
                ),
                -min(
                    counts[reference_mode].get(feature_name, 0),
                    counts[compare_mode_a].get(feature_name, 0),
                    counts[compare_mode_b].get(feature_name, 0),
                ),
                -counts[reference_mode].get(feature_name, 0),
                -counts[compare_mode_a].get(feature_name, 0),
                -counts[compare_mode_b].get(feature_name, 0),
                feature_name,
            ),
        },
        {
            "name": f"{compare_a_title}-skewed\nthree-way shared",
            "sort_key": lambda feature_name, counts: (
                -counts[compare_mode_a].get(feature_name, 0),
                -counts[reference_mode].get(feature_name, 0),
                -counts[compare_mode_b].get(feature_name, 0),
                feature_name,
            ),
        },
        {
            "name": f"{compare_b_title}-skewed\nthree-way shared",
            "sort_key": lambda feature_name, counts: (
                -counts[compare_mode_b].get(feature_name, 0),
                -counts[reference_mode].get(feature_name, 0),
                -counts[compare_mode_a].get(feature_name, 0),
                feature_name,
            ),
        },
        {
            "name": f"{reference_title} / {compare_a_title}\nshared",
            "sort_key": lambda feature_name, counts: (
                -counts[compare_mode_a].get(feature_name, 0),
                -counts[reference_mode].get(feature_name, 0),
                feature_name,
            ),
        },
        {
            "name": f"{reference_title} / {compare_b_title}\nshared",
            "sort_key": lambda feature_name, counts: (
                -counts[compare_mode_b].get(feature_name, 0),
                -counts[reference_mode].get(feature_name, 0),
                feature_name,
            ),
        },
        {
            "name": f"{compare_a_title} / {compare_b_title}\nshared",
            "sort_key": lambda feature_name, counts: (
                -(
                    counts[compare_mode_a].get(feature_name, 0)
                    + counts[compare_mode_b].get(feature_name, 0)
                ),
                -max(
                    counts[compare_mode_a].get(feature_name, 0),
                    counts[compare_mode_b].get(feature_name, 0),
                ),
                feature_name,
            ),
        },
        {
            "name": f"{reference_title} only",
            "sort_key": lambda feature_name, counts: (
                -counts[reference_mode].get(feature_name, 0),
                feature_name,
            ),
        },
        {
            "name": f"{compare_a_title} only",
            "sort_key": lambda feature_name, counts: (
                -counts[compare_mode_a].get(feature_name, 0),
                feature_name,
            ),
        },
        {
            "name": f"{compare_b_title} only",
            "sort_key": lambda feature_name, counts: (
                -counts[compare_mode_b].get(feature_name, 0),
                feature_name,
            ),
        },
    ]

    feature_rows_by_name = {}
    category_to_features = {section_spec["name"]: [] for section_spec in section_specs}
    all_features = sorted(set().union(*[set(counter.keys()) for counter in counts_by_mode.values()]))

    for feature_name in all_features:
        reference_count = counts_by_mode[reference_mode].get(feature_name, 0)
        compare_a_count = counts_by_mode[compare_mode_a].get(feature_name, 0)
        compare_b_count = counts_by_mode[compare_mode_b].get(feature_name, 0)
        if max(reference_count, compare_a_count, compare_b_count) < min_feature_count:
            continue

        if (
            reference_count >= core_shared_min
            and compare_a_count >= core_shared_min
            and compare_b_count >= core_shared_min
        ):
            section_name = "Core shared"
            category_rule = (
                f"All three modes recur in at least {core_shared_min} folds."
            )
            category_reason = (
                f"{feature_name} appears in {reference_count} {reference_mode} folds, "
                f"{compare_a_count} {compare_mode_a} folds, and {compare_b_count} {compare_mode_b} folds."
            )
        elif reference_count > 0 and compare_a_count > 0 and compare_b_count > 0:
            if compare_a_count >= compare_b_count:
                section_name = f"{compare_a_title}-skewed\nthree-way shared"
                category_rule = (
                    f"All three modes are present, but counts do not reach the core threshold and "
                    f"{compare_mode_a} recurrence is at least {compare_mode_b} recurrence."
                )
                category_reason = (
                    f"{feature_name} appears in all three modes "
                    f"({reference_count}, {compare_a_count}, {compare_b_count}) but is not core shared; "
                    f"{compare_a_count} >= {compare_b_count}, so it is grouped as {compare_a_title}-skewed."
                )
            else:
                section_name = f"{compare_b_title}-skewed\nthree-way shared"
                category_rule = (
                    f"All three modes are present, but counts do not reach the core threshold and "
                    f"{compare_mode_b} recurrence exceeds {compare_mode_a} recurrence."
                )
                category_reason = (
                    f"{feature_name} appears in all three modes "
                    f"({reference_count}, {compare_a_count}, {compare_b_count}) but is not core shared; "
                    f"{compare_b_count} > {compare_a_count}, so it is grouped as {compare_b_title}-skewed."
                )
        elif reference_count > 0 and compare_a_count > 0:
            section_name = f"{reference_title} / {compare_a_title}\nshared"
            category_rule = (
                f"{reference_mode} and {compare_mode_a} are present while {compare_mode_b} is absent."
            )
            category_reason = (
                f"{feature_name} appears in {reference_count} {reference_mode} folds and "
                f"{compare_a_count} {compare_mode_a} folds, with 0 {compare_mode_b} folds."
            )
        elif reference_count > 0 and compare_b_count > 0:
            section_name = f"{reference_title} / {compare_b_title}\nshared"
            category_rule = (
                f"{reference_mode} and {compare_mode_b} are present while {compare_mode_a} is absent."
            )
            category_reason = (
                f"{feature_name} appears in {reference_count} {reference_mode} folds and "
                f"{compare_b_count} {compare_mode_b} folds, with 0 {compare_mode_a} folds."
            )
        elif compare_a_count > 0 and compare_b_count > 0:
            section_name = f"{compare_a_title} / {compare_b_title}\nshared"
            category_rule = (
                f"{compare_mode_a} and {compare_mode_b} are present while {reference_mode} is absent."
            )
            category_reason = (
                f"{feature_name} appears in {compare_a_count} {compare_mode_a} folds and "
                f"{compare_b_count} {compare_mode_b} folds, with 0 {reference_mode} folds."
            )
        elif reference_count > 0:
            section_name = f"{reference_title} only"
            category_rule = (
                f"{reference_mode} is present while the other two modes are absent."
            )
            category_reason = (
                f"{feature_name} appears in {reference_count} {reference_mode} folds, "
                f"with 0 {compare_mode_a} folds and 0 {compare_mode_b} folds."
            )
        elif compare_a_count > 0:
            section_name = f"{compare_a_title} only"
            category_rule = (
                f"{compare_mode_a} is present while the other two modes are absent."
            )
            category_reason = (
                f"{feature_name} appears in {compare_a_count} {compare_mode_a} folds, "
                f"with 0 {reference_mode} folds and 0 {compare_mode_b} folds."
            )
        elif compare_b_count > 0:
            section_name = f"{compare_b_title} only"
            category_rule = (
                f"{compare_mode_b} is present while the other two modes are absent."
            )
            category_reason = (
                f"{feature_name} appears in {compare_b_count} {compare_mode_b} folds, "
                f"with 0 {reference_mode} folds and 0 {compare_mode_a} folds."
            )
        else:
            continue

        category_to_features[section_name].append(feature_name)
        feature_rows_by_name[feature_name] = {
            "section": section_name,
            "feature": feature_name,
            "category_rule": category_rule,
            "category_reason": category_reason,
            reference_mode: reference_count,
            compare_mode_a: compare_a_count,
            compare_mode_b: compare_b_count,
        }

    ordered_section_rows = []
    ordered_feature_rows = []
    for section_spec in section_specs:
        feature_names = _sort_feature_names(
            category_to_features[section_spec["name"]],
            counts_by_mode,
            section_spec["sort_key"],
        )
        if len(feature_names) == 0:
            continue
        ordered_section_rows.append((section_spec["name"], feature_names))
        for feature_name in feature_names:
            ordered_feature_rows.append(feature_rows_by_name[feature_name])

    return ordered_section_rows, ordered_feature_rows
