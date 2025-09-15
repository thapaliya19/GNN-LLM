#!/usr/bin/env python3
import itertools, csv, argparse, pathlib, json

seed_map = {
    "cora": 1,
    "pubmed": 2,
    "usa": 3,
    "europe": 4,
    "brazil": 5,
}

buffer_alpha_map = {
    "cora": 0.8,
    "pubmed": 0.5,
    "usa": 0.5,
    "europe": 0.7,
    "brazil": 0.5,
}


def as_list(x):
    return x if isinstance(x, list) else [x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scripts/manifest.tsv")
    ap.add_argument("--datasets", nargs="+", default=["cora", "europe"])
    ap.add_argument("--ex_block_templates", nargs="+", default=["default", "no_pred"])
    ap.add_argument(
        "--sim_methods",
        nargs="+",
        default=["weighted_sum", "max", "product", "concat_cosine"],
    )
    ap.add_argument("--buffer_k", nargs="+", type=int, default=[5, 10])
    ap.add_argument("--output_root", default="runs")
    ap.add_argument("--log_root", default="logs")
    ap.add_argument("--group", default="grid")
    args = ap.parse_args()

    rows = []

    for d, pt, sm, bk in itertools.product(
        args.datasets, args.ex_block_templates, args.sim_methods, args.buffer_k
    ):
        setting = "text_rich" if d in ("cora", "pubmed") else "text_free"
        node_text = "title_abstract" if setting == "text_rich" else "none"
        seed = seed_map[d]
        rows.append(
            dict(
                dataset=d,
                seed=seed,
                ex_block_template=pt,
                sim_method=sm,
                buffer_k=bk,
                setting=setting,
                node_text=node_text,
                output_root=args.output_root,
                log_root=args.log_root,
                group=f"{args.group}_{d}",
                buffer_alpha=buffer_alpha_map[d],
            )
        )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
