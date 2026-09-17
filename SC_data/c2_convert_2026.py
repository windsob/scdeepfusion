#!/usr/bin/env python3
"""
MSigDB C2 v2026.1 converter
====================================================================
Steps:
1. gmt -> base CSV (SYMBOL -> semicolon-separated label IDs; IDs increase
   in gmt row order) + ID-to-name mapping
2. Parent filtering: parent sets with >= 50 genes that have a child set
   with >= 90% overlap are removed (min_child_keep_parent=0)
3. Expert-group mapping (MoE routing groups): module_groups_v2026.json
   (6 groups) and module_groups_v2026_typed.json (typed groups)
4. REACTOME+CGP subset CSV (the pipeline's default module library)

Outputs (written next to this script):
- c2.all.v2026.1.Hs.symbols.csv / ..._ID_to_category_name.csv
- c2.all.v2026.1.Hs.symbols_filtered_parent.csv        (full filtered library)
- c2.all.v2026.1.Hs.symbols_ID_to_category_name_filtered_parent.csv
- c2.reactome_cgp.v2026.1.Hs.symbols_filtered_parent.csv (REACTOME+CGP subset)
- module_groups_v2026.json / module_groups_v2026_typed.json
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

SC = Path(__file__).resolve().parent
GMT = SC / "c2.all.v2026.1.Hs.symbols.gmt"
OVERLAP_THRES = 0.90
MIN_PARENT_SIZE = 50
MIN_CHILD_KEEP_PARENT = 0

CP_DBS = {'REACTOME', 'KEGG', 'WP', 'BIOCARTA', 'PID'}


def main():
    # ---- 1. Parse gmt (tab-separated: name \t link \t genes...) ----
    gene_dict = defaultdict(set)      # gene -> {label_id}
    categories = []                   # (label_id, name, link)
    name2id = {}
    with open(GMT) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name, link, genes = parts[0], parts[1], [g for g in parts[2:] if g.strip()]
            if name not in name2id:
                name2id[name] = len(categories) + 1
                categories.append((name2id[name], name, link))
            lid = name2id[name]
            for g in genes:
                gene_dict[g.strip()].add(lid)
    print(f"gmt parsed: {len(categories)} gene sets, {len(gene_dict)} genes")

    cat_df = pd.DataFrame(categories, columns=['labels', 'name', 'link'])
    cat_df.to_csv(SC / "c2.all.v2026.1.Hs.symbols_ID_to_category_name.csv", index=False)
    gene_df = pd.DataFrame(
        [(g, ';'.join(str(i) for i in sorted(ids))) for g, ids in gene_dict.items()],
        columns=['SYMBOL', 'gene_labels'])
    gene_df.to_csv(SC / "c2.all.v2026.1.Hs.symbols.csv", index=False)

    # ---- 2. Parent filtering ----
    label2genes = defaultdict(set)
    for _, row in gene_df.iterrows():
        for lb in str(row['gene_labels']).split(';'):
            label2genes[int(lb.strip())].add(str(row['SYMBOL']).strip())

    items = sorted(label2genes.items(), key=lambda kv: len(kv[1]), reverse=True)
    parent_tags = set()
    for tag_p, genes_p in items:
        if len(genes_p) < MIN_PARENT_SIZE:
            continue
        n_children = 0
        for tag_c, genes_c in items:
            if tag_c == tag_p or len(genes_c) >= len(genes_p):
                continue
            if len(genes_c) > 0 and len(genes_p & genes_c) / len(genes_c) >= OVERLAP_THRES:
                n_children += 1
        if n_children > MIN_CHILD_KEEP_PARENT:
            parent_tags.add(tag_p)
    print(f"parent filtering: removed {len(parent_tags)} redundant parent sets")

    gene_df['label_list'] = gene_df['gene_labels'].astype(str).str.split(';')
    gene_df['label_list_filtered'] = gene_df['label_list'].apply(
        lambda lbs: [lb for lb in lbs if int(lb) not in parent_tags])
    gene_df_filtered = gene_df[gene_df['label_list_filtered'].apply(len) > 0].copy()
    gene_df_filtered['gene_labels'] = gene_df_filtered['label_list_filtered'].str.join(';')

    final_labels = set()
    for lbs in gene_df_filtered['label_list_filtered']:
        final_labels.update(int(lb) for lb in lbs)
    cat_df_filtered = cat_df[cat_df['labels'].isin(final_labels)]

    gene_df_filtered[['SYMBOL', 'gene_labels']].to_csv(
        SC / "c2.all.v2026.1.Hs.symbols_filtered_parent.csv", index=False)
    cat_df_filtered.to_csv(
        SC / "c2.all.v2026.1.Hs.symbols_ID_to_category_name_filtered_parent.csv", index=False)
    print(f"after filtering: {len(gene_df_filtered)} genes, {len(final_labels)} gene sets")

    # ---- 3. Expert-group mappings (two variants) ----
    id2name = dict(zip(cat_df_filtered['labels'], cat_df_filtered['name']))
    prefixes = cat_df_filtered['name'].str.split('_').str[0].value_counts()
    print("name prefix distribution:", prefixes.head(12).to_dict())

    def grp6(name):
        p = str(name).split('_')[0]
        return p if p in CP_DBS else 'CGP'

    def grp_typed(name):
        p = str(name).split('_')[0]
        if p in CP_DBS:
            return p
        if str(name).endswith('_UP'):
            return 'CGP_UP'
        if str(name).endswith('_DN'):
            return 'CGP_DN'
        return 'CGP_OTH'

    gmap = {f"Module_{k}": grp6(v) for k, v in id2name.items()}
    json.dump(gmap, open(SC / "module_groups_v2026.json", "w"))
    json.dump({f"Module_{k}": grp_typed(v) for k, v in id2name.items()},
              open(SC / "module_groups_v2026_typed.json", "w"))
    print("group mappings written: module_groups_v2026.json / module_groups_v2026_typed.json")
    print(pd.Series([grp6(v) for v in id2name.values()]).value_counts().to_dict())

    # ---- 4. REACTOME+CGP subset (pipeline default module library) ----
    keep_ids = {int(k.split('_')[1]) for k, v in gmap.items() if v in ('REACTOME', 'CGP')}
    sub = gene_df_filtered[['SYMBOL', 'label_list_filtered']].copy()
    sub['gene_labels'] = sub['label_list_filtered'].apply(
        lambda lbs: ';'.join(lb for lb in lbs if int(lb) in keep_ids))
    sub = sub[sub['gene_labels'].str.len() > 0][['SYMBOL', 'gene_labels']]
    sub.to_csv(SC / "c2.reactome_cgp.v2026.1.Hs.symbols_filtered_parent.csv", index=False)
    n_sets = len({int(lb) for lbs in sub['gene_labels'].str.split(';') for lb in lbs})
    print(f"REACTOME+CGP subset: {len(sub)} genes, {n_sets} gene sets")


if __name__ == '__main__':
    main()
