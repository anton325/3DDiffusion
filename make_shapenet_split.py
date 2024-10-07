import pandas as pd
from pathlib import Path

def read_shapenet_split(split, category):
    df = pd.read_csv(f"/globalwork/giese/gecco_shapenet/shapenet_split.csv")
    split_df = df[df["split"]==split]
    category_df = split_df[split_df["synsetId"]==int(category[1:])]
    return category_df['modelId'].tolist()


def dict_to_lst_file(d,split,location):
    with open(Path(location,f'{split}.lst'), 'w') as file:
        # Write train paths
        num = 0
        for cat in d:
            for ins in d[cat]:
                num += 1
                file.write(f"{cat}/{ins}\n")
        print(f"Written {num} lines to {split}.lst")

def read_lst(path):
    with open(path) as split_file:
        split_ids = [line.strip() for line in split_file]
    return split_ids

def read_available_instances(category,variant):
    splits = ["train","val","test"]
    instances = []
    for s in splits:
        path = Path(f"/globalwork/giese/gaussians/{category}/{s}{variant}.lst")
        instances.extend(read_lst(path))
    return instances

def write_available_instances(category,variant,instances):
    splits = ["train","val","test"]
    for s in splits:
        official_shapenet_split = read_shapenet_split(s,category)
        instances_this_split = []
        for i in instances:
            if i in official_shapenet_split:
                instances_this_split.append(i)
        path = Path(f"/globalwork/giese/gaussians/{category}/shapenet_{s}{variant}.lst")
        with open(path, 'w') as file:
            for ins in instances_this_split:
                file.write(f"{ins}\n")

def reorder_splits(cat = "02691156"):
    variants = ['_good', '_mid', '']
    for v in variants:
        available_instances = read_available_instances(cat,v)
        write_available_instances(cat,v,available_instances)
        

root = Path("/globalwork/giese/gaussians")
categories = [f for f in root.iterdir() if f.is_dir()]
for c in categories:
    reorder_splits(c.name)

