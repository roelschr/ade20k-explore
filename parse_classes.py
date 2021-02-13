import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

text_files = []
for currpath, folders, files in os.walk("data/ADE20K_2016_07_26/images/training"):
    for file in files:
        if file.endswith("_atr.txt"):
            text_files.append(os.path.join(currpath, file))

def opentxt(f):
    lines = []
    with open(f, "r") as fp:
        for line in fp:
            arr = line.strip().split("#")
            arr.insert(0, f)
            arr = [s.strip() for s in arr]
            arr[-1] = arr[-1].strip("\"").split(",") if arr[-1] != "\"\"" else []
            lines.append(arr)
    
    return lines

result = []
with tqdm(total=len(text_files)) as pbar:
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(opentxt, f) for f in text_files]
        for future in as_completed(futures):
            result += future.result()
            pbar.update(1)

instances = pd.DataFrame(
    result,
    columns=[
        "file_name",
        "instance_number",
        "part_level",
        "is_occluded",
        "class_name",
        "raw_name",
        "attr_list"
    ]
)
instances.to_csv("data/parsed_training.csv", index=False)