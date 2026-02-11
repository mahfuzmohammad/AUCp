import csv
from glob import glob

aucp_csvs = sorted(glob("output/*.csv"))

print("Found {} AUCp CSVs".format(len(aucp_csvs)))

for csv in aucp_csvs:
    aucs, aucps = [], []
    with open(csv, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "AUC" not in line:
                aucs.append(float(line.split(",")[1]))
                aucps.append(float(line.split(",")[2]))
        dataset = csv.split("/")[-1].split("_")[0]
        strategy = csv.split("/")[-1].split("_")[1]
        
        max_aucp = max(aucp for aucp in aucps if aucp > 0)
        index_max_aucp = aucps.index(max_aucp)
        max_auc = aucs[index_max_aucp]
        last_auc = aucs[-1]

    # print(f"Dataset: {dataset}, Strategy: {strategy}, Max AUCp: {max_aucp:.4f} (AUC: {max_auc:.4f}), Last AUC: {last_auc:.4f}")
    print(f"Dataset: {dataset}, Strategy: {strategy}, Last AUC: {last_auc:.4f}, AUC at max AUCp: {max_auc:.4f}")
        