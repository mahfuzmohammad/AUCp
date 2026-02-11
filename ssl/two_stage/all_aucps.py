import csv
from glob import glob

aucp_csvs = sorted(glob("models/*/*/aucp_results.csv"))

print("Found {} AUCp CSVs".format(len(aucp_csvs)))

rsna_anat_aucs, rsna_anat_aucps = [], []
rsna_cutpaste3_aucs, rsna_cutpaste3_aucps = [], []

for csv in aucp_csvs:
    aucs, aucps = [], []
    # print (f"Processing {csv}...")
    with open(csv, "r") as f:
        lines = f.readlines()
        for line in lines:
            if ".tch" in line:
                aucs.append(float(line.split(",")[2]))
                aucps.append(float(line.split(",")[3]))
        dataset = line.split(",")[1].split("/")[1]
        strategy = line.split(",")[1].split("/")[2]
        
        max_aucp = max(aucp for aucp in aucps if aucp > 0)
        index_max_aucp = aucps.index(max_aucp)
        max_auc = aucs[index_max_aucp]
        last_auc = aucs[-1]

    results_txt = csv.replace("aucp_results.csv", "results.txt")
    with open(results_txt, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "AUC" in line:
                orig_auc = float(line.split()[-1])

    if dataset == "rsna":
        if strategy == "AnatPaste":
            rsna_anat_aucs = aucs
            rsna_anat_aucps = aucps
        elif strategy == "CutPaste3Way":
            rsna_cutpaste3_aucs = aucs
            rsna_cutpaste3_aucps = aucps

    print(f"Dataset: {dataset}, Strategy: {strategy}, Max AUCp: {max_aucp:.4f} (AUC: {max_auc:.4f}), Last AUC: {last_auc:.4f}, Original AUC: {orig_auc:.4f}, Index of Max AUCp: {index_max_aucp}")
    # print(f"Dataset: {dataset}, Strategy: {strategy}, Last AUC: {last_auc:.4f}, AUC at max AUCp: {max_auc:.4f}, Original AUC: {orig_auc:.4f}")

################################################################

print("length of vin_aeu_aucs:", len(rsna_anat_aucs))
print("length of vin_aeu_aucps:", len(rsna_anat_aucps))
print("length of brain_aeu_aucs:", len(rsna_cutpaste3_aucs))
print("length of brain_aeu_aucps:", len(rsna_cutpaste3_aucps))

# Can we draw 3 plots (one for each) where x-axis is the model i and y axis shows AUCp and AUC?
import matplotlib.pyplot as plt

# VIN plot
plt.figure(figsize=(8, 4))
plt.scatter(range(1, len(rsna_anat_aucs)+1), rsna_anat_aucs, label='RSNA AnatPaste AUC', color='blue', marker='o')
plt.scatter(range(1, len(rsna_anat_aucps)+1), rsna_anat_aucps, label='RSNA AnatPaste AUCp', color='orange', marker='x')
plt.xlabel('Model Index')
plt.ylabel('Value')
plt.title('RSNA AnatPaste: AUC and AUCP Scatter Plot')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('rsna_anatpaste_auc_aucp_scatter.png')
plt.close()

# Brain plot
plt.figure(figsize=(8, 4))
plt.scatter(range(1, len(rsna_cutpaste3_aucs)+1), rsna_cutpaste3_aucs, label='RSNA CutPaste3Way AUC', color='green', marker='o')
plt.scatter(range(1, len(rsna_cutpaste3_aucps)+1), rsna_cutpaste3_aucps, label='RSNA CutPaste3Way AUCP', color='red', marker='x')
plt.xlabel('Model Index')
plt.ylabel('Value')
plt.title('RSNA CutPaste3Way: AUC and AUCP Scatter Plot')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('rsna_cutpaste3way_auc_aucp_scatter.png')
plt.close()
