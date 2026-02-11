import csv
from glob import glob

# auc_txts = glob("Experiment/MedIAnomaly/*/*/fold_0/auc_metrics_*.txt")
# aucp_txts = glob("Experiment/MedIAnomaly/*/*/fold_0/aucp_metrics_*.txt")

# print(f"Found {len(auc_txts)} AUC text files.")
# print(f"Found {len(aucp_txts)} AUCP text files.")

# datasets="rsna vin brain lag isic c16 brats"
# methods="memae aeu"

datasets="rsna vin brain lag"
methods="ae ae-l1"


vin_aeu_aucs, vin_aeu_aucps = [], []
brain_aeu_aucs, brain_aeu_aucps = [], []

for dataset in datasets.split():
    for method in methods.split():
        aucs, aucps = [], []
        #auc_txts = sorted(glob(f"Experiment/MedIAnomaly/{dataset}/{method}/fold_0/auc_metrics_*.txt"))
        #aucp_txts = sorted(glob(f"Experiment/MedIAnomaly/{dataset}/{method}/fold_0/aucp_metrics_*.txt"))
        auc_txts = sorted(glob(f"/data/amciilab/Fazle/AUCp/jay/AUCp_experiments/MedIAnomaly/reconstruction/New_Result/{dataset}/{method}/fold_0/auc_metrics_*.txt"))
        aucp_txts = sorted(glob(f"/data/amciilab/Fazle/AUCp/jay/AUCp_experiments/MedIAnomaly/reconstruction/New_Result/{dataset}/{method}/fold_0/aucp_metrics_*.txt"))
        # print(f"Processing dataset: {dataset}, method: {method}...")
        # print(f"Found {len(auc_txts)} AUC text files.")
        # print(f"Found {len(aucp_txts)} AUCP text files.")

        for aucp_txt in aucp_txts:
            
            auc_txt = aucp_txt.replace("aucp_metrics", "auc_metrics")
            #metrics_txt = f"Experiment/MedIAnomaly/{dataset}/{method}/fold_0/metrics.txt"
            metrics_txt = f"/data/amciilab/Fazle/AUCp/jay/AUCp_experiments/MedIAnomaly/reconstruction/New_Result/{dataset}/{method}/fold_0/metrics.txt"
            if auc_txt in auc_txts:
                with open(auc_txt, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if "AUC" in line and not "PixAUC" in line:
                            aucs.append(float(line.split()[-1]))
                            
                
                with open(aucp_txt, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if "AUC" in line and not "PixAUC" in line:
                            aucps.append(float(line.split()[-1]))
           # print(f"Processed {len(aucp_txts)} AUCP files for dataset: {dataset}, method: {method}.")
            with open(metrics_txt, "r") as f:   
                lines = f.readlines()
                for line in lines:
                    if "AUC" in line and not "PixAUC" in line:
                        orig_auc = float(line.split()[-1])

        max_aucp = max(aucp for aucp in aucps if aucp > 0)
        index_max_aucp = aucps.index(max_aucp)
        max_auc = aucs[index_max_aucp]
        last_auc = aucs[-1]

        # print(f"Dataset: {dataset}, Method: {method}, Max AUCP: {max_aucp:.4f} (AUC: {max_auc:.4f}), Last AUC: {last_auc:.4f}, Original AUC: {orig_auc:.4f}")
        print(f"Dataset: {dataset}, Method: {method}, Last AUC: {last_auc:.4f}, AUC for max AUCp: {max_auc:.4f}")
        print("----------------------------------------------------------------")

# ################################################################
#         if dataset == "vin" and method == "aeu":
#             vin_aeu_aucs = aucs
#             vin_aeu_aucps = aucps
#         elif dataset == "brain" and method == "aeu":
#             brain_aeu_aucs = aucs
#             brain_aeu_aucps = aucps
#         else:
#             continue

# print("length of vin_aeu_aucs:", len(vin_aeu_aucs))
# print("length of vin_aeu_aucps:", len(vin_aeu_aucps))
# print("length of brain_aeu_aucs:", len(brain_aeu_aucs))
# print("length of brain_aeu_aucps:", len(brain_aeu_aucps))

# # Can we draw 3 plots (one for each) where x-axis is the model i and y axis shows AUCp and AUC?
# import matplotlib.pyplot as plt

# # VIN plot
# plt.figure(figsize=(8, 4))
# plt.scatter(range(1, len(vin_aeu_aucs)+1), vin_aeu_aucs, label='VIN AUC', color='blue', marker='o')
# plt.scatter(range(1, len(vin_aeu_aucps)+1), vin_aeu_aucps, label='VIN AUCP', color='orange', marker='x')
# plt.xlabel('Model Index')
# plt.ylabel('Value')
# plt.title('VIN AEU: AUC and AUCP Scatter Plot')
# plt.ylim(0, 1)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('vin_aeu_auc_aucp_scatter.png')
# plt.close()

# # Brain plot
# plt.figure(figsize=(8, 4))
# plt.scatter(range(1, len(brain_aeu_aucs)+1), brain_aeu_aucs, label='Brain AUC', color='green', marker='o')
# plt.scatter(range(1, len(brain_aeu_aucps)+1), brain_aeu_aucps, label='Brain AUCP', color='red', marker='x')
# plt.xlabel('Model Index')
# plt.ylabel('Value')
# plt.title('Brain AEU: AUC and AUCP Scatter Plot')
# plt.ylim(0, 1)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('brain_aeu_auc_aucp_scatter.png')
# plt.close()
