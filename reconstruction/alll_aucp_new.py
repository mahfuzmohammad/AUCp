import csv
from glob import glob

# auc_txts = glob("Experiment/MedIAnomaly/*/*/fold_0/auc_metrics_*.txt")
# aucp_txts = glob("Experiment/MedIAnomaly/*/*/fold_0/aucp_metrics_*.txt")

# print(f"Found {len(auc_txts)} AUC text files.")
# print(f"Found {len(aucp_txts)} AUCP text files.")

datasets="rsna vin isic lag c16 brain"
methods="ae-ssim ae-perceptual"

for dataset in datasets.split():
    for method in methods.split():
        aucs, aucps = [], []
        # auc_txts = sorted(glob(f"Experiment/MedIAnomaly/{dataset}/{method}/fold_0/auc_metrics_*.txt"))
        # aucp_txts = sorted(glob(f"Experiment/MedIAnomaly/{dataset}/{method}/fold_0/aucp_metrics_*.txt"))

        auc_txts = sorted(glob(f"/data/amciilab/Fazle/AUCp/jay/AUCp_experiments/MedIAnomaly/reconstruction/New_Result/{dataset}/{method}/fold_0/auc_metrics_*.txt"))
        aucp_txts = sorted(glob(f"/data/amciilab/Fazle/AUCp/jay/AUCp_experiments/MedIAnomaly/reconstruction/New_Result/{dataset}/{method}/fold_0/aucp_metrics_*.txt"))

        print(f"Processing dataset: {dataset}, method: {method}...")
        print(f"Found {len(auc_txts)} AUC text files.")
        print(f"Found {len(aucp_txts)} AUCP text files.")

        for aucp_txt in aucp_txts:
            auc_txt = aucp_txt.replace("aucp_metrics", "auc_metrics")

            #metrics_txt = f"Experiment/MedIAnomaly/{dataset}/{method}/fold_0/metrics.txt"
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
        print(f"Found {len(aucs)} AUC values.")
        print(f"Found {len(aucp_txts)} AUCP values.")
        # with open(metrics_txt, "r") as f:   
        #     lines = f.readlines()
        #     for line in lines:
        #         if "AUC" in line and not "PixAUC" in line:
        #             orig_auc = float(line.split()[-1])

        max_aucp = max(aucp for aucp in aucps if aucp > 0)
        index_max_aucp = aucps.index(max_aucp)
        max_auc = aucs[index_max_aucp]
        last_auc = aucs[-1]

        #print(f"Dataset: {dataset}, Method: {method}, Max AUCP: {max_aucp:.4f} (AUC: {max_auc:.4f}), Last AUC: {last_auc:.4f}, Original AUC: {orig_auc:.4f}")
        print(f"Dataset: {dataset}, Method: {method}, Max AUCP: {max_aucp:.4f} (AUC: {max_auc:.4f}), Last AUC: {last_auc:.4f}")