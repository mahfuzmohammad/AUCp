import os
import sys
import numpy as np
from torchvision import transforms as T
import cv2
import pandas as pd
import torch
from self_sup_data.chest_xray import SelfSupChestXRay
from model.resnet import resnet18_enc_dec
from proj_model import ProjectionNet
from train_med import SETTINGS
from experiments.chest_xray_tasks import test_real_anomalies
import matplotlib.pyplot as plt
import json
import warnings
import argparse
from thop import profile
import glob
from torchvision import transforms
import copy
import csv

# Make ``aucp`` importable when this script runs from ssl/two_stage/.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from aucp.paths import data_root as _data_root

# warnings.filterwarnings('ignore')


def test(test_dat, setting, model_dir, device, preact=False, pool=True, final=True, show=False, plots=False,
         pix_metrics=False):
    if final:
        fname = os.path.join(model_dir, setting.get('out_dir'), 'final_' + setting.get('fname'))
        # fname = os.path.join(model_dir, setting.get('out_dir'), setting.get('fname')[:-3]+"_249.pt")
    else:
        # fname = os.path.join(model_dir, setting.get('out_dir'), setting.get('fname'))
        #fname = os.path.join(model_dir, setting.get('out_dir'), setting.get('fname')[:-3] + "_249.pt")
        fname = model_dir
    print(fname)
    if not os.path.exists(fname):
        return np.nan, np.nan

    # model = resnet18_enc_dec(num_classes=1, preact=preact, pool=pool, in_channels=1,
    #                          final_activation=setting.get('final_activation')).to(device)
    head_layers = [512] * 1 + [128]
    # num_classes = 2 if cutpate_type is not CutPaste3Way else 3
    num_classes = 2

    model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=num_classes)
    if final:
        model.load_state_dict(torch.load(fname, map_location=device))
    else:
        checkpoint = torch.load(fname, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
       # model.load_state_dict(torch.load(fname, map_location=device))
    print("Load successfully!")

    if plots:
        print("begin test")
        results, fig, preds, image_names = test_real_anomalies(model, test_dat, device=device, batch_size=16, show=show,
                                                               full_size=True, pix_metrics=pix_metrics)
        fig.savefig(os.path.join(out_dir, setting.get('fname')[:-3] + '.png'))
        plt.close(fig)
    else:
        results, _, preds, image_names = test_real_anomalies(model, test_dat, device=device, batch_size=16, show=show,
                                                             plots=plots, full_size=True, pix_metrics=pix_metrics)

    # vis_dir = os.path.join(model_dir, setting.get('out_dir'), "vis")
    # if not os.path.exists(vis_dir):
    #     os.makedirs(vis_dir)

    # preds = torch.tensor(preds)
    # print(preds.shape)
    # # Vis
    # for i in range(preds.shape[0]):
    #     name = image_names[i]
    #     # print(name)
    #     pred = preds[i]

    #     pred = transforms.ToPILImage()(pred)
    #     image_path = os.path.join(vis_dir, name + ".png")
    #     pred.save(image_path)

    example_in = torch.zeros((1, 1, 224, 224)).to(device)
    flops, params = profile(copy.deepcopy(model), inputs=(example_in,))
    flops, params = round(flops * 1e-6, 4), round(params * 1e-6, 4)  # 1e6 = M
    flops, params = str(flops) + "M", str(params) + "M"

    return results, flops, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=False, type=str, default='rsna')
    parser.add_argument("-g", "--gpu", required=False, type=int, default=0)
    parser.add_argument("-s", "--setting", required=False, type=str, default='CutPasteNormal')
    # parser.add_argument("-f", '--fold', type=int, default=0, help='0-4, experiment index')

    args = parser.parse_args()

    #device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    setting = SETTINGS.get(args.setting)
    data = args.data
    final = True

    # Data root is configurable via AUCP_DATA_ROOT (see aucp/paths.py).
    data_root = str(_data_root())
    out_dir = os.path.join("models", args.data)
    model_dir = out_dir

    if data in ['rsna', 'vin', 'brain', 'lag']:
        path = None
        if data == 'rsna':
            path = os.path.join(data_root, 'RSNA')
            # 250
        elif data == 'vin':
            path = os.path.join(data_root, "VinCXR")
            # 400
        elif data == 'brain':
            path = os.path.join(data_root, "BrainTumor")
            # 400
        elif data == 'lag':
            path = os.path.join(data_root, "LAG")
            # 250
        with open(os.path.join(path, "data.json")) as f:
            data_dict = json.load(f)
        normal_test_files_auc = data_dict["test"]["0"]
        anom_test_files_auc = data_dict["test"]["1"]
        normal_test_files_auc = [os.path.join(path, "images", e) for e in normal_test_files_auc]
        anom_test_files_auc = [os.path.join(path, "images", e) for e in anom_test_files_auc]

        
        
        # For AUCp we need to assume all the train data are normal and all the test data are abnormal
        normal_test_files = data_dict["train"]["0"]
        anom_test_files = data_dict["test"]["0"] + data_dict["test"]["1"]
        normal_test_files = [os.path.join(path, "images", e) for e in normal_test_files]
        anom_test_files = [os.path.join(path, "images", e) for e in anom_test_files]
        
        print(f"normal: {len(normal_test_files)}")
        print(f"len(anom_test_files)", len(anom_test_files))
        
        mask_files = None
        
        
        
    elif data == 'isic':
        #for AUC
        path = os.path.join(data_root, "ISIC2018_Task3")
        data_csv = pd.read_csv(os.path.join(path, "ISIC2018_Task3_Test_GroundTruth",
                                            "ISIC2018_Task3_Test_GroundTruth.csv"))
        test_normal_auc = list(data_csv[data_csv['NV'] == 1]['image'])
        test_normal_auc = [e + ".jpg" for e in test_normal_auc]
        test_abnormal_auc = list(data_csv[data_csv['NV'] == 0]['image'])
        test_abnormal_auc = [e + ".jpg" for e in test_abnormal_auc]
        normal_test_files_auc = [os.path.join(path, "ISIC2018_Task3_Test_Input", e) for e in test_normal_auc]
        anom_test_files_auc = [os.path.join(path, "ISIC2018_Task3_Test_Input", e) for e in test_abnormal_auc]
        
        # for aucp
        path = os.path.join(data_root, "ISIC2018_Task3")
        data_train_csv = pd.read_csv(os.path.join(path, "ISIC2018_Task3_Training_GroundTruth",
                                            "ISIC2018_Task3_Training_GroundTruth.csv"))
        test_normal = list(data_train_csv[data_train_csv['NV'] == 1]['image'])
        test_normal = [e + ".jpg" for e in test_normal]
        data_test_csv = pd.read_csv(os.path.join(path, "ISIC2018_Task3_Test_GroundTruth",
                                             "ISIC2018_Task3_Test_GroundTruth.csv"))
        test_abnormal = list(data_test_csv[data_test_csv['NV'] == 0]['image']) + list(
            data_test_csv[data_test_csv['NV'] == 1]['image'])
        test_abnormal = [e + ".jpg" for e in test_abnormal]
        normal_test_files = [os.path.join(path, "ISIC2018_Task3_Training_Input", e) for e in test_normal]
        anom_test_files = [os.path.join(path, "ISIC2018_Task3_Test_Input", e) for e in test_abnormal]
        mask_files = None
        
        
    elif data == 'c16':
        path = os.path.join(data_root, "Camelyon16")
        normal_test_files_auc = glob.glob(os.path.join(path, "test", "good", "*.png"))
        anom_test_files_auc = glob.glob(os.path.join(path, "test", "Ungood", "*.png"))
        
        
        normal_test_files = glob.glob(os.path.join(path, "train", "good", "*.png"))
        good_test_files = glob.glob(os.path.join(path, "test", "good", "*.png"))
        ungood_tests = glob.glob(os.path.join(path, "test", "Ungood", "*.png"))
        anom_test_files = ungood_tests + good_test_files
        mask_files = None
    elif data == 'brats':
        path = os.path.join(data_root, "BraTS2021")
        normal_test_files_auc = glob.glob(os.path.join(path, "test", "normal", "*.png"))
        anom_test_files_auc = glob.glob(os.path.join(path, "test", "tumor", "*.png"))
        
        normal_test_files = glob.glob(os.path.join(path, "train", "*.png"))
        anom_test_files = normal_test_files_auc + anom_test_files_auc
        mask_files = None
        #mask_files = [e.replace("tumor", "annotation").replace("flair", "seg") for e in anom_test_files_auc]
    else:
        raise Exception("Invalid dataset: {}".format(data))

   # pix_metrics = True if data == 'brats' else False
    pix_metrics = False

    test_dat = SelfSupChestXRay(normal_files=normal_test_files, anom_files=anom_test_files,
                                mask_files=mask_files, is_train=False, res=256, transform=T.CenterCrop(224))

    # results, flops, params = test(test_dat, setting, model_dir, device, preact=False, pool=True, final=final,
    #                               pix_metrics=pix_metrics)
    
    test_dat_auc = SelfSupChestXRay(normal_files=normal_test_files_auc, anom_files=anom_test_files_auc,
                                       mask_files=mask_files ,is_train=False, res=256, transform=T.CenterCrop(224))
    
    # list all models
    # model_paths = os.listdir(f"{model_dir}/{setting.get('out_dir')}/models")
    model_paths = os.listdir(f"{model_dir}/{args.setting}")
    
    # model_paths = [f"{model_dir}/{setting.get('out_dir')}/models/{e}" for e in model_paths]
    model_paths = [os.path.join(model_dir, args.setting, e) for e in model_paths if e.endswith('.tch')]
    print(model_paths)
    
    # results, flops, params = test(test_dat, setting, model_dir, device, preact=False, pool=True, final=final,
    #                             pix_metrics=pix_metrics)
    all_results = []

    epoch = 0
    for model_path in model_paths:
        print(model_path)
        print("begin test aucp")
        results, flops, params = test(
            test_dat, 
            args.setting, 
            model_path, 
            device, 
            preact=False, 
            pool=True, 
            final=False,
            pix_metrics=pix_metrics
        )
        
        aucp, ap = results['sample_auc'], results['sample_ap']
        print(f"aucp: {aucp}, ap: {ap}")
        
        print(f"epoch: {epoch}, {model_path}, aucp: {aucp}, ap: {ap}")
        
        print("begin test auc")
        results_auc , flops_auc, params_auc = test(
            test_dat_auc, 
            setting, 
            model_path, 
            device, 
            preact=False, 
            pool=True, 
            final=False,
            pix_metrics=pix_metrics
        )
        # Append the current epoch's results to all_results
        all_results.append({
            "epoch": epoch,
            "model_path": model_path,
            "aucp": aucp,
            "auc": results_auc['sample_auc'],
            "ap": results_auc['sample_ap']
        })
        
        epoch += 1  # increment epoch as needed

    # Once all results are collected, write them to a CSV file
    csv_filename = f"{model_dir}/{setting.get('out_dir')}/{args.data}_aucp.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "model_path", "aucp", "auc", "ap"])
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    print(f"All results have been written to {csv_filename}")
        
        
        

    # auc, ap = results['sample_auc'], results['sample_ap']
    # pix_ap, best_dice = results.setdefault('pixel_ap', None), results.setdefault('best_dice', None)

    # print(args.setting, data, "\tAUCp", auc, "\tAP", ap, "\tPixAP", pix_ap, "\tBestDice", best_dice,
    #       "\tFLOPs", flops, "\tParams", params)

    # performance = pd.DataFrame({"Dataset": [data], "FLOPs": [flops], "params": [params],
    #                             "AUCp": [auc], "AP": [ap], "PixAP": [pix_ap], "BestDice": [best_dice]})
    # performance.to_csv(os.path.join(out_dir, "AUCp{}.csv".format(args.setting)), index=False)
