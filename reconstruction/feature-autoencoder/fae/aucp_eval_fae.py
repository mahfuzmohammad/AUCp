import os
from argparse import ArgumentParser
from collections import defaultdict
from time import time
from warnings import warn

import numpy as np
import torch
import wandb
import os
from torchvision import transforms
from thop import profile

# DATAROOT = os.environ.get('DATAROOT')
# CAMCANROOT = os.path.join(DATAROOT, 'CamCAN')
# BRATSROOT = os.path.join(DATAROOT, 'BraTS')
# MOODROOT = os.path.join(DATAROOT, 'MOOD')
# PHYSIONETROOT = os.path.join(DATAROOT, 'Physionet-ICH')

# WANDBNAME = os.environ.get('WANDBNAME')
# WANDBPROJECT = os.environ.get('WANDBPROJECT')
# WANDBDIR = os.environ.get(
#     'WANDBDIR',
#     os.path.join(os.path.expanduser('~'), 'wandb')
# )
WANDBNAME = 'AD'
# WANDBPROJECT = 'fae'
WANDBPROJECT = 'fae-tmp'
WANDBDIR = 'wandb'
# from fae import WANDBNAME, WANDBPROJECT, WANDBDIR
# from fae.configs.base_config import base_parser
# from fae.data import datasets
# from fae.models import models
# from fae.utils.utils import seed_everything
# from fae.utils import evaluation

# from fae import WANDBNAME, WANDBPROJECT, WANDBDIR
from configs.base_config import base_parser
from data import datasets
from models import models
from utils.utils import seed_everything
from utils import evaluation
import csv

""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""

parser = ArgumentParser(
    description="Arguments for training the Feature Autoencoder",
    parents=[base_parser],
    conflict_handler='resolve'
)
config = parser.parse_args()
config.method = "FAE"

if not config.train and config.resume_path is None:
    warn("Testing untrained model")

# Select training device
# config.device = 'cuda:6' if torch.cuda.is_available() else 'cpu'
config.device = 'cuda:{}'.format(config.gpu)
torch.cuda.set_device(config.gpu)

""""""""""""""""""""""""""""""" Reproducibility """""""""""""""""""""""""""""""
# seed_everything(config.seed)


""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""


def get_model(config):
    if config.model in models.__dict__:
        model_cls = models.__dict__[config.model]
    else:
        raise ValueError(f'Model {config.model} not found')

    return model_cls(config)


print("Initializing model...")
model = get_model(config).to(config.device)

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                             weight_decay=config.weight_decay)
# Print model
# print(model.ae.enc)
# print(model.ae.dec)
# print(model.ae)

# if config.resume_path is not None:
# if not config.train:
#     print("Loading model from checkpoint...")
#     # path = os.path.join('output', 'checkpoints', config.train_dataset+".pt")
#     # model.load(config.resume_path)
#     model.load(config)


""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""


print("Loading data...")
t_load_data_start = time()
train_loader, val_loader, test_loader = datasets.get_dataloaders(config)
train_loader_aucp, val_loader_aucp, test_loader_aucp = datasets.get_dataloaders_aucp(config)
print(f'Loaded {config.train_dataset} in '
      f'{time() - t_load_data_start:.2f}s')


""""""""""""""""""""""""""""""""""""" W&B """""""""""""""""""""""""""""""""""""

# wandb_dir = f"{WANDBDIR}/fae/{config.method}"
# os.makedirs(wandb_dir, exist_ok=True)
# # wandb.init(project=WANDBPROJECT, entity=WANDBNAME, config=config,
# #            mode="disabled" if config.debug else "online",
# #            dir=wandb_dir)
# wandb.init(project=WANDBPROJECT, config=config,
#            mode="disabled" if config.debug else "online",
#            dir=wandb_dir)
# wandb.watch(model)


""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""


def train_step(model, optimizer, x, device):
    model.train()
    optimizer.zero_grad()
    x = x.to(device)
    loss_dict = model.loss(x)
    loss = loss_dict['rec_loss']
    loss.backward()
    optimizer.step()
    return loss_dict


def train(model, optimizer, train_loader, val_loader, config):
    print('Starting training...')
    i_iter = 0
    i_epoch = 0

    train_losses = defaultdict(list)

    t_start = time()
    while True:
        for data_batch in train_loader:
            x = data_batch['img']
            i_iter += 1
            loss_dict = train_step(model, optimizer, x, config.device)

            # Add to losses
            for k, v in loss_dict.items():
                train_losses[k].append(v.item())

            if i_iter % config.log_frequency == 0:
                # Print training loss
                log_msg = " - ".join([f'{k}: {np.mean(v):.4f}' for k,
                                     v in train_losses.items()])
                log_msg = f"Iteration {i_iter} - " + log_msg
                log_msg += f" - time: {time() - t_start:.2f}s"
                print(log_msg)

                # # Log to w&b
                # wandb.log({
                #     f'train/{k}': np.mean(v) for k, v in train_losses.items()
                # }, step=i_iter)

                # Reset
                train_losses = defaultdict(list)

            if i_iter % config.val_frequency == 0:
                validate(model, val_loader, config.device, i_iter)

            # Save model weights
            if i_iter % config.save_frequency == 0:
                # model.save(config, 'last.pt')
                model.save(config, f'fae_{config.train_dataset}_{i_iter}.pt')   

            if i_iter >= config.max_steps:
                print(
                    f'Reached {config.max_steps} iterations. Finished training.')

                # Final validation
                print("Final validation...")
                validate(model, val_loader, config.device, i_iter)
                return

        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({i_iter} iterations)')
        # Save model weights
        # model.save(config, f'fae_{config.loss_fn}_{config.train_dataset}_{i_epoch}.pt')


def val_step(model, x, device):
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        loss_dict = model.loss(x)
        anomaly_map, anomaly_score = model.predict_anomaly(x)
    return loss_dict, anomaly_map.cpu(), anomaly_score.cpu()


def validate(model, val_loader, device, i_iter):
    val_losses = defaultdict(list)
    # pixel_aps = []
    labels = []
    masks = []
    anomaly_scores = []
    anomaly_maps = []
    i_val_step = 0

    # for x, y, label in val_loader:
    for data_batch in val_loader:
        x, label = data_batch['img'], data_batch['label']
        # x, y, anomaly_map: [b, 1, h, w]
        # Compute loss, anomaly map and anomaly score
        loss_dict, anomaly_map, anomaly_score = val_step(model, x, device)

        # Compute metrics
        # pixel_ap = evaluation.compute_average_precision(anomaly_map, y)

        # for k, v in loss_dict.items():
        #     val_losses[k].append(v.item())
        # pixel_aps.append(pixel_ap)
        labels.append(label)
        anomaly_scores.append(anomaly_score.detach().cpu())
        anomaly_maps.append(anomaly_map.detach().cpu())

        if config.train_dataset == 'brats':
            masks.append(data_batch['mask'])

        # i_val_step += 1
        # if i_val_step >= config.val_steps:
        #     break

    # Compute sample-wise average precision and AUROC over all validation steps
    labels = torch.cat(labels)
    anomaly_scores = torch.cat(anomaly_scores)
    sample_ap = evaluation.compute_average_precision(anomaly_scores, labels)
    sample_auroc = evaluation.compute_auroc(anomaly_scores, labels)

    results = {'auc': sample_auroc, 'ap': sample_ap}
    if config.train_dataset == 'brats':
        score_maps = torch.cat(anomaly_maps, dim=0).numpy()
        masks = torch.cat(masks, dim=0).unsqueeze(1).numpy()  # # NxHxW -> Nx1xHxW
        pix_ap = evaluation.compute_average_precision(score_maps, masks)
        best_dice, best_thresh = evaluation.compute_best_dice(score_maps, masks)
        results.update({'PixAP': pix_ap, 'BestDice': best_dice, 'BestThresh': best_thresh})

    # Print validation results
    print("\nValidation results:")
    # log_msg = " - ".join([f'val {k}: {np.mean(v):.4f} - ' for k,
    #                      v in val_losses.items()])
    # log_msg += f"\npixel-wise average precision: {np.mean(pixel_aps):.4f}\n"
    # log_msg = " - ".join([f'val {k}: {np.mean(v):.4f} - ' for k, v in val_losses.items()])
    # log_msg += f"sample-wise AUROC: {sample_auroc:.4f} - "
    # log_msg += f"sample-wise average precision: {sample_ap:.4f} - "
    # log_msg += f"Average positive label: {labels.float().mean():.4f}\n"
    # print(log_msg)
    keys = list(results.keys())
    for key in keys:
        print(key + ": {:.5f}".format(results[key]), end="  ")
        # results["val/" + key] = results.pop(key)
    print()

    # # Log to w&b
    # wandb.log({
    #     f'val/{k}': np.mean(v) for k, v in val_losses.items()
    # }, step=i_iter)
    # wandb.log({
    #     # 'val/pixel-ap': np.mean(pixel_aps),
    #     'val/sample-ap': np.mean(sample_ap),
    #     'val/sample-auroc': np.mean(sample_auroc),
    #     'val/pix-ap': results.setdefault('PixAP', 0),
    #     'val/best-dice': results.setdefault('BestDice', 0),
    #     'val/input images': wandb.Image(x.cpu()[:config.num_images_log]),
    #     # 'val/targets': wandb.Image(y.float().cpu()[:config.num_images_log]),
    #     'val/anomaly maps': wandb.Image(anomaly_map.cpu()[:config.num_images_log])
    # }, step=i_iter)


def test(model, test_loader, config):
    val_losses = defaultdict(list)
    labels = []
    masks = []
    anomaly_scores = []
    anomaly_maps = []
    img_names = []

    # for x, y, label in test_loader:
    for data_batch in test_loader:
        img_names.append(data_batch['name'][0])
        x, label = data_batch['img'], data_batch['label']
        # x, y, anomaly_map: [b, 1, h, w]
        # Compute loss, anomaly map and anomaly score
        loss_dict, anomaly_map, anomaly_score = val_step(model, x, config.device)

        # Compute metrics
        # pixel_ap = evaluation.compute_average_precision(anomaly_map, y)

        for k, v in loss_dict.items():
            val_losses[k].append(v.item())
        # pixel_aps.append(pixel_ap)
        labels.append(label)
        anomaly_scores.append(anomaly_score.detach().cpu())
        anomaly_maps.append(anomaly_map.detach().cpu())

        if config.train_dataset == 'brats':
            masks.append(data_batch['mask'])
        #
        # i_val_step += 1
        # if i_val_step >= config.val_steps:
        #     break

    # Compute sample-wise average precision and AUROC over all validation steps
    labels = torch.cat(labels)
    anomaly_scores = torch.cat(anomaly_scores)
    sample_ap = evaluation.compute_average_precision(anomaly_scores, labels)
    sample_auroc = evaluation.compute_auroc(anomaly_scores, labels)

    results = {'auc': sample_auroc, 'ap': sample_ap}

    score_maps = torch.cat(anomaly_maps, dim=0)
    if config.train_dataset == 'brats':
        masks = torch.cat(masks, dim=0).unsqueeze(1).numpy()  # # NxHxW -> Nx1xHxW
        pix_ap = evaluation.compute_average_precision(score_maps.numpy(), masks)
        best_dice, best_thresh = evaluation.compute_best_dice(score_maps.numpy(), masks)
        results.update({'PixAP': pix_ap, 'BestDice': best_dice, 'BestThresh': best_thresh})

        np.save("output/score.npy", score_maps.numpy())
        np.save("output/true.npy", masks)

    # Model Cost
    example_in = torch.zeros((1, 3, config.image_size, config.image_size)).cuda()
    flops, params = profile(model, inputs=(example_in,))
    flops, params = round(flops * 1e-6, 4), round(params * 1e-6, 4)  # 1e6 = M
    flops, params = str(flops) + "M", str(params) + "M"

    results.update({"FLOPs": flops, "params": params})
    # Print validation results
    print("\nTest results:")
    # log_msg = " - ".join([f'val {k}: {np.mean(v):.4f} - ' for k,
    #                      v in val_losses.items()])
    # log_msg += f"\npixel-wise average precision: {np.mean(pixel_aps):.4f}\n"
    # log_msg = " - ".join([f'val {k}: {np.mean(v):.4f} - ' for k, v in val_losses.items()])
    # log_msg += f"sample-wise AUROC: {sample_auroc:.4f} - "
    # log_msg += f"sample-wise average precision: {sample_ap:.4f} - "
    # log_msg += f"Average positive label: {labels.float().mean():.4f}\n"
    # print(log_msg)
    keys = list(results.keys())
    return keys, results

    # # Visualization
    # vis_dir = os.path.join("output", "vis", config.train_dataset)
    # if not os.path.exists(vis_dir):
    #     os.makedirs(vis_dir)

    # # clamp_max = torch.quantile(score_maps, 0.9999, interpolation="nearest")
    # # clamp_max = torch.quantile(score_maps, 0.999, interpolation="nearest")
    # # score_maps = torch.clamp(score_maps, min=0., max=clamp_max)
    # # score_maps = (score_maps - torch.min(score_maps)) / (torch.max(score_maps) - torch.min(score_maps))

    # for i in range(len(img_names)):
    #     img_name = img_names[i]
    #     anomaly_map = score_maps[i]
    #     anomaly_map = (anomaly_map - torch.min(anomaly_map)) / (torch.max(anomaly_map) - torch.min(anomaly_map))

    #     anomaly_map = transforms.ToPILImage()(anomaly_map)

    #     save_path = os.path.join(vis_dir, img_name + ".png")
    #     anomaly_map.save(save_path)

    # # # Log to tensorboard
    # # wandb.log({
    # #     f'val/{k}': np.mean(v) for k, v in val_losses.items()
    # # }, step=config.max_steps + 1)
    # # wandb.log({
    # #     # 'val/pixel-ap': pixel_ap,
    # #     'val/sample-ap': sample_ap,
    # #     'val/sample-auroc': sample_auroc,
    # #     # 'val/iou-at-5fpr': iou_at_5fpr,
    # #     # 'val/dice-at-5fpr': dice_at_5fpr,
    # #     'val/input images': wandb.Image(x.cpu()[:config.num_images_log]),
    # #     # 'val/targets': wandb.Image(y.float().cpu()[:config.num_images_log]),
    # #     # 'val/anomaly maps': wandb.Image(anomaly_map.cpu()[:config.num_images_log]),
    # # }, step=config.max_steps + 1)


@torch.no_grad()
def test_pitfalls(model, config):
    import random
    from glob import glob
    from tqdm import tqdm
    from functools import partial
    import torch.nn.functional as F
    # from fae.data.data_utils import load_files_to_ram, load_nii_nn
    # from fae.data.artificial_anomalies import sample_position, intensity_anomaly
    from data.data_utils import load_files_to_ram, load_nii_nn
    from data.artificial_anomalies import sample_position, intensity_anomaly
    files = glob('/datasets/MOOD/brain/test_raw/*.nii.gz')
    load_fn = partial(load_nii_nn, slice_range=(128, 129), size=config.image_size)
    imgs = load_files_to_ram(files, load_fn)
    imgs = np.stack([s for vol in imgs for s in vol], axis=0)

    radius = 10
    intensities = np.linspace(0., 1., num=100)

    ap_results = []
    for intensity in tqdm(intensities):
        aps = []
        random.seed(0)

        for img in imgs:
            position = sample_position(img)
            img_anomal, label = intensity_anomaly(img, position, radius, intensity)

            img_ = torch.tensor(img[None]).to(config.device)
            img_anomal_ = torch.tensor(img_anomal[None]).to(config.device)

            # Experiment 3.1
            feats, rec = model(img_)
            feats_anomal, rec_anomal = model(img_anomal_)
            pred = model.loss_fn(rec, feats_anomal).mean(1, keepdim=True)
            pred = F.interpolate(pred, img_.shape[-2:], mode='bilinear',
                                 align_corners=True)
            pred = pred[0, 0].cpu().numpy()

            # Experiment 3.2
            # pred = model.predict_anomaly(img_anomal_)[0][0].detach().cpu().numpy()

            # Compute average precision
            ap = evaluation.compute_average_precision(pred, label)
            aps.append(ap)

        ap_results.append(np.mean(aps))
        print(f'Intensity: {intensity:.4f} - AP: {ap_results[-1]:.4f}')

    ap_results = np.array(ap_results)
    np.save('ex3_1_mood_aps.npy', ap_results)


if __name__ == '__main__':
    # Training
    if config.train:
        train(model, optimizer, train_loader, val_loader, config)

    # test_pitfalls(model, config)
    if not config.train:
        for epoch in range(100, 10001, 100):
            print("Loading model from checkpoint of epoch", epoch)
            model.load(config, epoch=epoch)

            # Testing
            print('Testing...')
            keys, results = test(model, test_loader, config)
            keys_aucp, results_aucp = test(model, test_loader_aucp, config)

            for key in keys:
                print(key + ": {}".format(results[key]), end="  ")
            print()
            for key in keys_aucp:
                print(key + ": {}".format(results_aucp[key]), end="  ")

            with open(f"output/{config.train_dataset}_{config.loss_fn}_aucp_results.csv", "a") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(['epoch', 'AUC', 'AUCp', 'AP', 'AP (aucp)'])
                csvwriter.writerow([
                    epoch,
                    results['auc'],
                    results_aucp['auc'],
                    results['ap'],
                    results_aucp['ap']
                ])