import logging
import time

from utils.ae_worker import AEWorker
from utils.util import AverageMeter
import os
import torch


class AEUWorker(AEWorker):
    def __init__(self, opt):
        super(AEUWorker, self).__init__(opt)
        self.logger = logging.getLogger(__name__)
        from aucp.paths import output_root
        log_dir = output_root() / "reconstruction" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(log_dir / f'log_{self.opt.dataset}_{self.opt.model["name"]}.log'),
            level=logging.INFO,
        )

    def train_epoch(self):
        self.net.train()
        losses, recon_losses, log_vars = AverageMeter(), AverageMeter(), AverageMeter()
        for idx_batch, data_batch in enumerate(self.train_loader):
            img = data_batch['img']
            img = img.cuda()

            net_out = self.net(img)

            loss, recon_loss, log_var = self.criterion(img, net_out)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            bs = img.size(0)
            losses.update(loss.item(), bs)
            recon_losses.update(recon_loss, bs)
            log_vars.update(log_var, bs)
        return losses.avg, recon_losses.avg, log_vars.avg

    def run_train(self):
        num_epochs = self.opt.train['epochs']
        print("=> Initial learning rate: {:g}".format(self.opt.train['lr']))
        t0 = time.time()
        lowest_val_loss = float('inf')
        lowest_val_loss_epoch = -1
        lowest_loss_AUC=0
        for epoch in range(1, num_epochs + 1):
            train_loss, recon_loss, log_var = self.train_epoch()
            self.logger.info(f"Epoch {epoch}: train/loss={train_loss}, train/recon_loss={recon_loss}, train/log_var={log_var}")
            
            if epoch == 1 or epoch % self.opt.train['eval_freq'] == 0:
                eval_results = self.evaluate()

                t = time.time() - t0
                print("Epoch[{:3d}/{:3d}]  Time:{:.1f}s  loss:{:.5f}  recon_loss:{:.5f}  log_var:{:.5f}".format(
                    epoch, num_epochs, t, train_loss, recon_loss, log_var), end="  |  ")

                keys = list(eval_results.keys())
                for key in keys:
                    print(key+": {:.4f}".format(eval_results[key]), end="  ")
                    eval_results["val/"+key] = eval_results.pop(key)
                print()
                if train_loss < lowest_val_loss:
                    lowest_val_loss = train_loss
                    lowest_val_loss_epoch = epoch
                    lowest_loss_AUC=eval_results["val/AUC"]
                self.logger.info(f"Epoch {epoch}: eval_results={eval_results}")
                t0 = time.time()
                torch.save(self.net.state_dict(), os.path.join(self.opt.train['save_dir'], "checkpoints", "epoch_{}.pt".format(epoch)))
        print(f"Lowest val loss: {lowest_val_loss:.5f} at epoch {lowest_val_loss_epoch} with AUC: {lowest_loss_AUC:.5f}")
        self.logger.info(f"Lowest val loss: {lowest_val_loss:.5f} at epoch {lowest_val_loss_epoch} with AUC: {lowest_loss_AUC:.5f}")
        self.save_checkpoint()
        #self.logger.info("Training finished.")