from train import get_method
from options import Options


def main():
    opt = Options(isTrain=False)
    opt.parse()
    # opt.save_options()

    worker = get_method(opt)
    worker.set_gpu_device()
    worker.set_seed()
    worker.set_network_loss()
    worker.set_logging(test=True)
    # worker.set_dataloader(test=True)

    for epoch in range(1, opt.epochs[opt.dataset] + 1):
        print("=> Evaluating epoch {}".format(epoch))
        worker.set_test_loader()
        worker.load_checkpoint(epoch=epoch)
        worker.run_eval(epoch=epoch, aucp=False)

        worker.set_test_loader_aucp()
        worker.run_eval(epoch=epoch, aucp=True)

        # worker.data_rept()


if __name__ == "__main__":
    main()
