from configs import cfg
cfg.resume = True  
from core.utils.log_util import Logger
from core.data import create_dataloader
from core.nets import create_network
from core.train import create_trainer, create_optimizer


def main():
    log = Logger()
    log.print_config()

    model = create_network()
    optimizer = create_optimizer(model)
    trainer = create_trainer(model, optimizer)
    train_loader = create_dataloader('train')
    model.load_smpl(train_loader.dataset.avg_beta)
    trainer.iter = 9999
    trainer.progress()

if __name__ == '__main__':
    main()
