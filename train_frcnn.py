import torch
import math
from advanced_logger import LogPriority
from dataloaders import get_FRCNN_dataloaders
from models import get_FRCNN_model
from detection.engine import evaluate_loss, evaluate_simplified, train_one_epoch_simplified, evaluate_simplified

def main(cfg, experimenter):

    LR = cfg['LR']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    NUM_EPOCHS = cfg['NUM_EPOCHS']
    BATCH_SIZE = cfg['BATCH_SIZE']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model =  get_FRCNN_model().to(device)
    train_loader, val_loader = get_FRCNN_dataloaders(experimenter.config, batch_size=BATCH_SIZE, data_dir = experimenter.config['DATA_DIR'])
    optimizer = torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=WEIGHT_DECAY)

    for epoch in range(NUM_EPOCHS):
        experimenter.start_epoch()
        train_one_epoch_simplified(model, optimizer, train_loader, device, epoch, experimenter = experimenter)
        evaluate_simplified(model, val_loader, device=device, experimenter = experimenter)
        loss = evaluate_loss(model, device, val_loader, experimenter = experimenter)
        experimenter.log('Validation Loss: {}'.format(loss), priority = LogPriority.MEDIUM)
        experimenter.end_epoch(loss, model = model, device = device)
        experimenter.save_model(model)
    experimenter.generate_predictions(model, device)

if __name__ == '__main__':
    from experimenter import Experimenter
    cfg_file = 'configs/AIIMS_C1.cfg'
    experimenter = Experimenter(cfg_file)
    main(experimenter.config['FRCNN'], experimenter)