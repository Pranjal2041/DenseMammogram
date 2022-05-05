import torch
import math
from advanced_logger import LogPriority
from dataloaders import get_bilateral_dataloaders
from models import get_FRCNN_model, Bilateral_model
from detection.engine import evaluate_loss, train_one_epoch_simplified

def main(cfg, experimenter):

    LR = cfg['LR']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    NUM_EPOCHS = cfg['NUM_EPOCHS']
    BATCH_SIZE = cfg['BATCH_SIZE']


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frcnn_model =  get_FRCNN_model().to(device)
    frcnn_model.load_state_dict(torch.load(cfg['FRCNN_MODEL_PATH']))

    model = Bilateral_model(frcnn_model).to(device)

    train_loader, val_loader = get_bilateral_dataloaders(experimenter.config, batch_size = BATCH_SIZE, data_dir = experimenter.config['DATA_DIR'])
    
    if cfg["OPTIM"] == "SGD":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad , model.roi_heads.parameters()),lr=LR,momentum=0.9,weight_decay=WEIGHT_DECAY)
    elif cfg["OPTIM"] == "ADAM":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = LR, weight_decay = WEIGHT_DECAY)
    elif cfg["OPTIM"] == "ADAGRAD":
        optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.roi_heads.parameters()), lr = LR, weight_decay = WEIGHT_DECAY)
    for epoch in range(NUM_EPOCHS):
        experimenter.start_epoch()
        train_one_epoch_simplified(model, optimizer, train_loader, device, epoch, experimenter = experimenter,optimizer_backbone=None)
        loss = evaluate_loss(model, device, val_loader, experimenter = experimenter)
        experimenter.log('Validation Loss: {}'.format(loss), priority = LogPriority.MEDIUM)
        
        experimenter.end_epoch(loss, model, device)
        experimenter.save_model(model)
    experimenter.generate_predictions(model, device)


if __name__ == '__main__':
    from experimenter import Experimenter
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    cfg_file = 'configs/default.cfg'
    experimenter = Experimenter(cfg_file)
    main(experimenter.config['BILATERAL'], experimenter)