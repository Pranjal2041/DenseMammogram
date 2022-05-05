import os
import torch
from os.path import join
from model_utils import generate_predictions, generate_predictions_bilateral
from models import get_FRCNN_model, Bilateral_model
from froc_by_pranjal import get_froc_points
from auc_by_pranjal import get_auc_score

####### PARAMETERS TO ADJUST #######
exp_name = 'BILATERAL'
OUT_FILE = 'aiims_full_test_results/bil_complete.txt'
BILATERAL = True
dataset_path = 'AIIMS_highres_reliable/test_2'
####################################




if os.path.split(OUT_FILE)[0]:
    os.makedirs(os.path.split(OUT_FILE)[0], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frcnn_model =  get_FRCNN_model().to(device)

if BILATERAL:
    model = Bilateral_model(frcnn_model).to(device)
    MODEL_PATH = f'experiments/{exp_name}/bilateral_models/bilateral_model.pth'
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    model = frcnn_model
    MODEL_PATH = f'experiments/{exp_name}/frcnn_models/frcnn_model.pth'
    model.load_state_dict(torch.load(MODEL_PATH))


test_path = join('../bilateral_new', 'MammoDatasets',dataset_path)


def get_aiims_dict(test_path, corr_file):    
    extract_file = lambda x: x[x.find('test_2/')+7:]
    corr_dict = {extract_file(line.split()[0].replace('"','')):extract_file(line.split()[1].replace('"','')) for line in open(corr_file).readlines()}
    corr_dict = {join(test_path,k):join(test_path,v) for k,v in corr_dict.items()}
    return corr_dict

if BILATERAL:
    pred_dir = f'preds_bilateral_{exp_name}'
    generate_predictions_bilateral(model,device,test_path, get_aiims_dict(test_path, '../bilateral_new/corr_lists/aiims_corr_list_with_val_full_test.txt'),'aiims',pred_dir)
else:
    pred_dir = f'preds_frcnn_{exp_name}'
    generate_predictions(model, device, test_path, preds_folder = pred_dir)


file = open(OUT_FILE, 'a')
file.writelines(f'{exp_name}  FROC Score:\n')
senses, fps = get_froc_points(pred_dir, root_fol= test_path, fps_req = [0.025,0.05,0.1,0.15,0.2,0.3,1.0,1.5])
for s,f in zip(senses, fps):
    print(f'Sensitivty at {f}: {s}')
    file.writelines(f'Sensitivty at {f}: {s}\n')
file.close()

print('AUC Score:',get_auc_score(pred_dir, test_path, retAcc = True, acc_thresh = 1.))

