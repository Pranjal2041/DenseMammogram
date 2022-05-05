import os
from os.path import join
import glob
from sklearn.metrics import roc_auc_score, roc_curve
import sys

def file_to_score(file):
    try:
        content = open(file, 'r').readlines()
        st = 0
        if len(content) == 0:
            # Empty File Should Return []
            return 0.
        if content[0].split()[0].isalpha():
            st = 1
        return max([float(line.split()[st]) for line in content])
    except FileNotFoundError:
        print(f'No Corresponding Box Found for file {file}, using [] as preds')
        return []
    except Exception as e:
        print('Some Error',e)
        return []

# Create the image dict
def generate_image_dict(preds_folder_name='preds_42',
                        root_fol='/home/krithika_1/densebreeast_datasets/AIIMS_C1',
                        mal_path=None, ben_path=None, gt_path=None,
                        mal_img_path = None, ben_img_path = None
                        ):

    mal_path = join(root_fol, mal_path) if mal_path else join(
        root_fol, 'mal', preds_folder_name)
    ben_path = join(root_fol, ben_path) if ben_path else join(
        root_fol, 'ben', preds_folder_name)
    mal_img_path = join(root_fol, mal_img_path) if mal_img_path else join(
        root_fol, 'mal', 'images')
    ben_img_path = join(root_fol, ben_img_path) if ben_img_path else join(
        root_fol, 'ben', 'images')
    gt_path = join(root_fol, gt_path) if gt_path else join(
        root_fol, 'mal', 'gt')


    '''
        image_dict structure:
            'image_name(without txt/png)' : {'gt' : [[...]], 'preds' : score}
    '''
    image_dict = dict()

    # GT Might be sightly different from images, therefore we will index gts based on
    # the images folder instead.
    for file in os.listdir(mal_img_path):
    # for file in glob.glob(join(gt_path, '*.txt')):
        if not file.endswith('.png'):
            continue
        file = file[:-4] + '.txt'
        file = join(gt_path, file)
        key = os.path.split(file)[-1][:-4]
        image_dict[key] = dict()
        image_dict[key]['gt'] = 1.
        image_dict[key]['preds'] = 0.

    for file in glob.glob(join(mal_path, '*.txt')):
        key = os.path.split(file)[-1][:-4]
        assert key in image_dict
        image_dict[key]['preds'] = file_to_score(file)

    for file in os.listdir(ben_img_path):
    # for file in glob.glob(join(ben_path, '*.txt')):
        if not file.endswith('.png'):
            continue

        file = file[:-4] + '.txt'
        file = join(ben_path, file)
        key = os.path.split(file)[-1][:-4]
        # if key == 'Calc-Test_P_00353_LEFT_CC' or key == 'Calc-Training_P_00600_LEFT_CC':
        #     continue
        if key in image_dict:
            print(key)
            print('SHIT')
            continue
        # assert key not in image_dict
        image_dict[key] = dict()
        image_dict[key]['preds'] = file_to_score(file)
        image_dict[key]['gt'] = 0.
    return image_dict

def get_auc_score_from_imdict(image_dict):
    keys = list(image_dict.keys())
    y = [image_dict[k]['gt']for k in keys]    
    preds = [image_dict[k]['preds']for k in keys]
    return roc_auc_score(y, preds)

def get_accuracy_from_imdict(image_dict, thresh = 0.3):
    keys = list(image_dict.keys())
    ys = [image_dict[k]['gt']for k in keys]    
    preds = [image_dict[k]['preds']for k in keys]
    acc = 0
    for y,pred in zip(ys,preds):
        if pred < thresh and y == 0.:
            acc+=1
        elif pred > thresh and y == 1.:
            acc+=1
    return acc/len(preds)


def get_auc_score(preds_image_folder, root_fol, retAcc = False, acc_thresh = 0.3):
    im_dict = generate_image_dict(preds_image_folder, root_fol = root_fol)
    if retAcc:
        return get_auc_score_from_imdict(im_dict), get_accuracy_from_imdict(im_dict, acc_thresh)
    else:
        return get_auc_score_from_imdict(im_dict)

if __name__ == '__main__':
    seed = '42' if len(sys.argv)== 1 else sys.argv[1]

    root_fol = '../bilateral_new/MammoDatasets/AIIMS_highres_reliable/test'

    auc_score = get_auc_score(f'preds_{seed}',root_fol)
    print(f'ROC AUC Score: {auc_score}')
    