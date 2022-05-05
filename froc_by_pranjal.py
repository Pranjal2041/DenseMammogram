import os
import glob
import sys
from os.path import join


'''
    Note: Anywhere empty boxes means [] and not [[]]
'''


def remove_true_positives(gts, preds):

    def true_positive(gt, pred):
        # If center of pred is inside the gt, it is a true positive
        c_pred = ((pred[0]+pred[2])/2., (pred[1]+pred[3])/2.)
        if (c_pred[0] >= gt[0] and c_pred[0] <= gt[2] and
                c_pred[1] >= gt[1] and c_pred[1] <= gt[3]):
            return True
        return False

    tps = 0
    fns = 0

    for gt in gts:
        # First check if any true positive exists
        # If more than one exists, do not include it in next set of preds
        add_tp = False
        new_preds = []
        for pred in preds:
            if true_positive(gt, pred):
                add_tp = True
            else:
                new_preds.append(pred)
        preds = new_preds
        if add_tp:
            tps += 1
        else:
            fns += 1
    return preds, tps, fns



def calc_metric_single(gts, preds, threshold,):
    '''
        Returns fp, tp, tn, fn
    '''
    preds = list(filter(lambda x: x[0] >= threshold, preds))
    preds = [pred[1:] for pred in preds]  # Remove the scores

    if len(gts) == 0:
        return len(preds), 0, 1 if len(preds) == 0 else 0, 0
    preds, tps, fns = remove_true_positives(gts, preds)
    # All remaining will have to fps
    fps = len(preds)
    return fps, tps, 0, fns


def calc_metrics_at_thresh(im_dict, threshold):
    '''
        Returns fp, tp, tn, fn
    '''
    fps, tps, tns, fns = 0, 0, 0, 0
    for key in im_dict:
        fp,tp,tn,fn = calc_metric_single(im_dict[key]['gt'],
                           im_dict[key]['preds'], threshold)
        fps+=fp
        tps+=tp
        tns+=tn
        fns+=fn

    return fps, tps, tns, fns

from joblib import Parallel, delayed

def calc_metrics(inp):
    im_dict, tr = inp
    out = dict()
    for t in tr:
        fp, tp, tn, fn = calc_metrics_at_thresh(im_dict, t)
        out[t] = [fp, tp, tn, fn]
    return out
    

def calc_froc_from_dict(im_dict, fps_req = [0.025,0.05,0.1,0.15,0.2,0.3], save_to = None):

    num_images = len(im_dict)

    gap = 0.005
    n = int(1/gap)
    thresholds = [i * gap for i in range(n)]
    fps = [0 for _ in range(n)]
    tps = [0 for _ in range(n)]
    tns = [0 for _ in range(n)]
    fns = [0 for _ in range(n)]


    for i,t in enumerate(thresholds):
        fps[i], tps[i], tns[i], fns[i] = calc_metrics_at_thresh(im_dict, t)


    # Now calculate the sensitivities
    senses = []
    for t,f in zip(tps, fns):
        try: senses.append(t/(t+f))
        except: senses.append(0.)

    if save_to is not None:
        f = open(save_to, 'w')
        for fp,s in zip(fps, senses):
            f.write(f'{fp/num_images} {s}\n')
        f.close()

    senses_req = []
    for fp_req in fps_req:
        for i,f in enumerate(fps):
            if f/num_images < fp_req:
                if fp_req == 0.1:
                    print(fps[i], tps[i], tns[i], fns[i])
                    prec = tps[i]/(tps[i] + fps[i])
                    recall = tps[i]/(tps[i] + fns[i])
                    f1 = 2*prec*recall/(prec+recall)
                    spec = tns[i]/ (tns[i] + fps[i])
                    print(f'Specificity: {spec}')
                    print(f'Precision: {prec}')
                    print(f'Recall: {recall}')
                    print(f'F1: {f1}')
                senses_req.append(senses[i-1])
                break
    return senses_req, fps_req




def file_to_bbox(file_name):
    try:
        content = open(file_name, 'r').readlines()
        st = 0
        if len(content) == 0:
            # Empty File Should Return []
            return []
        if content[0].split()[0].isalpha():
            st = 1
        return [[float(x) for x in line.split()[st:]] for line in content]
    except FileNotFoundError:
        print(f'No Corresponding Box Found for file {file_name}, using [] as preds')
        return []
    except Exception as e:
        print('Some Error',e)
        return []

def generate_image_dict(preds_folder_name='preds_42',
                        root_fol='/home/pranjal/densebreeast_datasets/AIIMS_C1',
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
            'image_name(without txt/png)' : {'gt' : [[...]], 'preds' : [[]]}
    '''
    image_dict = dict()

    # GT Might be sightly different from images, therefore we will index gts based on
    # the images folder instead.
    for file in os.listdir(mal_img_path):
        if not file.endswith('.png'):
            continue
        file = file[:-4] + '.txt'
        file = join(gt_path, file)
        key = os.path.split(file)[-1][:-4]
        image_dict[key] = dict()
        image_dict[key]['gt'] = file_to_bbox(file)
        image_dict[key]['preds'] = []

    for file in glob.glob(join(mal_path, '*.txt')):
        key = os.path.split(file)[-1][:-4]
        assert key in image_dict
        image_dict[key]['preds'] = file_to_bbox(file)

    for file in os.listdir(ben_img_path):
        if not file.endswith('.png'):
            continue

        file = file[:-4] + '.txt'
        file = join(ben_path, file)
        key = os.path.split(file)[-1][:-4]
        if key == 'Calc-Test_P_00353_LEFT_CC' or key == 'Calc-Training_P_00600_LEFT_CC': # Corrupt Files in Dataset
            continue
        if key in image_dict:
            print(key)
        # assert key not in image_dict
        if key in image_dict:
            print(f'Unexpected Error. {key} exists in multiple splits')
            continue
        image_dict[key] = dict()
        image_dict[key]['preds'] = file_to_bbox(file)
        image_dict[key]['gt'] = []
    return image_dict


def pretty_print_fps(senses,fps):
    for s,f in zip(senses,fps):
        print(f'Sensitivty at {f}: {s}')

def get_froc_points(preds_image_folder, root_fol, fps_req = [0.025,0.05,0.1,0.15,0.2,0.3], save_to = None):
    im_dict = generate_image_dict(preds_image_folder, root_fol = root_fol)
    # print(im_dict)
    print(len(im_dict))
    senses, fps = calc_froc_from_dict(im_dict, fps_req, save_to = save_to)
    return senses, fps

if __name__ == '__main__':
    seed = '42' if len(sys.argv)== 1 else sys.argv[1]

    root_fol = '../bilateral_new/MammoDatasets/AIIMS_highres_reliable/test_2'

    if len(sys.argv) <= 2:
        save_to = None
    else:
        save_to = sys.argv[2]
    senses, fps = get_froc_points(f'preds_{seed}',root_fol, save_to = save_to)

    pretty_print_fps(senses, fps)
