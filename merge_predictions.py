import os
import glob
import torch
from os.path import join
import numpy as np
from froc_by_pranjal import file_to_bbox, calc_froc_from_dict, pretty_print_fps
import sys
from ensemble_boxes import *
import json
import pickle



get_file_id = lambda x: x.split('_')[1]
get_acr_cat = lambda x: '0' if x not in acr_cat else acr_cat[x]
cat_to_idx = {'a':1,'b':2,'c':3,'d':4}


def get_image_dict(dataset_paths, labels = ['mal','ben'], allowed = [], USE_ACR = False, acr_cat = None, mp_dict = None):
    image_dict = dict()
    if allowed == []:
        allowed = [i for i in range(len(dataset_paths))]
    for label in labels:
        images = list(set.intersection(*map(set, [os.listdir(dset.format(label)) for dset in dataset_paths])))
        for image in images:
            if USE_ACR:
                acr = get_acr_cat(get_file_id(image))
            # print(acr, image)
            key = image[:-4]
            gts = []
            preds = []
            for i,dset in enumerate(dataset_paths):
                if i not in allowed:
                    continue
                if USE_ACR:
                    if dset.find('AIIMS_C')!=-1:
                        if acr == '0': continue
                        if dset.find(f'AIIMS_C{cat_to_idx[acr]}') == -1:
                            continue
                        # Now choose dset to be the acr category one
                        dset = dset.replace('/test',f'/test_{acr}')
                # print('ds',dset)
                pred_file = join(dset.format(label), key+'.txt')
                gt_file = join(os.path.split(dset.format(label))[0],'gt', key+'.txt')
                if label == 'mal':
                    gts.append(file_to_bbox(gt_file))
                else:
                    gts.append([])

                # TODO: Note this
                flag = False
                for mp in mp_dict:
                    if dataset_paths[i].find(mp) != -1:
                        preds.append(mp_dict[mp](file_to_bbox(pred_file)))
                        flag = True
                        break
                if not flag:
                    preds.append(file_to_bbox(pred_file))

            # Ensure all gts are same
            gt = gts[0]
            for g in gts[1:]:
                assert g == gt
                gt = g
            
            # Flatten Preds
            preds = [np.array(p) for p in preds]
            preds = [np.array([[0.,0.,0.,0.,0.]]) if pred.shape==(0,) else pred for pred in preds]
            preds = [np.vstack((p, np.zeros((100 - len(p), 5)))) for p in preds]
            image_dict[key] = dict()
            image_dict[key]['gt'] = gts[0]
            image_dict[key]['preds'] = preds
    return image_dict


def apply_merge(image_dict, METHOD = 'wbf', weights = None, conf_type = None):
    FACTOR = 5000
    fusion_func = weighted_boxes_fusion if METHOD == 'wbf' else non_maximum_weighted
    for key in image_dict:
        preds = np.array(image_dict[key]['preds'])
        if len(preds) != 0:
            boxes_list = [pred[:,1:]/FACTOR for pred in preds]
            scores_list = [pred[:,0] for pred in preds]
            labels = [[0. for _ in range(len(p))] for p in preds]
            if weights is None:
                weights = [1 for _ in range(len(preds))]
            if METHOD == 'wbf' and conf_type is not None:
                boxes,scores,_ = fusion_func(boxes_list, scores_list, labels, weights = weights,iou_thr  = 0.5, conf_type = conf_type)
            else:
                boxes,scores,_ = fusion_func(boxes_list, scores_list, labels, weights = weights,iou_thr  = 0.5,)
            preds_t = [[scores[i],FACTOR*boxes[i][0],FACTOR*boxes[i][1],FACTOR*boxes[i][2],FACTOR*boxes[i][3]] for i in range(len(boxes))]
            image_dict[key]['preds'] = preds_t
    return image_dict

def manipulate_preds(preds):
    return preds



def manipulate_preds_4(preds):
    return preds

tot = 0
def manipulate_preds_t1(preds): #return manipulate_preds(preds)
    preds  = list(filter(lambda x: x[0]>0.6,preds))

    return preds

def manipulate_preds_t2(preds): return manipulate_preds_t1(preds)


if __name__ == '__main__':
    USE_ACR = False
    dataset_paths = [
        'MammoDatasets/AIIMS_C1/test/{0}/preds_frcnn_AIIMS_C1',
        'MammoDatasets/AIIMS_C2/test/{0}/preds_frcnn_AIIMS_C2',
        'MammoDatasets/AIIMS_C3/test/{0}/preds_frcnn_AIIMS_C3',
        'MammoDatasets/AIIMS_C4/test/{0}/preds_frcnn_AIIMS_C4',
        'MammoDatasets/AIIMS_highres_reliable/test/{0}/preds_bilateral_BILATERAL',
        'MammoDatasets/AIIMS_highres_reliable/test/{0}/preds_frcnn_16',
    ]


    st = int(sys.argv[1])
    end = len(dataset_paths) - int(sys.argv[2])
    allowed = [i for i in range(st,end)]
    allowed = [0,1,2,3,4,5]

    OUT_FILE = 'contrast_frcnn.txt'
    if OUT_FILE is not None:
        fol = os.path.split(OUT_FILE)[0]
        if fol != '':
            os.makedirs(fol, exist_ok=True)
    
    acr_cat = json.load(open('aiims_categories.json','r'))
    print(allowed)

    mp_dict = {
        'preds_frcnn_AIIMS_C3': manipulate_preds,
        'preds_frcnn_AIIMS_C4': manipulate_preds_4,
        'AIIMS_T2': manipulate_preds_t2,
        'AIIMS_T1': manipulate_preds_t1,
    }

    image_dict = get_image_dict(dataset_paths, allowed = allowed, USE_ACR = USE_ACR, acr_cat = acr_cat, mp_dict = mp_dict)
    
    image_dict = apply_merge(image_dict, METHOD = 'nms') # or wbf

    if OUT_FILE:
        pickle.dump(image_dict, open(OUT_FILE.replace('.txt','.pkl'),'wb'))
        senses, fps = calc_froc_from_dict(image_dict, fps_req = [0.025,0.05,0.1,0.15,0.2,0.3,1.],save_to=OUT_FILE)
    pretty_print_fps(senses, fps)
