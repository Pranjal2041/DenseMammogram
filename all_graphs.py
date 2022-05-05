import os
from os.path import join
from merge_predictions import get_image_dict, apply_merge
from froc_by_pranjal import calc_froc_from_dict, pretty_print_fps
import numpy as np 
import matplotlib.pyplot as plt


OUT_DIR = 'euro_results_auto'
numbers_dir = os.path.join(OUT_DIR, 'numbers')
graphs_dir = os.path.join(OUT_DIR, 'graphs')

BASE_FOLDER = '../bilateral_new/MammoDatasets'

MIN_CLIP_FPI = 0.02
def plot_froc(input_files, save_file, TITLE = 'FRCNN vs BILATERAL FROC', SHOW = False, CLIP_FPI = 1.2):
    for file in input_files:
        lines = open(file).readlines()
        x = np.array([float(line.split()[0]) for line in lines])
        y = np.array([float(line.split()[1]) for line in lines])
        y = y[x<CLIP_FPI]
        x = x[x<CLIP_FPI]
        y = y[MIN_CLIP_FPI<x]
        x = x[MIN_CLIP_FPI<x]
        plt.plot(x, y, label = input_files[file])
        plt.legend()

    plt.title(TITLE)
    plt.xlabel('Average False Positive Per Image')
    plt.ylabel('Sensitivity')

    if SHOW:
        plt.show()
    plt.savefig(save_file)
    plt.clf()


dsets = [('AIIMS_highres_reliable', 'AIIMS'), ('IRCHVal', 'IRCHVal')]
dsets = dsets[1:]
for dset in dsets:
    test_splits = ['test_2', 'test_dense', 'test_iso'][::-1]
    for test_split in test_splits:
        main_dataset = join(BASE_FOLDER, dset[0], test_split)

        contrast_datasets =  [join(BASE_FOLDER,f'{dset[1]}_C{i+1}',test_split) for i in range(4)]
        threshold_datasets = [join(BASE_FOLDER,f'{dset[1]}_T{i+1}',test_split) for i in range(2)]
        frcnn_preds = 'preds_frcnn_frcnn'
        contrast_preds = [
            'preds_frcnn_AIIMS_C1',
            'preds_frcnn_AIIMS_C2',
            'preds_frcnn_AIIMS_C3',
            'preds_frcnn_AIIMS_C4',
        ]
        bilateral_preds = 'preds_bilateral_BILATERAL'
        threshold_preds = [
            'preds_frcnn_AIIMS_T1',
            'preds_frcnn_AIIMS_T2',
        ]

        input_files = []
        dataset_paths = [join(main_dataset,  '{0}', frcnn_preds)]
        dataset_paths +=[join(dset, '{0}', preds) for (dset,preds) in zip(contrast_datasets, contrast_preds)]
        dataset_paths +=[join(dset, '{0}', preds) for (dset,preds) in zip(threshold_datasets, threshold_preds)]
        dataset_paths +=[join(main_dataset, '{0}', bilateral_preds)]


        CONFIGS = {
            'Baseline' : ('Baseline Model', [0]),
            'Bilateral' : ('Bilateral Model', [7]),
            'Contrast' : ('CABD Model', [0,1,2,3,4]),
            'Threshold' : ('TI Model', [0,5,6]),
            'Proposed' : ('Proposed Model', [1,2,3,4,5,6,7])
        }

        # Now handle the directories
        num_dir = os.path.join(numbers_dir, dset[1], test_split)
        os.makedirs(num_dir, exist_ok=True)


        for config in CONFIGS:
            title = CONFIGS[config][0]
            allowed = CONFIGS[config][1]

            weight_map = {
                0 : 1.,
                1 : 1,
                2 : 1.,
                3 : 1.,
                4 : .5, # C4
                5 : 0.5,
                6 : 0.5,
                7 : 1
            }

            weights = [weight_map[x] for x in allowed]
            
            # generate the required mp dicts
            def c2_manp(preds):
                preds  = list(filter(lambda x: x[0]>0.85,preds)) # keep preds lower than 0.6 confidence
                return preds

            def c3_manp(preds):
                preds  = list(filter(lambda x: x[0]>0.85,preds)) # keep preds lower than 0.6 confidence
                return preds

            def t1_manp(preds):
                preds  = list(filter(lambda x: x[0]>0.6,preds)) # keep preds lower than 0.6 confidence
                return preds

            t2_manp = t1_manp
            mp_dict = {
                f'{dset[1]}_C2' : c2_manp,
                f'{dset[1]}_C3' : c3_manp,
                f'{dset[1]}_T1' : t1_manp,
                f'{dset[1]}_T2' : t2_manp,
                f'{dset[1]}_C4' : c3_manp
            }

            image_dict = get_image_dict(dataset_paths, allowed = allowed, USE_ACR = False, acr_cat = None, mp_dict = mp_dict)
            image_dict = apply_merge(image_dict, METHOD = 'nms', weights= weights, conf_type='absent_model_aware_avg')
            

            senses, fps = calc_froc_from_dict(image_dict, fps_req = [0.025,0.05,0.1,0.15,0.2,0.3,1.], save_to = os.path.join(num_dir, f'{title}.txt'))


        # Lets plot now

        GRAPHS = [
            ('Bilateral','Baseline'),
            ('Contrast','Baseline'),
            ('Threshold','Baseline'),
            ('Proposed','Baseline'),
            ('Proposed', 'Bilateral'),
            ('Proposed', 'Contrast'),
            ('Proposed', 'Threshold'),
        ]


        # Now handle the directories
        graph_dir = os.path.join(graphs_dir, dset[1], test_split)
        os.makedirs(graph_dir, exist_ok=True)

        for graph in GRAPHS:
            if graph[0] not in CONFIGS or graph[1] not in CONFIGS: continue
            file_name1 = f'{CONFIGS[graph[0]][0]}.txt'
            file_name2 = f'{CONFIGS[graph[1]][0]}.txt'

            title1 = CONFIGS[graph[0]][0]
            title2 = CONFIGS[graph[1]][0]

            plot_froc({
                join(num_dir, file_name1): title1,
                join(num_dir, file_name2) : title2,
            }, join(graph_dir,f'{title1}_vs_{title2}.png'),f'{title1} vs {title2} FROC', CLIP_FPI = 0.3 if dset[0] == 'IRCHVal' else 0.8)


