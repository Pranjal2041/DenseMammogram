import matplotlib.pyplot as plt
import numpy as np

####### PARAMETERS TO ADJUST #######

# Specify the files generated from merge_nms and plot corresponding graphs
base_fol = 'normal_test'
input_files = {
    f'thresh_uni.txt' : 'Thresh + Uni',
    f'thresh_nouni.txt' : 'Thresh + NoUni',
}
save_file = 'uni_vs_nouni.png'
# TITLE = 'Thresh + Contrast + Bilateral vs Contrast + Bilateral FROC Comparison (Normal Test)'
TITLE = 'Uni vs NoUni FROC Comparison (Normal Test)'

SHOW = False
CLIP_FPI = 1.2
MIN_CLIP_FPI = 0.0
####################################

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
    plt.ylabel('Sensetivity')

    if SHOW:
        plt.show()
    plt.savefig(save_file)
    plt.clf()

if __name__ == '__main__':
    plot_froc(input_files, save_file, TITLE, SHOW, CLIP_FPI)