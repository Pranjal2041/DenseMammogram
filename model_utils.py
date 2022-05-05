import os
import torchvision.transforms as T
import cv2
from tqdm import tqdm
import detection.transforms as transforms
from dataloaders import get_direction

def generate_predictions_bilateral(model,device,testpath_,cor_dict,dset='aiims',preds_folder='preds_new'):
    transform = T.Compose([T.ToPILImage(),T.ToTensor()])
    model.eval()
    for label in ['mal','ben']:
        testpath = os.path.join(testpath_,label)
        # testpath = os.path.join(dataset_path,'Training', 'train',label)
        testimg = os.path.join(testpath, 'images')

        #preds_folder  = 'preds_new'
        os.makedirs(os.path.join(testpath, preds_folder),exist_ok=True)

        if not os.path.exists(os.path.join(testpath,preds_folder)):
            os.makedirs(os.path.join(testpath+preds_folder),exist_ok = True)

        for file in tqdm(os.listdir(testimg)):
            img1 = cv2.imread(os.path.join(testimg,file))
            img1 = transform(img1)
            # if False:
            if(os.path.join(testimg,file) in cor_dict and os.path.isfile(cor_dict[os.path.join(testimg,file)])):
                print('Using Bilateral')
                img2 = cv2.imread(cor_dict[os.path.join(testimg,file)])
                img2 = transform(img2)
                if(get_direction(dset,file)==1):
                    img1,_ = transforms.RandomHorizontalFlip(1.0)(img1)
                    
                    images = [img1.to(device),img2.to(device)]
                    output = model([images])[0]
                    img1,output = transforms.RandomHorizontalFlip(1.0)(img1,output)
                else:
                    img2,_ = transforms.RandomHorizontalFlip(1.0)(img2)
                    
                    images = [img1.to(device),img2.to(device)]
                    output = model([images])[0]
            else:
                print('Using FRCNN')
                output = model.frcnn([img1.to(device)])[0]
            #output = model.frcnn([img1.to(device)])[0]
            boxes = output['boxes']
            scores = output['scores']
            labels = output['labels']
            f = open(os.path.join(testpath,preds_folder,file[:-4]+'.txt'),'w')
            for i in range(len(boxes)):
                box = boxes[i].detach().cpu().numpy()
                #f.write('{} {} {} {} {} {}\n'.format(scores[i].item(),labels[i].item(),box[0],box[1],box[2],box[3]))
                f.write('{} {} {} {} {}\n'.format(scores[i].item(),box[0],box[1],box[2],box[3]))


def generate_predictions(model,device,testpath_,preds_folder='preds_frcnn'):
    transform = T.Compose([T.ToPILImage(),T.ToTensor()])
    model.eval()
    for label in ['mal','ben']:
        testpath = os.path.join(testpath_,label)
        # testpath = os.path.join(dataset_path,'Training', 'train',label)
        testimg = os.path.join(testpath, 'images')

        #preds_folder  = 'preds_new'
        os.makedirs(os.path.join(testpath, preds_folder),exist_ok=True)

        if not os.path.exists(os.path.join(testpath,preds_folder)):
            os.makedirs(os.path.join(testpath+preds_folder),exist_ok = True)

        for file in tqdm(os.listdir(testimg)):
            im = cv2.imread(os.path.join(testimg,file))
            if file == 'Mass-Training_P_00444_LEFT_CC.png':
                print('Test this')
                continue
            im = transform(im)

            output = model([im.to(device)])[0]
            boxes = output['boxes'] #/ FAC
            scores = output['scores']
            labels = output['labels']
            f = open(os.path.join(testpath,preds_folder,file[:-4]+'.txt'),'w')
            for i in range(len(boxes)):
                box = boxes[i].detach().cpu().numpy()
                f.write('{} {} {} {} {}\n'.format(scores[i].item(),box[0],box[1],box[2],box[3]))
