from turtle import backward
import torch
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from glob import glob
from metrics import *

class CLEAR10IMG(Dataset):
    """ Learning CLEAR10 """

    def __init__(self, root_dir, bucket, form="all", split_ratio=0.7, debug=False, transform=None):
        '''
        Args: 
            root_dir(str list): folder path of 11 images
            bucket(int): time bucket id
            form(str): all -> whole dataset; train -> train dataset; test -> test dataset
            split_ratio(float, optional): proportion of train images in dataset
            transform(optional): transformation
        '''
        self.root_dir = root_dir
        self.transform = transform
        self.bucket = bucket
        self.form = form
        self.input_folders = self.root_dir+"/"+str(bucket+1)
        self.img_paths = list(filter(lambda x: x.endswith(".jpg"), glob(self.input_folders + '/**',recursive=True)))
        
        # code classes by alphabetical order
        self.targets = [self.img_paths[idx][len(self.input_folders):].split("/")[1] for idx in range(len(self.img_paths))]
        classes_name = sorted(list(set(self.targets)))
        classes_code = range(len(classes_name))
        self.classes_mapping = dict(zip(classes_name,classes_code))
        self.targets = torch.Tensor([self.classes_mapping[x] for x in self.targets]).int()
        
        if debug == True:
            self.img_paths = self.img_paths[:25]
            self.targets = self.targets[:25]
        if form != "all":
            self.train_img_paths = set(random.sample(self.img_paths,int(len(self.img_paths)*split_ratio)))
            self.test_img_paths = list(set(self.img_paths) - self.train_img_paths) 
            self.train_img_paths = list(self.train_img_paths)
            if form == "train":
                self.targets = [self.train_img_paths[idx][len(self.input_folders):].split("/")[1] for idx in range(len(self.train_img_paths))]
                self.targets = torch.Tensor([self.classes_mapping[x] for x in self.targets]).int()
            else:
                self.targets = [self.test_img_paths[idx][len(self.input_folders):].split("/")[1] for idx in range(len(self.test_img_paths))]
                self.targets = torch.Tensor([self.classes_mapping[x] for x in self.targets]).int()

    def __len__(self): 
        if self.form == "all":
            return len(self.img_paths)
        elif self.form == "train":
            return len(self.train_img_paths)
        else:
            return len(self.test_img_paths)

    def __getitem__(self,idx):
        if self.form == "all":
            img = Image.open(self.img_paths[idx])
            if img.mode != "RGB":
                img = img.convert("RGB")
            label = self.img_paths[idx][len(self.input_folders):].split("/")[1] # exclude the first empty entry
        elif self.form == "train":
            img = Image.open(self.train_img_paths[idx])
            if img.mode != "RGB":
                img = img.convert("RGB")
            label = self.train_img_paths[idx][len(self.input_folders):].split("/")[1]
        else:
            img = Image.open(self.test_img_paths[idx])
            if img.mode != "RGB":
                img = img.convert("RGB")
            label = self.test_img_paths[idx][len(self.input_folders):].split("/")[1]
        sample = {'img': img, 'target': self.classes_mapping[label]}
        if self.transform is not None:
            sample['img'] = self.transform(sample['img'])
        return sample['img'], sample['target']


def train(train_loader, use_gpu):
    model = torchvision.models.squeezenet1_1(pretrained=True)
    model.num_classes = 10 + 1
    model.classifier[1].out_channels = model.num_classes
    if use_gpu:
        model.cuda()

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss, train_acc = [], []
    max_iters = 5
    for itr in range(max_iters):
        model.train()
        total_loss = 0
        total_acc = 0
        for xb, yb in train_loader:
            if use_gpu:
                xb, yb = Variable(xb.cuda()), Variable(yb.cuda())
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item() * xb.size(0)
            _, preds = torch.max(y_pred.data, 1)
            total_acc += torch.sum(preds == yb.data)
        train_loss.append(total_loss / len(train_loader.dataset))
        train_acc.append(total_acc / len(train_loader.dataset))
        
        print("Itr: {:02d}".format(itr + 1))
        print("Train loss: {:.2f} \t acc : {:.2f}".format(train_loss[-1], train_acc[-1]))

    return model


def test(model, test_loader, use_gpu):
    total_test_acc = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            if use_gpu:
                xb, yb = Variable(xb.cuda()), Variable(yb.cuda())
            y_pred = model(xb)
            _, preds = torch.max(y_pred.data, 1)
            total_test_acc += torch.sum(preds == yb.data)
    avg_test_acc = total_test_acc / len(test_loader.dataset)
    print('Test Accuracy: {:.2f}'.format(avg_test_acc))
    
    return avg_test_acc



if __name__ == "__main__":
    use_gpu = True
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    dtype = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),            
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    

    root_dir = "../data/CLEAR-10-PUBLIC/labeled_images"
    num_buckets = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    debug = False

    '''
    Make a dataloader for each bucket --> #bucket of models will be trained
    '''
    train_data = [CLEAR10IMG(root_dir, i, form="train", debug=debug, transform=transform) for i in range(num_buckets)]
    train_loaders = [DataLoader(train_data[i], batch_size=20, shuffle=True, num_workers=4) for i in range(len(train_data))]

    test_data = [CLEAR10IMG(root_dir, i, form="test", debug=debug, transform=transform) for i in range(num_buckets)]
    test_loaders = [DataLoader(test_data[i], shuffle=False, num_workers=4) for i in range(len(test_data))]

    # To be used in the streaming protocol
    streaming = [CLEAR10IMG(root_dir, i, form="all", debug=debug, transform=transform) for i in range(num_buckets)]

    '''
    Concatenate all datasets and make one single dataloader --> only one model will be trained
    Note: This will not be used in the evaluation below. 
    It is just served as the approach to get the upper bounded score (because this is equivalent to joint training)
    '''
    # train = [CLEAR10IMG(root_dir, i, form="train", debug=debug, transform=transform) for i in range(num_buckets)]
    # train = ConcatDataset(train)
    # train_loader = DataLoader(train, batch_size=20, shuffle=True, num_workers=4)

    # test = [CLEAR10IMG(root_dir, i, form="test", debug=debug, transform=transform) for i in range(num_buckets)]
    # test = ConcatDataset(test)
    # test_loader = DataLoader(test, shuffle=False, num_workers=4)

    '''
    Train (iid)
    '''
    iid_models = []
    for i, train_loader in enumerate(train_loaders):
        print('Training on Bucket %d with iid protocol' % i)
        iid_models.append(train(train_loader, use_gpu))

    '''
    Test (iid)
    '''
    iid_R = np.zeros((num_buckets,)*2)  # accuracy matrix
    for i, model in enumerate(iid_models):
        for j, test_loader in enumerate(test_loaders):
            print('Evaluate timestamp %d model on bucket %d' % (i, j))
            iid_R[i, j] = test(model, test_loader, use_gpu)

    '''
    Train (streaming)
    '''
    streaming_models = []
    train_stream = []
    for i in range(num_buckets):
        print('Training on Bucket %d with streaming protocol' % (i))
        train_stream.append(streaming[i])
        train_stream_loader = DataLoader(ConcatDataset(train_stream), shuffle=False, num_workers=4)
        print(len(train_stream_loader))
        streaming_models.append(train(train_stream_loader, use_gpu))

    '''
    Test (streaming)
    '''
    streaming_R = np.zeros((num_buckets,)*2)  # accuracy matrix
    for i, model in enumerate(streaming_models):
        for j in range(i+1, num_buckets):
            print('Evaluate timestamp %d model on bucket %d' % (i, j))
            test_stream_loader = DataLoader(streaming[j], shuffle=False, num_workers=4)
            streaming_R[i, j] = test(model, test_stream_loader, use_gpu)
    

    '''
    Evaluation
    '''
    evals = {}  # to be displayed as each column in the leaderboard
    
    # iid metrics: In-domain Acc, Next-domain Acc, Acc, BwT, FwT
    evals['in_domain'] = in_domain(iid_R)
    evals['next_domain_iid'] = next_domain(iid_R)
    evals['acc'] = accuracy(iid_R)
    evals['bwt'] = backward_transfer(iid_R)
    evals['fwt_iid'] = forward_transfer(iid_R)
    
    # streaming metrics: Next-domain Acc, FwT
    evals['next_domain_streaming'] = next_domain(streaming_R)
    evals['fwt_streaming'] = forward_transfer(streaming_R)
    
    # Final leaderboard score (weighted average)
    score = sum(evals.values) / len(evals)  # use the weight for the moment

    print("Metrics: ", evals)
    print("Score: ", score)
