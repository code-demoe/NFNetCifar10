import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, CIFAR10
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
import time, datetime
from io import StringIO
import argparse
from tqdm.notebook import tqdm
from csv import writer
import timm
from timm.models.layers import get_act_layer, GELU
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.utils import adaptive_clip_grad, AverageMeter
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

from RandAugment import RandAugment

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

parser = argparse.ArgumentParser(description='PyTorch NfNet CIFAR10 Training')
parser.add_argument('--mname', default='resnet50', type=str, help='model name')
parser.add_argument('--act', default='gelu', type=str, help='activation layer: \
                    silu,swish,mish,relu,relu6,leaky_relu,elu,celu,selu,gelu,\
                    sigmoid,tanh,hard_sigmoid,hard_mish,hard_swish')
parser.add_argument('--clipthresh', default=0.08, type=float, help='Clipping threshold')
parser.add_argument('--aug', default=True, type=bool, help='Augmentation')
parser.add_argument('--pretrained', default=False, type=bool, help='Imagenet pretrained')
parser.add_argument('--ran', default=4, type=float, help='RandAugment N')
parser.add_argument('--ram', default=5, type=float, help='RandAugment M')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')


# Optimizer parameters
parser.add_argument('--opt', default='nesterov', type=str,help='Optimizer nesterov,\
                    adam,adabelief,adamw,nadam,radam,adamp,sgdp,adadelta,adafactor,\
                    adahession,rmsprop,rmsproptf,novograd,nvnovograd')
parser.add_argument('--opt-eps', default=None, type=float,
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.00002,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--lr', default=0.01, type=float, help='leraning rate')

def split_indices(n, val_pct=0.2,seed=99):
    n_val=int(val_pct*n)
    np.random.seed(seed)
    idxs=np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]


def convert_act(act_layer,model):
    for child_name, child in model.named_children():
        if isinstance(child, GELU):
            if str(child).endswith('(inplace=True)'):
                setattr(model, child_name, act_layer(inplace=True))
            else:
                setattr(model, child_name, act_layer())
        else:
            convert_act(act_layer,child)
    return model

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

#Plotting graphs
def plot_accuracies(history,save_dir):
    accuracies1 = [x['val_acc_top1'] for x in history]
    accuracies2=[x['val_acc_top2'] for x in history]
    sns.set_style("white")
    plot1=plt.figure(1)
    plt.plot(accuracies1)
    plt.plot(accuracies2)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['top1', 'top2'])
    plt.title('Accuracy vs. No. of epochs');
    plt.savefig(save_dir+'/accuracy.png',bbox_inches='tight',dpi=300)

def plot_losses(history,save_dir):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    sns.set_style("white")
    plot2=plt.figure(2)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');
    plt.savefig(save_dir+'/loss.png',bbox_inches='tight',dpi=300)

def plot_lrs(history,save_dir):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    sns.set_style("white")
    plot3=plt.figure(3)
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');
    plt.savefig(save_dir+'/lrs.png',bbox_inches='tight',dpi=300)

def plot_confusion_matrix(cm, classes,save_dir,normalize=False,title=None,cmap=plt.get_cmap('Wistia'),n=4):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.set_style("white")
    plot4=plt.figure(n)
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')


    #Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_dir+'/'+title+'.png',bbox_inches='tight',dpi=300)
    return ax

def plot_roc(fpr,tpr,class_names,save_dir,n=6):
    sns.set_style("white")
    plot6=plt.figure(n)
    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i],label=class_names[i])
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig(save_dir+'/roc_curves.png',bbox_inches='tight',dpi=300)   

def plot_pvr(prec, recall, class_names, save_dir, n=8):
    sns.set_style("white")
    plot7=plt.figure(n)
    for i in range(len(class_names)):
        plt.plot(recall[i],prec[i], label=class_names[i])
    plt.title('Multiclass Precision vs recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.savefig(save_dir+'/pvr_curves.png',bbox_inches='tight',dpi=300)

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, y_pred = output.topk(k=maxk, dim=1)
        y_pred = y_pred.t()
        target_reshaped = target.view(1, -1).expand_as(y_pred)
        correct = (y_pred == target_reshaped)
        list_topk_accs = []
        for k in topk:
            ind_which_topk_matched_truth = correct[:k]
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)
            topk_acc = tot_correct_topk / batch_size
            list_topk_accs.append(topk_acc)
        return list_topk_accs

def validation_step(model, batch):
    images, labels = batch
    out = model(images) # Generate predictions
    loss = F.cross_entropy(out, labels)   # Calculate loss
    acc1,acc2 = accuracy(out, labels,topk=(1,2))           # Calculate accuracy
    return {'val_loss': loss.detach(), 'val_acc_top1': acc1,'val_acc_top2': acc2}

def validation_epoch_end(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs_top1 = [x['val_acc_top1'] for x in outputs]
    batch_accs_top2 = [x['val_acc_top2'] for x in outputs]
    epoch_acc_top1 = torch.stack(batch_accs_top1).mean()      # Combine accuracies
    epoch_acc_top2 = torch.stack(batch_accs_top2).mean()      # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc_top1': epoch_acc_top1.item(),
            'val_acc_top2': epoch_acc_top2.item()}


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [validation_step(model,batch) for batch in val_loader]
    return validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_cycle(start_epoch,epochs, max_lr, model, train_loader, val_loader,optimizer,aug,save_dir):
    torch.cuda.empty_cache()
    history = []
    global batch_size
    df=pd.DataFrame(columns=['epoch','train_loss','val_loss','val_acc_top1',
    'val_acc_top2','batch_time','data_time'])
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))
    for epoch in range(start_epoch,epochs):
        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()

        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        end = time.time()
        with tqdm(train_loader,unit='batch') as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                data_time_m.update(time.time() - end)
                images, labels = batch
                if aug:
                    mixup_args=dict(
                    mixup_alpha=0.2, cutmix_alpha=0.5, cutmix_minmax=None,
                    prob=0.5, switch_prob=0.5, mode='batch',
                    label_smoothing=0.1, num_classes=10
                    )
                    mixup_fn = Mixup(**mixup_args)
                    input, target = mixup_fn(images, labels)
                    loss_fn=SoftTargetCrossEntropy()
                    out=model(input)
                    loss=loss_fn(out,target)
                else:
                    out = model(images)                  # Generate predictions
                    loss = F.cross_entropy(out, labels) #grad_fn=<NllLossBackward>
                optimizer.zero_grad()
                loss.backward()
                train_losses.append(loss)

                #Agc
                adaptive_clip_grad([p for p in model.parameters()][:-2], clip_factor=0.08, eps=1e-3, norm_type=2.0)
                optimizer.step()
                batch_time_m.update(time.time() - end)
                lrs.append(get_lr(optimizer))
                sched.step()
                end = time.time()
                tepoch.set_postfix({'batch_time':batch_time_m.val,'data_time':data_time_m.val})
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs

        df=df.append({'epoch':epoch+1,'train_loss':result['train_loss'],
        'val_loss':result['val_loss'],'val_acc_top1':result['val_acc_top1'],
        'val_acc_top2':result['val_acc_top2'],
        'batch_time':batch_time_m.avg,'data_time':data_time_m.avg},ignore_index=True)
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f},batch_time:{batch_time.avg:.3f}s, data_time:{data_time.avg:.3f}s".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc_top1'],
            batch_time=batch_time_m,data_time=data_time_m))
        history.append(result)
        # if epoch!=start_epoch and epoch%10==0:
        #     print('==>Saving checkpoint..')
        #     torch.save({'net':model.state_dict(),'epoch':epoch},save_dir+'/checkpoint'+str(epoch)+'.pth')
    df.to_csv(save_dir+'/Summary.csv')
    return history

def main(mname='dm_nfnet_f0',act='gelu',opt='nesterov',aug=True,pretrained=False,ram=5,epochs=2,resume=0):
    time_start=datetime.datetime.now()
    file_dir=os.path.dirname(os.path.realpath(__file__))
    save_dir=file_dir+'/'
    args = parser.parse_args('')
    global batch_size
    batch_size=args.bs
    args.mname=mname
    args.act=act
    args.opt=opt
    args.aug=aug
    a='na'
    args.pretrained=pretrained
    p='np'
    args.ram=ram
    args.resume=resume
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.aug:
        # Add RandAugment with N, M(hyperparameter)
        transform_train.transforms.insert(0, RandAugment(args.ran, args.ram))
        a='a'
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    #Defining the class labels
    class_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(class_names)
    
    # The data directory
    data_dir = './data/cifar10'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    train_dataset = CIFAR10(root=data_dir, train=True,download=True, transform=transform_train)
    valid_dataset = CIFAR10(root=data_dir, train=True,download=True, transform=transform_test)
    test_dataset = CIFAR10(root=data_dir, train=False,download=True, transform=transform_test)

    train_indices, val_indices=split_indices(len(train_dataset))
    train_sampler = SubsetRandomSampler(train_indices,generator=torch.Generator().manual_seed(42))
    val_sampler = SubsetRandomSampler(val_indices,generator=torch.Generator().manual_seed(42))

    # PyTorch data loaders
    train_dl = DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
    valid_dl = DataLoader(valid_dataset, batch_size, sampler=val_sampler, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size,shuffle=False,num_workers=2,pin_memory=True)
    
    #model
    print('==> Building model..')
    net=timm.create_model(args.mname,pretrained=args.pretrained,num_classes=10)
    act_layer=get_act_layer(args.act)
    net=convert_act(act_layer, net)
    if args.pretrained:
        p='p'
        for n,m in net.named_parameters():
            if not n.startswith(('stages.3','final_conv','head')):
                m.requires_grad=False
                
    save_dir=save_dir+'experiments/'+args.mname+'_'+args.act+'_'+args.opt+'_'+a+'_'+p
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(save_dir+'/checkpoint'+args.resume+'.pth')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']

    optimizer = create_optimizer_v2(net, **optimizer_kwargs(cfg=args))

    device = get_default_device()
    print('Using device:',device)
    
    torch.cuda.empty_cache()
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)
    model=to_device(net,device)

    epochs = epochs
    max_lr = 0.01

    history = [evaluate(model, valid_dl)]
    print('Without Training:',history)
    
    train_start_time = datetime.datetime.now()
    history += fit_one_cycle(start_epoch,epochs, max_lr, model, train_dl, valid_dl,
    optimizer = optimizer,aug=args.aug,save_dir=save_dir)
    train_end_time = datetime.datetime.now()
    diff = train_end_time - train_start_time
    diff_seconds = int(diff.total_seconds())
    minute_seconds, seconds = divmod(diff_seconds, 60)
    hours, minutes = divmod(minute_seconds, 60)
    train_time = f"{hours}h {minutes}m {seconds}s"
    
    print('==>Testing on test dataset..')
    test_dl = DeviceDataLoader(test_dl, device)
    model.eval()
    test_batch_losses=[]
    test_batch_acc1s=[]
    test_batch_acc2s=[]
    test_batch_outs=[]
    test_labels = []
    for batch in test_dl:
        images, labels = batch
        test_batch_out = model(images) # Generate predictions
        test_batch_loss = F.cross_entropy(test_batch_out, labels)
        test_batch_losses.append(test_batch_loss.detach())
        test_batch_acc1,test_batch_acc2 = accuracy(test_batch_out, labels,topk=(1,2))
        test_batch_acc1s.append(test_batch_acc1)
        test_batch_acc2s.append(test_batch_acc2)
        test_batch_outs.extend([x for x in test_batch_out.detach().cpu().numpy()])
        test_labels.extend([x for x in labels.detach().cpu().numpy()])
    test_loss = torch.stack(test_batch_losses).mean()
    test_acc1 = torch.stack(test_batch_acc1s).mean()
    test_acc2 = torch.stack(test_batch_acc2s).mean()
    test_out = np.stack(test_batch_outs,axis=0)
    y_true = np.stack(test_labels,axis=0)
    y_pred = np.argmax(test_out,axis=1)
    
    #Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    crep = classification_report(y_true,y_pred)
    print('\n',crep,'\n')
    crep = crep.splitlines()
    crep_lists = []
    for l in crep:
        if l == '':
            continue
        else:
            l = l.split(' ')
            l = [i for i in l if i]
            if l[0] == 'precision':
                l.insert(0,'Class Index')
            crep_lists.append(l)

    # roc curve for classes
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    prec = dict()
    recall = dict()
    pvr_auc = dict()
    
    for i in range(len(class_names)):    
        fpr[i], tpr[i], _ = roc_curve(y_true, test_out[:,i], pos_label=i)
        prec[i], recall[i], _ = precision_recall_curve(y_true, test_out[:,i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
        pvr_auc[i] = auc(recall[i],prec[i])
        print(class_names[i],roc_auc[i],pvr_auc[i])

    print('\nTest_loss:',test_loss.item(),'Test_acc:',test_acc1.item(),test_acc2.item())
    
    print('\n==>Plotting Graphs..')
    plot_accuracies(history,save_dir)
    plot_losses(history,save_dir)
    plot_lrs(history,save_dir)
    # Plotting non-normalized confusion matrix
    plot_confusion_matrix(cm, class_names, save_dir, title='Confusion matrix')
    #Plotting normalized confusion matrix
    plot_confusion_matrix(cm, class_names, save_dir, normalize = True, title = 'Normalized confusion matrix',n=5)
    plot_roc(fpr=fpr,tpr=tpr,class_names=class_names,n=7,save_dir=save_dir)
    plot_pvr(prec=prec, recall = recall, class_names= class_names, n=8, save_dir=save_dir )
    
    time_end=datetime.datetime.now()
    diff = time_end - time_start
    diff_seconds = int(diff.total_seconds())
    minute_seconds, seconds = divmod(diff_seconds, 60)
    hours, minutes = divmod(minute_seconds, 60)
    total_time = f"{hours}h {minutes}m {seconds}s"
    print('==>Saving csv file..')
    with open(save_dir+'/Summary.csv','a+',newline='') as f:
        csv_writer=writer(f)
        csv_writer.writerow([])
        csv_writer.writerow(['Test Accuracy top1',test_acc1.item(),'Test Accuracy top2',test_acc2.item(),'Test_loss',test_loss.item()])
        csv_writer.writerow([])
        for l in crep_lists:
            csv_writer.writerow(l)
        csv_writer.writerow([])
        csv_writer.writerow(['Class Name', 'ROC AUC', 'Precision vs Recall AUC'])
        for i in range(len(class_names)):
            csv_writer.writerow([class_names[i],roc_auc[i], pvr_auc[i]])
        csv_writer.writerow([])
        csv_writer.writerow(['Model_name',args.mname])
        csv_writer.writerow(['Activation',args.act])
        csv_writer.writerow(['Optimizer',args.opt])
        csv_writer.writerow(['Pretrained',args.pretrained])
        csv_writer.writerow(['Augmentation',args.aug])
        csv_writer.writerow([])
        csv_writer.writerow(['Training time',train_time])
        csv_writer.writerow(['Total time',total_time])

if __name__=='__main__':
    main()
