import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from logging_utils.dir_manage import get_directories
from torch.utils.tensorboard import SummaryWriter
from logging_utils.tbtools import AverageMeter, ProgressMeter
import wandb
from data.data_generation import *
from model.progress_c import *
from model.progress_r import *
from sklearn.datasets import make_moons,make_circles
from scipy import sparse
from hypergrad.hypernet import Hypernet
from hypergrad.hypernet_prob import HypernetProb

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:,i] = (x[:,i] - self.mean[i])/self.std[i]
        return x

class FNNet_regression(nn.Module):
# =============================================================================
#     Single Layer fully connected layer 
# =============================================================================
    def __init__(self, input_dim,  output_dim=1):
        super(FNNet_regression, self).__init__()

        self.input_dim = input_dim
        self.fc= nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class FNNet_classification(nn.Module):
# =============================================================================
#     Single Layer fully connected layer 
# =============================================================================
    def __init__(self, input_dim,output_dim=2):
        super(FNNet_classification, self).__init__()

        self.input_dim = input_dim
        self.fc= nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x
    
class simplified_NAM(nn.Module):
# =============================================================================
#     Single Layer fully connected layer 
# =============================================================================
    def __init__(self, input_dim, interm_dim=16, output_dim=1):
        super(simplified_NAM, self).__init__()

        self.input_dim = input_dim
        self.fc= nn.Linear(interm_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, output_dim, maxpool=True, base_hid=32):
        super(ConvNet, self).__init__()
        self.base_hid = base_hid
        self.conv1 = nn.Conv2d(1, base_hid, 5, 1)
        self.dp1 = torch.nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(base_hid, base_hid*2, 5, 1)
        self.dp2 = torch.nn.Dropout(0.5)
        self.fc1 = nn.Linear(4 * 4 * base_hid*2, base_hid*4)
        self.dp3 = torch.nn.Dropout(0.5)
        self.fc2 = nn.Linear(base_hid*4, output_dim)
        self.maxpool = maxpool

    def forward(self, x, return_feat=False):
        x = self.embed(x)
        out = self.fc2(x)
        if return_feat:
            return out, x.detach()
        return out

    def embed(self, x):
        x = F.relu(self.dp1(self.conv1(x)))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.dp2(self.conv2(x)))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 2*self.base_hid)
        x = F.relu(self.dp3(self.fc1(x)))
        return x

class ConvNetNoDropout(nn.Module):
    def __init__(self, output_dim, maxpool=True, ifnormalize=False, base_hid=32):
        super(ConvNetNoDropout, self).__init__()
        self.base_hid = base_hid
        self.conv1 = nn.Conv2d(1, base_hid, 5, 1)
        self.conv2 = nn.Conv2d(base_hid, base_hid*2, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * base_hid*2, base_hid*4)
        self.fc2 = nn.Linear(base_hid*4, output_dim)
        self.maxpool = maxpool
        self.ifnormalize = ifnormalize
        self.normalize = Normalize((0.1307,), (0.3081,))

    def forward(self, x, return_feat=False):
        if self.ifnormalize:
            x = self.normalize(x)
        x = self.embed(x)
        out = self.fc2(x)
        if return_feat:
            return out, x.detach()
        return out

    def embed(self, x):
        x = F.relu(self.conv1(x))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 2*self.base_hid)
        x = F.relu(self.fc1(x))
        return x

class ConvNetHyper(Hypernet):
    def __init__(self, output_dim, maxpool=True, base_hid=32, *args, **kwargs):
        super(ConvNetHyper, self).__init__(*args, **kwargs)
        self.base_hid = base_hid
        self.conv1 = nn.Conv2d(1, base_hid, 5, 1)
        self.conv2 = nn.Conv2d(base_hid, base_hid*2, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * base_hid*2, base_hid*4)
        self.fc2 = nn.Linear(base_hid*4, output_dim)
        self.maxpool = maxpool
        self.num_params = len([p for p in self.parameters()])
        self.init_wdecay(self.weight_decay_type, self.weight_decay_init)


    def extract_feat(self, x):
        x = F.relu(self.conv1(x))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 2*self.base_hid)
        x = F.relu(self.fc1(x))
        return x

    def predict(self, feat):
        out = self.fc2(feat)
        return out

class ConvNetHyperProb(HypernetProb):
    def __init__(self, output_dim, maxpool=True, base_hid=32, *args, **kwargs):
        super(ConvNetHyperProb, self).__init__(*args, **kwargs)
        self.base_hid = base_hid
        self.conv1 = nn.Conv2d(1, base_hid, 5, 1)
        self.conv2 = nn.Conv2d(base_hid, base_hid*2, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * base_hid*2, base_hid*4)
        self.fc2 = nn.Linear(base_hid*4, output_dim)
        self.maxpool = maxpool
        self.num_params = len([p for p in self.parameters()])
        self.init_wdecay(self.weight_decay_type, self.weight_decay_init)


    def extract_feat(self, x):
        x = F.relu(self.conv1(x))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 2*self.base_hid)
        x = F.relu(self.fc1(x))
        return x

    def predict(self, feat):
        out = self.fc2(feat)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, input_dim=3, planes=64):
        super(ResNet, self).__init__()
        self.planes = planes

        self.conv1 = nn.Conv2d(input_dim, self.planes,  kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, self.planes, self.planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.planes, 2*self.planes, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 2*self.planes, 4*self.planes, num_blocks[2],stride=2)
        self.layer4 = self._make_layer(block, 4*self.planes, 8*self.planes, num_blocks[3], stride=2)
        self.linear = nn.Linear(8*self.planes * block.expansion, num_classes)

    def _make_layer(self, block, inplanes, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(inplanes, planes, stride))
            inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.embed(x)
        out = self.linear(out)
        return out

    def embed(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def ResNet18(input_dim=3, planes=64):
    return ResNet(BasicBlock, [2, 2, 2, 2], input_dim=input_dim, planes=planes)


class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size, ifbn=False):
        super(HiddenLayer, self).__init__()
        self.ifbn = ifbn
        self.fc = nn.Linear(input_size, output_size)
        if ifbn:
            self.bn = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()
        torch.nn.init.xavier_uniform(self.fc.weight)

    def forward(self, x):
        out = self.fc(x)
        if self.ifbn:
            out = self.bn(out)
        return self.relu(out)


class ScoreNet(nn.Module):
    def __init__(self,input=10, hidden=100, num_layers=1, ifbn=False, activation="sigmoid"):
        super(ScoreNet, self).__init__()
        # self.normalize = Normalize()
        self.activation = activation
        self.first_hidden_layer = HiddenLayer(input, hidden, ifbn)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden, hidden, ifbn) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden, 1)

    def forward(self, x):
        # x = self.normalize(x)
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        else:
            return torch.relu(x)

class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)

class LeNet(nn.Module):
    def __init__(self, input_dim=1, ifnormalize=True):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.ifnormalize = ifnormalize
        self.normalize = Normalize((0.1307,), (0.3081,))

    def forward(self, x):
        if self.ifnormalize:
            x = self.normalize(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out

class LogisticRegressionProb(HypernetProb):
    def __init__(self, input_dim, out_dim=10, *args, **kwargs):
        super(LogisticRegressionProb, self).__init__(*args, **kwargs)
        self.fc = torch.nn.Parameter(torch.zeros(input_dim, out_dim))
        self.num_params = len([p for p in self.parameters()])
        self.init_wdecay(self.weight_decay_type, self.weight_decay_init)

    def forward(self, x):
        out = x @ self.fc
        return out

class LogisticRegression(Hypernet):
    def __init__(self, input_dim, out_dim=10, *args, **kwargs):
        super(LogisticRegression, self).__init__(*args, **kwargs)
        self.fc = torch.nn.Parameter(torch.zeros(input_dim, out_dim))
        self.num_params = len([p for p in self.parameters()])
        self.init_wdecay(self.weight_decay_type, self.weight_decay_init)


    def forward(self, x):
        out = x @ self.fc
        return out
    
def demo_c():
    parser = argparse.ArgumentParser(description='Summary')
    parser.add_argument('--method', type=str, default="Semi-supervised Meta Additive Models")
    parser.add_argument('--coreset_size', default=2, type=int) 
    parser.add_argument('--true_size', default=2, type=int)
    parser.add_argument('--start_coreset_size', default=2, type=int)
    parser.add_argument('--train_epoch', default=1000, type=int)
    parser.add_argument('--max_outer_iter', default=1000, type=int)
    parser.add_argument('--runs_name', default="S2MAM", type=str)
    parser.add_argument('--model', default="LapSVM", choices=["LapSVM", "LapLSR"], type=str)
    parser.add_argument('--project', default="feature_masks", type=str)
    parser.add_argument('--ts', default=0.16, type=float)
    parser.add_argument('--te', default=0.6, type=float)
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument('--random', default=False,action="store_true")
    parser.add_argument('--outer_lr', default=5e-2, type=float)
    args = parser.parse_args()
    if args.wandb:
        wandb.init(project=args.project, name=args.runs_name, config=args)
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)

    X, Y = make_moons(n_samples=200, noise=0.01)
    Xt, Yt = make_moons(n_samples=200, noise=0.01)

    X=np.concatenate((X,np.random.normal(0, 100,(200,8))),axis=1)
    Xt=np.concatenate((Xt,np.random.normal(0, 100,(200,8))),axis=1)

    ind_0 = np.nonzero(Y == 0)[0]
    ind_1 = np.nonzero(Y == 1)[0]
    Y[ind_0] = -1

    ind_0t = np.nonzero(Yt == 0)[0]
    ind_1t = np.nonzero(Yt == 1)[0]
    Yt[ind_0t] = -1

    ind_l0=np.random.choice(ind_0,10,False)
    ind_u0=np.setdiff1d(ind_0,ind_l0)

    ind_l1 = np.random.choice(ind_1, 10, False)
    ind_u1 = np.setdiff1d(ind_1, ind_l1)

    Xl=np.vstack([X[ind_l0,:],X[ind_l1,:]])
    Yl=np.hstack([Y[ind_l0],Y[ind_l1]])
    Xu=np.vstack([X[ind_u0,:],X[ind_u1,:]])
    Xl = torch.tensor(Xl).cpu()
    Yl = torch.tensor(Yl).cpu()
    Xu = torch.tensor(Xu).cpu()
    Xt = torch.tensor(Xt).cpu()
    Yt = torch.tensor(Yt).cpu()

# =============================================================================
# >> Parameter:
# >> - distance_function: The distance function for building the graph. This Pamater is valid when neighbor_mode is None.
# >> - gamma_d: Kernel parameters related to distance_function.
# >> - neighbor_mode: The edge weight after constructing the graph model by k-nearest neighbors. There are two options 'connectivity' and 'distance', 'connectivity' returns a 0-1 matrix, and 'distance' returns a distance matrix.
# >> - n_neighbor: k value of k-nearest neighbors.
# >> - kernel_function: The kernel function corresponding to SVM.
# >> - gamma_k: The gamma parameter corresponding to kernel_function.
# >> - gamma_A: Penalty weight for function complexity.
# >> - gamma_I: Penalty weight for smoothness of data distribution.
# =============================================================================
    if args.model=="LapSVM":
        opt={'neighbor_mode':'connectivity',
             'n_neighbor'   : 10,
             't':            10,
             'kernel_function':rbf,
             'kernel_parameters':{'gamma':1},
             'gamma_A':0.001,
             'gamma_I':0.01}
        model_select = LapSVM(opt)
    elif args.model=="FNNet_classification":
        model_select = models.FNNet(100*5,  2).cuda()
        model_select.eval()
    if args.random:
        subnet = torch.zeros_like(X[0])
        subnet_indices = np.random.choice(list(range(28*28)), args.coreset_size, replace=False)
        subnet.flatten()[subnet_indices] = 1
        subnet = subnet.cuda()
    else:
        subnet = solve(model_select, Xl, Yl,Xu, Xt, Yt)
        torch.save(subnet, f"{ckpt_base_dir}/subnet.pt")

    test_loss_final = AverageMeter("TestLossFinal", ":.3f", write_avg=False)
    test_top1_final = AverageMeter("TestAcc@1Final", ":6.2f", write_avg=False)
    test_top5_final = AverageMeter("TestAcc@5Final", ":6.2f", write_avg=False)
    l = [test_loss_final, test_top1_final, test_top5_final]
    progress = ProgressMeter(args.max_outer_iter, l, prefix="final test")
    # X= X.squeeze()
    acc_mean = []
    for i in range(3):
        if args.model=="LapSVM":
            opt={'neighbor_mode':'connectivity',
                 'n_neighbor'   : 10,
                 't':            10,
                 'kernel_function':rbf,
                 'kernel_parameters':{'gamma':1},
                 'gamma_A':0.001,
                 'gamma_I':0.01}
            model_train = LapSVM(opt)
        else:
            model_train = LapLSR()

        best_acc1, best_acc5, best_acc1_m, best_acc5_m, best_train_acc1, best_train_acc5 = 0, 0, 0, 0, 0 ,0
        data_t, target_t = Xt.cuda(), Yt.cuda()
        if args.model != "convnet":
            data_t = data_t.squeeze()
        if args.model == "LapSVM":
            subnet_detached = subnet.expand(Xl.size(0),-1).detach()
            subnet_detachedu = subnet.expand(Xu.size(0),-1).detach()
            subnet_detached_test = subnet.expand(data_t.size(0),-1).detach()
        else:
            subnet_detached = subnet.expand(Xl.size(0),-1).detach()
            subnet_detachedu = subnet.expand(Xu.size(0),-1).detach()
            subnet_detached_test = subnet.expand(data_t.size(0),-1).detach()

        for epoch in range(0, args.train_epoch):
            print("train_epoch:",epoch)
            train_acc1, train_acc5, train_loss = train(model_train, Xl*subnet_detached,Yl,Xu*subnet_detachedu) #data*subnet_detached, target
            test_acc1, test_acc5, test_loss = test(model_train, data_t, target_t)
            test_acc1_m, test_acc5_m, test_loss_m = test(model_train, data_t.cpu()*subnet_detached_test, target_t.cpu())
            is_best = test_acc1 > best_acc1
            best_acc1 = max(test_acc1, best_acc1)
            best_acc5 = max(test_acc5, best_acc5)
            best_acc1_m = max(test_acc1_m, best_acc1_m)
            best_acc5_m = max(test_acc5_m, best_acc5_m)
            best_train_acc1 = max(train_acc1, best_train_acc1)
            best_train_acc5 = max(train_acc5, best_train_acc5)
            if epoch % 50 == 0 or epoch == args.train_epoch - 1:
                print(f"epoch {epoch}, test acc1 {test_acc1}, test acc5 {test_acc5}, test loss {test_loss}")
                print(f"epoch {epoch}, test acc1_m {test_acc1_m}, test acc5_m {test_acc5_m}, test loss_m {test_loss_m}")
                print(f"epoch {epoch}, train acc1 {train_acc1}, train acc5 {train_acc5}, train loss {train_loss}")
                print(f"best acc1: {best_acc1}, best acc5: {best_acc5}, best acc1_m: {best_acc1_m}, best acc5_m: {best_acc5_m}, best train acc1: {best_train_acc1}, best test acc5: {best_train_acc5}, ckpt_base_dir: {ckpt_base_dir}, log_base_dir: {log_base_dir}")
        acc_mean.append((best_acc1,best_acc1_m))
        print(acc_mean)
    print('Final mask：',subnet)
    print(run_base_dir, ckpt_base_dir, log_base_dir)
    
def demo_r():
    parser = argparse.ArgumentParser(description='Summary')
    parser.add_argument('--method', type=str, default="Semi-supervised Meta Additive Models")
# =============================================================================
#     Selected and true feature settings
# =============================================================================
    parser.add_argument('--coreset_size', default=1, type=int) 
    parser.add_argument('--true_size', default=1, type=int)
    parser.add_argument('--start_coreset_size', default=1, type=int)
    parser.add_argument('--train_epoch', default=1000, type=int)
    parser.add_argument('--max_outer_iter', default=1000, type=int)
    parser.add_argument('--runs_name', default="S2MAM", type=str)
    parser.add_argument('--model', default="LapRLS", choices=["LapSVM", "LapRLS"], type=str)
    parser.add_argument('--project', default="feature_masks", type=str)
    parser.add_argument('--ts', default=0.16, type=float)
    parser.add_argument('--te', default=0.6, type=float)
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument('--random', default=False,action="store_true")
    parser.add_argument('--outer_lr', default=5e-1, type=float)
    args = parser.parse_args()
    if args.wandb:
        wandb.init(project=args.project, name=args.runs_name, config=args)
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    Xl,Yl,Xu,Xt,Yt = regression_data_generation(data_type=0)
# =============================================================================
# >> Parameter:
# >> - distance_function: The distance function for building the graph. This Pamater is valid when neighbor_mode is None.
# >> - gamma_d: Kernel parameters related to distance_function.
# >> - neighbor_mode: The edge weight after constructing the graph model by k-nearest neighbors. There are two options 'connectivity' and 'distance', 'connectivity' returns a 0-1 matrix, and 'distance' returns a distance matrix.
# >> - n_neighbor: k value of k-nearest neighbors.
# >> - kernel_function: The kernel function corresponding to SVM.
# >> - gamma_k: The gamma parameter corresponding to kernel_function.
# >> - gamma_A: Penalty weight for function complexity.
# >> - gamma_I: Penalty weight for smoothness of data distribution.
# =============================================================================
    if args.model=="LapSVM":
        opt1={'neighbor_mode':'connectivity',
             'n_neighbor'   : 10,
             't':            10,
             'kernel_function':rbf,
             'kernel_parameters':{'gamma':1},
             'gamma_A':0.001,
             'gamma_I':0.01}
        model_select = LapSVM(opt1)
    elif args.model=="LapRLS":
        model_select = LapRLS(n_neighbors=10, bandwidth=0.001, lambda_k=0.0005, lambda_u=0.0001, solver='closed-form')
    if args.random:
        subnet = torch.zeros_like(Xl[0])
        subnet_indices = np.random.choice(list(range(28*28)), args.coreset_size, replace=False)
        subnet.flatten()[subnet_indices] = 1
        subnet = subnet.cuda()
    else:
        subnet = solve(model_select, Xl, Yl,Xu, Xt, Yt)
        torch.save(subnet, f"{ckpt_base_dir}/subnet.pt")

    test_loss_final = AverageMeter("TestLossFinal", ":.3f", write_avg=False)
    test_top1_final = AverageMeter("TestAcc@1Final", ":6.2f", write_avg=False)
    test_top5_final = AverageMeter("TestAcc@5Final", ":6.2f", write_avg=False)
    l = [test_loss_final, test_top1_final, test_top5_final]
    progress = ProgressMeter(args.max_outer_iter, l, prefix="final test")
    # X= X.squeeze()
    acc_mean = []
    for i in range(3):  
        if args.model=="LapSVM":
            opt1={'neighbor_mode':'connectivity',
                 'n_neighbor'   : 10,
                 't':            10,
                 'kernel_function':rbf,
                 'kernel_parameters':{'gamma':1},
                 'gamma_A':0.001,
                 'gamma_I':0.01}
            model_train = LapSVM(opt1)
        else:
            model = LapRLS(n_neighbors=10, bandwidth=0.2, lambda_k=0.00025, lambda_u=0.005, solver='closed-form')

        best_acc1, best_acc5, best_acc1_m, best_acc5_m, best_train_acc1, best_train_acc5 = 0, 0, 0, 0, 0 ,0
        data_t, target_t = Xt.cuda(), Yt.cuda()
        if args.model == "LapSVM":
            subnet_detached = subnet.expand(Xl.size(0),-1).detach()
            subnet_detachedu = subnet.expand(Xu.size(0),-1).detach()
            subnet_detached_test = subnet.expand(data_t.size(0),-1).detach()
        else:
            subnet_detached = subnet.expand(Xl.size(0),-1).detach()
            subnet_detachedu = subnet.expand(Xu.size(0),-1).detach()
            subnet_detached_test = subnet.expand(data_t.size(0),-1).detach()

        for epoch in range(0, args.train_epoch):
            print("train_epoch:",epoch)
            train_acc1, train_acc5, train_loss = train(model_train, Xl*subnet_detached,Yl,Xu*subnet_detachedu) #data*subnet_detached, target
            test_acc1, test_acc5, test_loss = test(model_train, data_t, target_t)
            test_acc1_m, test_acc5_m, test_loss_m = test(model_train, data_t.cpu()*subnet_detached_test, target_t.cpu())
            is_best = test_acc1 > best_acc1
            best_acc1 = max(test_acc1, best_acc1)
            best_acc5 = max(test_acc5, best_acc5)
            best_acc1_m = max(test_acc1_m, best_acc1_m)
            best_acc5_m = max(test_acc5_m, best_acc5_m)
            best_train_acc1 = max(train_acc1, best_train_acc1)
            best_train_acc5 = max(train_acc5, best_train_acc5)
            if epoch % 50 == 0 or epoch == args.train_epoch - 1:
                print(f"epoch {epoch}, test acc1 {test_acc1}, test acc5 {test_acc5}, test loss {test_loss}")
                print(f"epoch {epoch}, test acc1_m {test_acc1_m}, test acc5_m {test_acc5_m}, test loss_m {test_loss_m}")
                print(f"epoch {epoch}, train acc1 {train_acc1}, train acc5 {train_acc5}, train loss {train_loss}")
                print(f"best acc1: {best_acc1}, best acc5: {best_acc5}, best acc1_m: {best_acc1_m}, best acc5_m: {best_acc5_m}, best train acc1: {best_train_acc1}, best test acc5: {best_train_acc5}, ckpt_base_dir: {ckpt_base_dir}, log_base_dir: {log_base_dir}")
        acc_mean.append((best_acc1,best_acc1_m))
        print(acc_mean)
    print('Final mask：',subnet)
    print(run_base_dir, ckpt_base_dir, log_base_dir)
    