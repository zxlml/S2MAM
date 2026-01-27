import gc
import wandb
import copy
import torch
import sklearn
import argparse
import numpy as np
import torch.nn.functional as F
from logging_utils.dir_manage import get_directories
from torch.utils.tensorboard import SummaryWriter
from varname import nameof
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from scipy.optimize import minimize
from scipy import sparse
import scipy.optimize as sco
from itertools import cycle, islice
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MNIST summary generator')
parser.add_argument('--method', type=str, default="probability_1step")
parser.add_argument('--coreset_size', default=1, type=int)
parser.add_argument('--true_size', default=1, type=int)
parser.add_argument('--start_coreset_size', default=2, type=int)
parser.add_argument('--K', default=10, type=int)
parser.add_argument('--outer_lr', default=5e-3, type=float)
parser.add_argument('--max_outer_iter', default=1000, type=int)
parser.add_argument('--runs_name', default="ours", type=str)
parser.add_argument('--model', default="LapRLS", choices=["LapSVM", "LapRLS"], type=str)
parser.add_argument('--project', default="pixel_selection", type=str)
parser.add_argument("--iterative", default=False, action="store_true")
parser.add_argument("--clip_grad", default=False, action="store_true")
parser.add_argument("--print_score", default=False, action="store_true")
parser.add_argument('--ts', default=0.16, type=float)
parser.add_argument('--te', default=0.6, type=float)
parser.add_argument('--clip_constant', default=5, type=float)
parser.add_argument('--test_freq', default=1000, type=int)
parser.add_argument('--wandb', action="store_true")
parser.add_argument('--random', action="store_true")
parser.add_argument("--vr", default=False, action="store_true")

args = parser.parse_args()
if args.wandb:
    wandb.init(project=args.project, name=args.runs_name, config=args)
run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)


torch.set_printoptions(precision=6,sci_mode=False)
def debug_memory():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

def name_print(s):
    print (nameof(s), ":", s)

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

class LapRLS(object):

    def __init__(self, n_neighbors, bandwidth, lambda_k, lambda_u,
                 learning_rate=1e-5, n_iterations=500, solver='closed-form'):
        """
        Laplacian Regularized Least Square algorithm

        Parameters
        ----------
        n_neighbors : integer
            Number of neighbors to use when constructing the graph
        lambda_k : float
        lambda_u : float
        Learning_rate: float
            Learning rate of the gradient descent
        n_iterations : integer
        solver : string ('closed-form' or 'gradient-descent' or 'L-BFGS-B')
            The method to use when solving optimization problem
        """
        self.n_neighbors = n_neighbors
        self.bandwidth = bandwidth
        self.lambda_k = lambda_k
        self.lambda_u = lambda_u
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.solver = solver
        

    def fit(self, X, Y,X_no_label):
        """
        Fit the model
        
        Parameters
        ----------
        X : ndarray shape (n_labeled_samples, n_features)
            Labeled data
        X_no_label : ndarray shape (n_unlabeled_samples, n_features)
            Unlabeled data
        Y : ndarray shape (n_labeled_samples,)
            Labels
        """

        l = X.shape[0]
        u = X_no_label.shape[0]
        n = l + u
        

        self.X = np.concatenate([X, X_no_label], axis=0)

        try:
            self.Y = np.concatenate([Y, np.zeros(u).reshape(-1,1)])
        except:
            self.Y = np.concatenate([Y, np.zeros(u)])       



        W = kneighbors_graph(self.X, self.n_neighbors, mode='connectivity')
        W = (((W + W.T) > 0) * 1)

        L = np.diag(W.sum(axis=0)) - W

        K = rbf_kernel(self.X, gamma=self.bandwidth)

        J = np.diag(np.concatenate([np.ones(l), np.zeros(u)]))
        
        if self.solver == 'closed-form':

            final = (J.dot(K) + self.lambda_k * l * np.identity(l + u) + ((self.lambda_u * l) / (l + u) ** 2) * L.dot(K))

            self.alpha = np.linalg.inv(final).dot(self.Y)

            del self.Y, W, L, K, J
            
        elif self.solver == 'gradient-descent':
            """
            If solver is Gradient-descent then a learning rate and an iteration number must be provided
            """
            
            print('Performing gradient descent...')
            

            self.alpha = np.zeros(n)

            grad_part1 = -(2 / l) * K.dot(self.Y)
            grad_part2 = ((2 / l) * K.dot(J) + 2 * self.lambda_k * np.identity(l + u) + \
                        ((2 * self.lambda_u) / (l + u) ** 2) * K.dot(L)).dot(K)

            def RLS_grad(alpha):
                return np.squeeze(np.array(grad_part1 + grad_part2.dot(alpha)))
                        

            del self.Y, W, L, K, J
        
            for i in range(self.n_iterations + 1):
                

                self.alpha -= self.learning_rate * RLS_grad(self.alpha)
                
                if i % 50 == 0:
                    print("\r[%d / %d]" % (i, self.n_iterations) ,end = "")
                    
            print('\n')
        
        elif self.solver == 'L-BFGS-B':
            
            print('Performing L-BFGS-B', end='...')
            
            x0 = np.zeros(n)

            grad_part1 = -(2 / l) * K.dot(self.Y)
            grad_part2 = ((2 / l) * K.dot(J) + 2 * self.lambda_k * np.identity(l + u) + \
                        ((2 * self.lambda_u) / (l + u) ** 2) * K.dot(L)).dot(K)

            def RLS(alpha):
                return np.squeeze(np.array((1 / l) * (self.Y - J.dot(K).dot(alpha)).T.dot((self.Y - J.dot(K).dot(alpha))) \
                        + self.lambda_k * alpha.dot(K).dot(alpha) + (self.lambda_u / n ** 2) \
                        * alpha.dot(K).dot(L).dot(K).dot(alpha)))

            def RLS_grad(alpha):
                return np.squeeze(np.array(grad_part1 + grad_part2.dot(alpha)))
            
            self.alpha, _, _ = sco.fmin_l_bfgs_b(RLS, x0, RLS_grad, args=(), pgtol=1e-30, factr =1e-30)
            
            print('done')
                                    
        new_K = rbf_kernel(self.X, X, gamma=self.bandwidth)
        f = np.squeeze(np.array(self.alpha)).dot(new_K)

        """
        def to_minimize(b):
            predictions = np.array((f > b) * 1)
            return - (sum(predictions == Y) / len(predictions))

        bs = np.linspace(0, 1, num=101)
        res = np.array([to_minimize(b) for b in bs])
        self.b = bs[res == np.min(res)][0]
        """
        

    def predict(self, Xtest):
        """
        Parameters
        ----------
        Xtest : ndarray shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        predictions : ndarray shape (n_samples, )
            Predicted labels for Xtest
        """

        new_K = rbf_kernel(self.X, Xtest, gamma=self.bandwidth)
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        return f
    

    def accuracy(self, Xtest, Ytrue):
        """
        Parameters
        ----------
        Xtest : ndarray shape (n_samples, n_features)
            Test data
        Ytrue : ndarray shape (n_samples, )
            Test labels
        """
        predictions = self.predict(Xtest)
        mse = sklearn.metrics.mean_squared_error(predictions, Ytrue)
        print('MSE: {}'.format(mse))

        

class LapSVM(object):
    def __init__(self,opt):
        self.opt=opt
        self.Q=0
        
    def fit(self,X,Y,X_u):

        classes, y_indices = np.unique(Y, return_inverse=True)
        self.class_dict={classes[0]:-1,classes[1]:1}
        self.rev_class_dict = {-1:classes[0] ,  1:classes[1]}
        self.X=np.vstack([X,X_u])
        Y=np.diag(Y)
        if self.opt['neighbor_mode']=='connectivity':
            W = kneighbors_graph(self.X, self.opt['n_neighbor'], mode='connectivity',include_self=False)
            W = (((W + W.T) > 0) * 1)
        elif self.opt['neighbor_mode']=='distance':
            W = kneighbors_graph(self.X, self.opt['n_neighbor'], mode='distance',include_self=False)
            W = W.maximum(W.T)
            W = sparse.csr_matrix((np.exp(-W.data**2/4/self.opt['t']),W.indices,W.indptr),shape=(self.X.shape[0],self.X.shape[0]))
        else:
            raise Exception()


        L = sparse.diags(np.array(W.sum(0))[0]).tocsr() - W


        K = self.opt['kernel_function'](self.X,self.X,**self.opt['kernel_parameters'])

        l=X.shape[0]
        u=X_u.shape[0]

        J = np.concatenate([np.identity(l), np.zeros(l * u).reshape(l, u)], axis=1)
        
        almost_alpha = np.linalg.inv(2 * self.opt['gamma_A'] * np.identity(l + u) \
                                     + ((2 * self.opt['gamma_I']) / (l + u) ** 2) * L.dot(K)).dot(J.T).dot(Y)


        self.Q = Y.dot(J).dot(K).dot(almost_alpha)
        self.Q = (self.Q+self.Q.T)/2

        del W, L, K, J

        e = np.ones(l)
        q = -e
        
        def objective_func(beta):
            return (1 / 2) * beta.dot(self.Q).dot(beta) + q.dot(beta)

        def objective_grad(beta):
            return np.squeeze(np.array(beta.T.dot(self.Q) + q))

        bounds = [(0, 1 / l) for _ in range(l)]

        def constraint_func(beta):
            return beta.dot(np.diag(Y))

        def constraint_grad(beta):
            return np.diag(Y)

        cons = {'type': 'eq', 'fun': constraint_func, 'jac': constraint_grad}

        x0 = np.zeros(l)

        beta_hat = minimize(objective_func, x0, jac=objective_grad, constraints=cons, bounds=bounds,tol=0.0001)['x']

        self.alpha = almost_alpha.dot(beta_hat)

        del almost_alpha, self.Q

        new_K = self.opt['kernel_function'](self.X,X,**self.opt['kernel_parameters'])
        f = np.squeeze(np.array(self.alpha)).dot(new_K)

        self.sv_ind=np.nonzero((beta_hat>1e-7)*(beta_hat<(1/l-1e-7)))[0]
        try:
            ind=self.sv_ind[0]
        except:
            ind=0
        self.b=np.diag(Y)[ind]-f[ind]


    def decision_function(self,X):
        new_K = self.opt['kernel_function'](self.X, X, **self.opt['kernel_parameters'])
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        return f+self.b
    
    def predict_proba(self,X):
        y_desision = self.decision_function(X)
        y_score = np.full((X.shape[0], 2), 0, np.float64)
        y_score[:,0]=1/(1+np.exp(y_desision))
        y_score[:, 1] =1- y_score[:,0]
        return torch.tensor(y_score).cpu()

    def predict(self,X):
        y_desision = self.decision_function(X)
        y_pred = np.ones(X.shape[0])
        y_pred[y_desision < 0] = -1
        return torch.tensor(y_pred).cpu()
    
    def accuracy(self, X, Y):
        """
        Parameters
        ----------
        X : ndarray shape (n_samples, n_features)
            Test data
        Y : ndarray shape (n_samples, )
            Test labels
        """
        predictions = self.predict(X)
        accuracy = sum(predictions == Y) / len(predictions)
        print('Accuracy: {}%'.format(round(accuracy * 100, 2)))
        
    
def rbf(X1,X2,**kwargs):
    return np.exp(-cdist(X1,X2)**2*kwargs['gamma'])


def train_to_converge(model, xl_coreset, yl_coreset, xu_coreset):  #大改，换成现成的模型
    model_copy = copy.deepcopy(model)
    xl_coreset, yl_coreset, xu_coreset = xl_coreset.cpu(), yl_coreset.cpu(),xu_coreset.cpu()
    diverged = False
    acc1=0
    acc5=0
    model_copy.fit(xl_coreset.cpu(), yl_coreset.cpu(), xu_coreset.cpu())
    output = torch.tensor(model_copy.predict(xl_coreset.cpu())).cpu()
    if args.model=="LapSVM":
        loss = np.average(-1 * (yl_coreset*output - np.log(1+np.exp(output))))
    elif args.model=="LapRLS":
        loss = mean_squared_error(output, yl_coreset)
    return model_copy, loss, acc1, acc5, diverged


def get_loss_on_full_train(model, xl, yl):
    loss_avg = 0
    data, target = xl, yl
    
    if args.model=="LapSVM":
        output = torch.tensor(model.predict(data.cpu())).cpu()
        loss = np.average(-1 * (yl*output - np.log(1+np.exp(output))))
    elif args.model=="LapRLS":
        output = torch.tensor(model.predict(data.cpu())).cpu()
        loss = sklearn.metrics.mean_squared_error(output, yl)
    loss_avg += loss.item()
    return loss_avg

def get_loss_on_full_test(model, xlt, ylt):
    loss_avg = 0
    data, target = xlt, ylt
    
    if args.model=="LapSVM":
        output = torch.tensor(model.predict(data.cpu())).cpu()
        loss = np.average(-1 * (ylt*output - np.log(1+np.exp(output))))
    elif args.model=="LapRLS":
        output = torch.tensor(model.predict(data.cpu())).cpu()
        loss = sklearn.metrics.mean_squared_error(output, ylt)
    loss_avg += loss.item()
    return loss_avg


def calculateGrad_vr(scores, fn_list, grad_list, fn_avg):
    scores.grad = torch.zeros_like(scores)
    for i in range(args.K):
        scores.grad += torch.tensor(1 / (args.K-1) * (fn_list[i]-fn_avg) * grad_list[i])

def calculateGrad(scores, fn_list, grad_list):
    scores.grad = torch.zeros_like(scores)
    for i in range(args.K):
        scores.grad += torch.tensor(1/args.K*fn_list[i]*grad_list[i])

def solve(model, xl, yl, xu,x_test,y_test):
    if args.model=="LapSVM":
        xl= xl.squeeze()
        xu =xu.squeeze()
        x_test =  x_test.squeeze()
        num_elements = len(xl[0])
    elif args.model=="LapRLS":
        xl= xl.squeeze()
        xu =xu.squeeze()
        x_test =  x_test.squeeze()
        num_elements = len(xl[0])
    pr_target = args.coreset_size / num_elements
    prune_rate = pr_target
    ts = int(args.ts * args.max_outer_iter)
    te = int(args.te * args.max_outer_iter)
    pr_start = prune_rate if not args.iterative else args.start_coreset_size / num_elements
    scores = torch.full_like(xl[0], pr_start, dtype=torch.float, requires_grad=True, device="cpu") 
    scores_opt = torch.optim.SGD([scores], lr=args.outer_lr)
    scores.grad = torch.zeros_like(scores)
    for outer_iter in range(args.max_outer_iter):
        if args.iterative:
            if outer_iter < ts:
                prune_rate = pr_start
            elif outer_iter < te:
                prune_rate = pr_target + (pr_start - pr_target) * (1 - (outer_iter - ts) / (te - ts)) ** 3
            else:
                prune_rate = pr_target
        args.coreset_size = prune_rate * num_elements
        assign_learning_rate(scores_opt, 0.5 * (1 + np.cos(np.pi * outer_iter / args.max_outer_iter)) * args.outer_lr)
        fn_list = []
        grad_list = []
        fn_avg = 0
        all_models = []
        for i in range(args.K):
            diverged = True
            while diverged:
                subnet, grad = obtain_mask(scores)
                grad_list.append(grad)
                subnet_detached = subnet.expand(xl.size(0),-1).detach()
                subnet_detachedu = subnet.expand(xu.size(0),-1).detach()
                subnet_detachedt = subnet.expand(x_test.size(0),-1).detach()
                xl_coreset, yl_coreset, xu_coreset = xl*subnet_detached, yl,xu*subnet_detachedu
                xt_coreset, yt_coreset = x_test*subnet_detachedt,   y_test

                model_copy_converged, loss, top1, top5, diverged \
                    = train_to_converge(model, xl_coreset, yl_coreset, xu_coreset )        
            scores_opt.zero_grad()
            all_models.append(model_copy_converged)
            with torch.no_grad():
                loss_on_full_train = get_loss_on_full_test(model_copy_converged, xt_coreset, yt_coreset)
            fn_list.append(loss_on_full_train)
            fn_avg += loss_on_full_train/args.K
        with torch.no_grad():
            if args.vr:
                calculateGrad_vr(scores, fn_list, grad_list, fn_avg)
            else:
                calculateGrad(scores, fn_list, grad_list)
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(scores, args.clip_constant)
        scores_opt.step()
        index_min = np.argmin(fn_list)
        model_copy_converged = all_models[index_min]
        constrainScoreByWhole(scores)
        # print("MASK:",subnet)
        if (outer_iter+1) % args.test_freq == 0:
            acc1, acc5, loss = test(model_copy_converged,x_test.cpu(), y_test.cpu())
            
            subnet_detached_test = subnet.expand(x_test.size(0),-1).detach()
            acc1_m, acc5_m, loss_m = test(model_copy_converged,x_test.cpu()*subnet_detached_test, y_test)
            
            print(f"{outer_iter}th iteration, test acc1 {acc1}, acc5 {acc5}, loss {loss}")
            print(f"{outer_iter}th iteration, test masked acc1 {acc1_m}, acc5 {acc5_m}, loss {loss_m}")
            
    print("++++++++++++++++finished solving++++++++++++++++++++")
    subnet = (torch.rand_like(scores) < scores).float()
    print("Final MASK:",subnet)
    return subnet

def solve_v_total(weight, subset):
    weight = weight.view(-1)
    k = subset
    a, b = 0, 0
    b = max(b, weight.max())
    def f(v):
        s = (weight - v).clamp(0, 1).sum()
        return s - k
    if f(0) < 0:
        return 0
    itr = 0
    while (1):
        itr += 1
        v = (a + b) / 2
        obj = f(v)
        if abs(obj) < 1e-3 or itr > 20:
            break
        if obj < 0:
            b = v
        else:
            a = v
    v = max(0, v)
    return v

def constrainScoreByWhole(scores):
    with torch.no_grad():
        v = solve_v_total(scores, args.coreset_size)
        scores.sub_(v).clamp_(0, 1)

def obtain_mask(scores):
    subnet = (torch.rand_like(scores) < scores).float()
    return subnet, (subnet - scores) / ((scores + 1e-8) * (1 - scores + 1e-8))

def train(model, Xl,Yl,Xu):
    model.fit(Xl.cpu(),Yl.cpu(),Xu.cpu())
    acc1 = 0
    acc5 = 0
    output = model.predict(Xl.cpu())
    if args.model=="LapSVM":
        output = torch.tensor(model.predict(Xl.cpu())).cpu()
        loss = np.average(-1 * (Yl*output - np.log(1+np.exp(output))))
    elif args.model=="LapRLS":
        output = torch.tensor(model.predict(Xl.cpu())).cpu()
        loss = sklearn.metrics.mean_squared_error(output, Yl)
    return acc1, acc5, loss

def test(model, X_test,y_test):
    acc1 = 0
    acc5 = 0
    output = model.predict(X_test.cpu())
    if args.model=="LapSVM":
        output = torch.tensor(model.predict(X_test.cpu())).cpu()
        loss = np.average(-1 * (y_test*output - np.log(1+np.exp(output))))
    elif args.model=="LapRLS":
        output = torch.tensor(model.predict(X_test.cpu())).cpu()
        loss = sklearn.metrics.mean_squared_error(output, y_test)
    return acc1, acc5, loss