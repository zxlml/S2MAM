
import numpy as np
import torch
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from random import sample
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
import matplotlib as plt
import random
from sklearn.datasets import make_moons,make_circles,make_friedman1,make_friedman2,load_diabetes

def classification_data_generation(data_type):
    if(data_type=='moon'):
        X, Y = make_moons(n_samples=200, noise=0.01)
        Xt, Yt = make_moons(n_samples=200, noise=0.01)
        X=np.concatenate((X,np.random.normal(0, 100,(200,8))),axis=1)
        Xt=np.concatenate((Xt,np.random.normal(0, 100,(200,8))),axis=1)
    elif(data_type=='circle'):
        X, Y = make_circles(n_samples=200, noise=0.01)
        Xt, Yt = make_circles(n_samples=200, noise=0.01)
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
    return Xl,Yl,Xu,Xt,Yt

def generate_Y(dataX):
    f1 = -2 * np.sin(2 * dataX[:, 0])
    f2 = 8 * np.square(dataX[:, 1])
    f3 = 7 * np.sin(dataX[:, 2]) / (2-np.sin(dataX[:,2]))
    f4 = 6 * np.exp(-dataX[:, 3])
    f5 = np.power(dataX[:, 4], 3) + 1.5*np.square(dataX[:, 4] - 1)
    f6 = 5 * dataX[:, 5]
    f7 = 10 * np.sin(np.exp(-dataX[:, 6]/2))
    f8 = -10 * norm.cdf(dataX[:, 7], loc=0.5, scale=0.8)

    Y = f1+f2+f3+f4+f5+f6+f7+f8
    Y = np.expand_dims(Y, axis=1)
    return Y

def regression_data_generation(data_type):
    
    if(data_type==0):
        n = 1000
        ntest = 200

        #random.seed(144)
        #np.random.seed(144)
        n2 = int(n / 2)
        sigma = 0.05
        xs = np.random.randn(n, 1) + 2
        xs[:n2, :] -= 4
        ys = sigma * np.random.randn(n, 1) + np.sin(xs / 2)
        #xt = np.random.randn(n, 1) + 2
        #xt[:n2, :] /= 2
        X = xs.copy()[:ntest]
        #xt[:n2, :] -= 3
        #xt=xs
        
        Y = sigma * np.random.randn(ntest, 1) + np.sin(X / 2)+0.4
        X += 2
        
        X=np.concatenate((X,np.random.normal(10, 100,(X.shape[0],49))),axis=1)
        Xt, Yt = X,Y
    if(data_type==1):
        X,Y = make_friedman1(n_samples=1000, n_features=5)
        Xt, Yt =  make_friedman1(n_samples=1000, n_features=5)
        X=np.concatenate((X,np.random.normal(10, 100,(X.shape[0],5))),axis=1)
        Xt=np.concatenate((Xt,np.random.normal(10, 100,(Xt.shape[0],5))),axis=1)
    elif(data_type==2):
# =============================================================================
#         442条数据，每条数据10维，即每条数据10个特征
# =============================================================================
        data  = load_diabetes()
        X,Y = data.data, data.target
        Xt, Yt =  data.data, data.target
        X=np.concatenate((X,np.random.normal(0, 100,(X.shape[0],90))),axis=1)
        Xt=np.concatenate((Xt,np.random.normal(0, 100,(Xt.shape[0],90))),axis=1)
    elif(data_type==3):
        N=200
        p=100
        X = np.random.uniform(-1, 1, size=(N, p))
        Xt = np.random.uniform(-1, 1, size=(N, p))

        Y = generate_Y(X)
        Yt  = generate_Y(Xt)

        
        scaler1 = StandardScaler()
        scaler1.fit(np.vstack((X, Xt)))
        X, Xt = map(scaler1.transform, [X, Xt])
        scaler2 = StandardScaler()
        scaler2.fit(np.vstack((Y, Yt)))
        Y, Yt = map(scaler2.transform, [Y, Yt])
        
        X=np.concatenate((X,np.random.normal(100, 100,(X.shape[0],10))),axis=1)
        Xt=np.concatenate((Xt,np.random.normal(100, 100,(Xt.shape[0],10))),axis=1)
        
    labeled_index=random.sample(range(0, X.shape[0]), 50)
    remain_index = list(set(range(X.shape[0]))-set(list(labeled_index)))
    
    Xl = torch.tensor(X[labeled_index]).cpu()
    Yl = torch.tensor(Y[labeled_index]).cpu()
    Xu = torch.tensor(X[remain_index]).cpu()
    Xt = torch.tensor(Xt).cpu()
    Yt = torch.tensor(Yt).cpu()
    return Xl,Yl,Xu,Xt,Yt


def bsplineBasis_j(x, t, j, M):
    #  bsplineBasis: compute a bspline basis function of order m.
    p = M-1; # polynomial degree
    
    # construct the b-spline recursively.
    if(p == 0): # piecewise constatnt
        Bj = ((x >= t[j]) & (x < t[j+1]))
    else:
        # If the two knots in the denominator are equal, we ignore the term (to not devide by zero).
        denom1 = t[j + M - 1] - t[j];
        if(denom1 == 0):
            Bj = np.zeros(len(x))
        else:
            Bj = (x - t[j]) * bsplineBasis_j(x, t, j, M-1) / denom1;
        
        denom2 = t[j + M] - t[j + 1]
        if(denom2 != 0):
            Bj = Bj + (t[j + M] - x) * bsplineBasis_j(x, t, j+1, M-1) / denom2;
    return Bj

def bsplinebasis(x, t, M):   
    m = len(x)
    # repeat the first and the last knots m times
    t1=list(t[0]*np.ones(M-1))
    t2=list(t[-1]*np.ones(M-1))
    t=t1+t+t2
    # j th basis function : 1 <= j <= length(t) - M -1 ; There are K + 2*M knots ; M = length(t) - 1;
    B = np.zeros([m, len(t) - M])
    for j in range(0,len(t)- M): #is the same as j=1 : K+M, K is the number of interior knots
        B[:,j] = bsplineBasis_j( x, t, j, M )# j-1 0 <= j <= len(tt) - n - 2.
    # print(B)
    return B

    
def transform_splines(trainX,  validX, testX,r):
# =============================================================================
#     Suggestion: r=3 for regression tasks and r=5 for classification tasks
# =============================================================================
    X_trn_spline = np.ones([trainX.shape[0],trainX.shape[1]*r])
    X_val_spline = np.ones([validX.shape[0],validX.shape[1]*r])
    X_tst_spline = np.ones([testX.shape[0],testX.shape[1]*r])
    
    for j in range(0,len(trainX[0])):
      min_k_trn = min(trainX[:,j])
      max_k_trn = max(trainX[:,j])
      knot_trn  = [ min_k_trn,max_k_trn]
      X_trn_spline[:,(j)*r:(j+1)*r] = bsplinebasis(trainX[:,j], knot_trn, r)
      # print(X_trn_spline[:,j*r:(j+1)*r])
                    
      min_k_val = min(validX[:,j])
      max_k_val = max(validX[:,j])
      knot_val  = [ min_k_val,max_k_val]
      X_val_spline[:,(j)*r:(j+1)*r] = bsplinebasis(validX[:,j], knot_val, r)
      # print(X_val_spline[:,j*r:(j+1)*r])
            
      min_k_tst = min(testX[:,j])
      max_k_tst = max(testX[:,j])
      knot_tst  = [ min_k_tst,max_k_tst]
      X_tst_spline[:,(j)*r:(j+1)*r] = bsplinebasis(testX[:,j], knot_tst, r)
      # print(X_tst_spline[:,j*r:(j+1)*r])
    
    return X_trn_spline,X_val_spline,X_tst_spline

class Regression:
    def __init__(self, N, p, noise="None"):
        self.N, self.p = N, p
        self.noise = noise
    
    def add_noise(self, dataY):
        N = len(dataY)
        noise = self.noise
        if noise == "studentT":
            noise = np.random.standard_t(df=2, size=(N, 1))
        elif noise == "mixGauss":
            noise = np.zeros((N, 1))
            for i in range(N):
                c = np.random.uniform()
                if c < 0.8:
                    noise[i][0] = np.random.normal(loc=-2, scale=1)
                else:
                    noise[i][0] = np.random.normal(loc=40, scale=1)
        elif noise == "mean":
            noise = np.zeros((N, 1))
            for i in range(N):
                c = np.random.uniform()
                if c < 0.8:
                    noise[i][0] = np.random.normal(loc=-2, scale=1)
                else:
                    noise[i][0] = np.random.normal(loc=8, scale=1)
        elif noise == "modal":
            noise = np.zeros((N, 1))
            for i in range(N):
                c = np.random.uniform()
                if c < 0.8:
                    noise[i][0] = np.random.normal(loc=0, scale=1)
                else:
                    noise[i][0] = np.random.normal(loc=20, scale=1)
        elif noise == "Gaussian":
            noise = np.random.randn(N, 1)
        elif noise == "Gauss2":
            noise = np.random.normal(loc=0, scale=2, size=(N, 1))
        elif noise == "chiSquare":
            noise = np.random.chisquare(1, size=(N, 1))
        elif noise == "None":
            noise = 0
        noiseY = dataY + noise
        # print("max y is :", np.max(np.abs(dataY)), "max noise is :", np.max(np.abs(noise)))
        return noiseY
    
    def plot_data(self, dataX, noiseY):
        dataY = self.generate_Y(dataX)
        plt.plot(range(self.N), dataY)
        plt.plot(range(self.N), noiseY)
        plt.savefig( "trainY.png")
        plt.show()
    
    def generate_Y(self, dataX):
        f1 = -2 * np.sin(2 * dataX[:, 0])
        f2 = 8 * np.square(dataX[:, 1])
        f3 = 7 * np.sin(dataX[:, 2]) / (2-np.sin(dataX[:,2]))
        f4 = 6 * np.exp(-dataX[:, 3])
        f5 = np.power(dataX[:, 4], 3) + 1.5*np.square(dataX[:, 4] - 1)
        f6 = 5 * dataX[:, 5]
        f7 = 10 * np.sin(np.exp(-dataX[:, 6]/2))
        f8 = -10 * norm.cdf(dataX[:, 7], loc=0.5, scale=0.8)
    
        Y = f1+f2+f3+f4+f5+f6+f7+f8
        Y = np.expand_dims(Y, axis=1)
        return Y


    def generate_data(self):
        trainX = np.random.uniform(-1, 1, size=(self.N, self.p))
        validX = np.random.uniform(-1, 1, size=(self.N, self.p))
        testX = np.random.uniform(-1, 1, size=(self.N, self.p))

        trainY = self.generate_Y(trainX)
        validY = self.generate_Y(validX)
        testY  = self.generate_Y(testX)

        # add noise to trainY 
        trainY = self.add_noise(trainY)

        scaler1 = StandardScaler()
        scaler1.fit(np.vstack((trainX, validX)))
        trainX, validX, testX = map(scaler1.transform, [trainX, validX, testX])
        scaler2 = StandardScaler()
        scaler2.fit(np.vstack((trainY, validY)))
        trainY, validY, testY = map(scaler2.transform, [trainY, validY, testY])
        
        return (trainX, trainY), (validX, validY), (testX, testY)
        
class Classfication_corrupted:
    def __init__(self, N, p, frac=0.1):
        self.N, self.p = N, p
        self.frac = frac
    
    def add_noise(self, dataY):
        N = len(dataY)
        frac = self.frac
        idx = sample(range(self.N), int(self.N*self.frac))
        dataY[idx] = 1 - dataY[idx]
        return dataY

    def plot_data(self, dataX, noiseY):
        dataY = self.generate_Y(dataX)
        pos_idx = np.where(dataY==1)[0]
        neg_idx = np.where(dataY==-1)[0]
        plt.scatter(dataX[pos_idx, 0], dataX[pos_idx, 1], c='green')
        plt.scatter(dataX[neg_idx, 0], dataX[neg_idx, 1], c='red')
        plt.savefig("trainY.png")
        plt.show()
    
    def generate_Y(self, dataX):
        f1 = np.square(dataX[:, 0] - 0.5)
        f2 = np.square(dataX[:, 1] - 0.5)
        Y = f1 + f2 - 0.08
        Y = np.expand_dims(Y, axis=1)
        Y[Y>0] = 1
        Y[Y<=0] = 0
        return Y.astype(int)

    def generate_X(self):
        N, p = self.N, self.p
        W = np.random.uniform(low=0, high=1, size=(N, p))
        U = np.random.uniform(low=0, high=1, size=(N, 1))
        return (W+U) / 2

    def generate_data(self):
        trainX = self.generate_X()
        validX = self.generate_X()
        testX = self.generate_X()

        trainY = self.generate_Y(trainX)
        validY = self.generate_Y(validX)
        testY  = self.generate_Y(testX)

        # add noise to trainY 
        trainY = self.add_noise(trainY)

        # plot data
        # self.plot_data(trainX, trainY)

        # standard data
        scaler1 = StandardScaler()
        scaler1.fit(np.vstack((trainX, validX)))
        trainX, validX, testX = map(scaler1.transform, [trainX, validX, testX])
        
        return (trainX, trainY), (validX, validY), (testX, testY)

class Classfication_imbalance:
    def __init__(self, N, p, frac=0.1):
        self.N, self.p = N, p
        self.frac = frac
    
    def sample_data(self, dataX, dataY, frac):
        neg_num = int(self.N * frac)
        pos_num = self.N - neg_num
        idx1 = sample(list(np.where(dataY==0)[0]), neg_num)
        idx2 = sample(list(np.where(dataY==1)[0]), pos_num)
        negX, negY = dataX[idx1], dataY[idx1]
        posX, posY = dataX[idx2], dataY[idx2]
        trainX = np.vstack((negX, posX))
        trainY = np.vstack((negY, posY))
        data = np.hstack((trainX, trainY))
        np.random.shuffle(data)
        trainX, trainY = data[:,:-1], data[:, -1]
        return trainX, np.expand_dims(trainY, axis=1)

    def plot_data(self, dataX, noiseY):
        dataY = self.generate_Y(dataX)
        pos_idx = np.where(dataY==1)[0]
        neg_idx = np.where(dataY==-1)[0]
        plt.scatter(dataX[pos_idx, 0], dataX[pos_idx, 1], c='green')
        plt.scatter(dataX[neg_idx, 0], dataX[neg_idx, 1], c='red')
        plt.savefig( "trainY.png")
        plt.show()
    
    def generate_Y(self, dataX):
        f1 = np.square(dataX[:, 0] - 0.5)
        f2 = np.square(dataX[:, 1] - 0.5)
        Y = f1 + f2 - 0.08
        Y = np.expand_dims(Y, axis=1)
        Y[Y>0] = 1
        Y[Y<=0] = 0
        return Y.astype(int)

    def generate_X(self):
        N, p = 10000, self.p
        W = np.random.uniform(low=0, high=1, size=(N, p))
        U = np.random.uniform(low=0, high=1, size=(N, 1))
        return (W+U) / 2

    def generate_data(self):
        trainX = self.generate_X()
        validX = self.generate_X()
        testX = self.generate_X()

        trainY = self.generate_Y(trainX)
        validY = self.generate_Y(validX)
        testY  = self.generate_Y(testX)

        trainX, trainY = self.sample_data(trainX, trainY, self.frac)
        validX, validY = self.sample_data(validX, validY, 0.5)
        testX,  testY  = self.sample_data(testX, testY, 0.5)
        # print(trainY.shape)

        # standard data
        scaler1 = StandardScaler()
        scaler1.fit(np.vstack((trainX, validX)))
        trainX, validX, testX = map(scaler1.transform, [trainX, validX, testX])

        return (trainX, trainY), (validX, validY), (testX, testY)


class Classfication_multi:
    def __init__(self, N, p, frac=0.1):
        self.N, self.p = N, p
        self.frac = frac

    def plot_data(self, dataX, noiseY):
        dataY = self.generate_Y(dataX)
        pos_idx = np.where(dataY==1)[0]
        neg_idx = np.where(dataY==-1)[0]
        plt.scatter(dataX[pos_idx, 0], dataX[pos_idx, 1], c='green')
        plt.scatter(dataX[neg_idx, 0], dataX[neg_idx, 1], c='red')
        plt.savefig("trainY.png")
        plt.show()
    
    def generate_X(self):
        N, p = 10000, self.p
        W = np.random.uniform(low=0, high=1, size=(N, p))
        U = np.random.uniform(low=0, high=1, size=(N, 1))
        return (W+U) / 2
    
    def generate_Y(self, dataX):
        f1 = np.square(dataX[:, 0] - 0.5)
        f2 = np.square(dataX[:, 1] - 0.5)
        Y = f1 + f2 - 0.08
        Y = np.expand_dims(Y, axis=1)
        Y[Y>0] = 1
        Y[Y<=0] = 0
        return Y.astype(int)
    
    def add_noise(self, dataY, frac2):
        N = len(dataY)
        if frac2>0:
            idx = sample(range(N), int(N*frac2))
            dataY[idx] = 1 - dataY[idx]
        return dataY.astype(int)
    
    def sample_data(self, dataX, dataY, frac, frac2=0):

        ## sample data accodring to fraction
        neg_num = int(self.N * frac)
        pos_num = self.N - neg_num
        idx1 = sample(list(np.where(dataY==0)[0]), neg_num)
        idx2 = sample(list(np.where(dataY==1)[0]), pos_num)
        negX, negY = dataX[idx1], dataY[idx1]
        posX, posY = dataX[idx2], dataY[idx2]

        ## add noise to Y
        negY = self.add_noise(negY, frac2)
        posY = self.add_noise(posY, frac2)

        trainX = np.vstack((negX, posX))
        trainY = np.vstack((negY, posY))
        data = np.hstack((trainX, trainY))
        np.random.shuffle(data)
        trainX, trainY = data[:,:-1], data[:, -1]
        return trainX, np.expand_dims(trainY, axis=1)

    def generate_data(self):
        trainX = self.generate_X()
        validX = self.generate_X()
        testX = self.generate_X()

        trainY = self.generate_Y(trainX)
        validY = self.generate_Y(validX)
        testY  = self.generate_Y(testX)

        trainX, trainY = self.sample_data(trainX, trainY, self.frac, frac2=0.3) #Imbalance  & Corruption = 0.3
        validX, validY = self.sample_data(validX, validY, 0.5, frac2=0)
        testX,  testY  = self.sample_data(testX, testY, 0.5, frac2=0)
        # print(trainY.shape)

        # standard data
        scaler1 = StandardScaler()
        scaler1.fit(np.vstack((trainX, validX)))
        trainX, validX, testX = map(scaler1.transform, [trainX, validX, testX])

        return (trainX, trainY), (validX, validY), (testX, testY)

def data_process(trainX, trainY, validX, validY,testX,testY,batch,r):
    trainX, validX, testX = transform_splines(trainX, validX, testX,r)
    train_data = Data.TensorDataset(torch.tensor(trainX), torch.tensor(trainY))
    val_data = Data.TensorDataset(torch.tensor(validX), torch.tensor(validY))
    test_data = Data.TensorDataset(torch.tensor(testX), torch.tensor(testY))
    
    train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=batch,
    shuffle=True,
    num_workers=0,
    )

    val_loader = Data.DataLoader(
    dataset=val_data,
    batch_size=batch,
    shuffle=True,
    num_workers=0,
    )
    
    test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=batch,
    shuffle=True,
    num_workers=0,
    )
    
    return train_loader,val_loader, test_loader


def generate_regression(number=1000,dimension=100,noise_type='Gaussian'):
    N, p = number,dimension
    noise = noise_type
    reg_data = Regression(N, p, noise=noise)

    for i in range(number):
        (trainX, trainY), (validX, validY), (testX, testY) = reg_data.generate_data()
    train_loder,val_loder,test_loader =data_process(trainX, trainY, validX, validY,testX,testY,batch=200,r=5)
    return train_loder,val_loder, test_loader

def generate_classification(number=1000,dimension=100,percentage=0.3):
    N, p = number,dimension
    frac = percentage
    reg_data = Classfication_corrupted(N, p, frac=frac)

    train_x, train_y = [], []
    valid_x, valid_y = [], []
    test_x,  test_y  = [], []


    for i in range(number):
        (trainX, trainY), (validX, validY),  (testX, testY) = reg_data.generate_data()
        train_x.append(trainX)
        train_y.append(trainY)
        valid_x.append(validX)
        valid_y.append(validY)
        test_x.append(testX)
        test_y.append(testY)
    train_loder,val_loder,test_loader =data_process(trainX, trainY, validX, validY,testX,testY,batch=200,r=5)
    return train_loder,val_loder, test_loader

def generate_corrupted_classification(number=1000,dimension=100,percentage=0.3):
    N, p = number,dimension
    frac = percentage
    reg_data = Classfication_corrupted(N, p, frac=frac)

    train_x, train_y = [], []
    valid_x, valid_y = [], []
    test_x,  test_y  = [], []


    for i in range(number):
        (trainX, trainY), (validX, validY),  (testX, testY) = reg_data.generate_data()
        train_x.append(trainX)
        train_y.append(trainY)
        valid_x.append(validX)
        valid_y.append(validY)
        test_x.append(testX)
        test_y.append(testY)
    train_loder,val_loder,testX =data_process(trainX, trainY, validX, validY,testX,batch=200,r=5)
    return train_loder,val_loder, testX, testY


def generate_imbalanced_classification(number=1000,dimension=100,ratio=0.15):
    N, p = number,dimension
    frac = ratio
    reg_data = Classfication_imbalance(N, p, frac=frac)

    train_x, train_y = [], []
    valid_x, valid_y = [], []
    test_x,  test_y  = [], []


    for i in range(number):
        (trainX, trainY), (validX, validY),  (testX, testY) = reg_data.generate_data()
        train_x.append(trainX)
        train_y.append(trainY)
        valid_x.append(validX)
        valid_y.append(validY)
        test_x.append(testX)
        test_y.append(testY)
    train_loder,val_loder,testX =data_process(trainX, trainY, validX, validY,testX,batch=200,r=5)
    return train_loder,val_loder,testX, testY

def generate_multi_classification(number=1000,dimension=100,ratio=0.15):
    N, p = number,dimension
    frac = ratio
    reg_data = Classfication_multi(N, p, frac=frac)

    train_x, train_y = [], []
    valid_x, valid_y = [], []
    test_x,  test_y  = [], []


    for i in range(number):
        (trainX, trainY), (validX, validY), (testX, testY) = reg_data.generate_data()
        train_x.append(trainX)
        train_y.append(trainY)
        valid_x.append(validX)
        valid_y.append(validY)
        test_x.append(testX)
        test_y.append(testY)
    train_loder,val_loder,testX =data_process(trainX, trainY, validX, validY,testX,batch=200,r=5)
    return train_loder,val_loder, testX, testY


# =============================================================================
#  论文（data.md）四个仿真数据集的忠实实现
#  均支持：部分标注（n_labeled / labeled_per_class）、无信息维度 p_u、
#  含噪声维度 p_n ~ N(100,100)、随机种子。
# =============================================================================
def _standardize(X, ref):
    mean, std = ref.mean(axis=0), ref.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def s2mam_additive_components(X):
    """data.md (2) 的 8 个可加分量（X 取值范围 [-1,1]）。"""
    from scipy.stats import norm as _norm
    f1 = -2 * np.sin(2 * X[:, 0])
    f2 = 8 * np.square(X[:, 1])
    f3 = 7 * np.sin(X[:, 2]) / (2 - np.sin(X[:, 2]))
    f4 = 6 * np.exp(-X[:, 3])
    f5 = np.power(X[:, 4], 3) + 1.5 * np.square(X[:, 4] - 1)
    f6 = 5 * X[:, 5]
    f7 = 10 * np.sin(np.exp(-X[:, 6] / 2))
    f8 = -10 * _norm.cdf(X[:, 7], loc=0.5, scale=0.8)
    return f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8


def s2mam_regression_data(dataset='additive', N=200, Ntest=200, n_labeled=50,
                          n_uninformative=None, n_noisy=10, noise_std=1.0,
                          seed=0, standardize=True):
    """论文回归仿真数据。

    dataset='additive'：data.md (2)。p*=8 个可加信息维度（U(-1,1) 生成），
        p_u=92 个 N(0,1) 无信息维度，p_n=10 个 N(100,100) 噪声维度，y=f*(X)+eps。
    dataset='friedman'：data.md (1)。p*=5 个 Friedman 信息维度（U(0,1)），
        p_u=95 个 N(0,1) 无信息维度，p_n=10 个 N(100,100) 噪声维度，
        f(X)=10 sin(pi x1 x2)+20(x3-0.5)^2+10x4+5x5+eps，eps~N(0,1)。

    返回 dict：Xl/Yl（部分标注训练集）、Xu（无标注）、Xt/Yt（测试集）、
    informative_idx（信息维度索引）、n_noisy、p。
    """
    rng = np.random.RandomState(seed)
    if dataset == 'additive':
        p_star, p_u = 8, (92 if n_uninformative is None else n_uninformative)
        X = rng.uniform(-1, 1, size=(N, p_star))
        Xt = rng.uniform(-1, 1, size=(Ntest, p_star))
        Y, Yt = s2mam_additive_components(X), s2mam_additive_components(Xt)
        eps = rng.normal(0, noise_std, size=N)
        Y = Y + eps
    elif dataset == 'friedman':
        p_star, p_u = 5, (95 if n_uninformative is None else n_uninformative)
        from sklearn.datasets import make_friedman1
        X, Y = make_friedman1(n_samples=N, n_features=p_star, noise=noise_std,
                              random_state=seed)
        Xt, Yt = make_friedman1(n_samples=Ntest, n_features=p_star,
                                noise=noise_std, random_state=seed + 1)
        Y, Yt = np.asarray(Y, float), np.asarray(Yt, float)
    else:
        raise ValueError(f'unknown dataset {dataset}')

    # 无信息维度 N(0,1) 与噪声维度 N(100,100)
    Xu_ = rng.normal(0, 1, size=(N, p_u))
    Xtu_ = rng.normal(0, 1, size=(Ntest, p_u))
    Xn = rng.normal(100, 100, size=(N, n_noisy))
    Xtn = rng.normal(100, 100, size=(Ntest, n_noisy))
    X = np.hstack([X, Xu_, Xn])
    Xt = np.hstack([Xt, Xtu_, Xtn])

    if standardize:
        ref = np.vstack([X, Xt])
        X, Xt = _standardize(X, ref), _standardize(Xt, ref)

    informative_idx = np.arange(p_star)

    # 部分标注：随机抽取 n_labeled 个有标注样本
    labeled = rng.choice(N, size=min(n_labeled, N), replace=False)
    unlabeled = np.setdiff1d(np.arange(N), labeled)
    Yt = np.asarray(Yt, float).reshape(-1)
    return {'Xl': X[labeled], 'Yl': Y[labeled],
            'Xu': X[unlabeled], 'Xt': Xt, 'Yt': Yt,
            'informative_idx': informative_idx, 'n_noisy': n_noisy,
            'p': X.shape[1], 'seed': seed}


def s2mam_classification_data(dataset='additive', N=200, Ntest=200,
                              labeled_per_class=10, n_uninformative=None,
                              n_noisy=10, noise=0.05, seed=0,
                              standardize=True):
    """论文分类仿真数据（标签映射到 ±1）。

    dataset='additive'：data.md (3)。p*=2，f(x)=(x1-0.5)^2+(x2-0.5)^2-0.08，
        x_j=(W_ij+U_i)/2，p_u=98 个 N(0,1) 无信息维度，p_n=10 个 N(100,100)
        噪声维度，y=1 若 f>0 否则 -1。
    dataset='moon'：data.md (4)。make_moons 双月牙，拼接无信息与噪声维度。

    部分标注：每类抽取 labeled_per_class 个有标注样本。
    """
    rng = np.random.RandomState(seed)
    if dataset == 'additive':
        p_star, p_u = 2, (98 if n_uninformative is None else n_uninformative)
        W = rng.uniform(0, 1, size=(N, p_star))
        U = rng.uniform(0, 1, size=(N, 1))
        Xs = (W + U) / 2
        Wt = rng.uniform(0, 1, size=(Ntest, p_star))
        Ut = rng.uniform(0, 1, size=(Ntest, 1))
        Xts = (Wt + Ut) / 2

        def _label(Xa):
            f = np.square(Xa[:, 0] - 0.5) + np.square(Xa[:, 1] - 0.5) - 0.08
            return np.where(f > 0, 1, -1).astype(float)

        Y, Yt = _label(Xs), _label(Xts)
        base_X, base_Xt = Xs, Xts
    elif dataset == 'moon':
        from sklearn.datasets import make_moons
        Xs, Y0 = make_moons(n_samples=N, noise=noise, random_state=seed)
        Xts, Yt0 = make_moons(n_samples=Ntest, noise=noise, random_state=seed + 1)
        Y = np.where(Y0 > 0, 1.0, -1.0)
        Yt = np.where(Yt0 > 0, 1.0, -1.0)
        p_star, p_u = 2, (8 if n_uninformative is None else n_uninformative)
        base_X, base_Xt = Xs, Xts
    else:
        raise ValueError(f'unknown dataset {dataset}')

    Xu_ = rng.normal(0, 1, size=(N, p_u))
    Xtu_ = rng.normal(0, 1, size=(Ntest, p_u))
    Xn = rng.normal(100, 100, size=(N, n_noisy))
    Xtn = rng.normal(100, 100, size=(Ntest, n_noisy))
    X = np.hstack([base_X, Xu_, Xn])
    Xt = np.hstack([base_Xt, Xtu_, Xtn])

    if standardize:
        ref = np.vstack([X, Xt])
        X, Xt = _standardize(X, ref), _standardize(Xt, ref)

    informative_idx = np.arange(p_star)

    # 部分标注：每类 labeled_per_class 个
    idx_pos = np.nonzero(Y == 1)[0]
    idx_neg = np.nonzero(Y == -1)[0]
    l_pos = rng.choice(idx_pos, size=min(labeled_per_class, len(idx_pos)), replace=False)
    l_neg = rng.choice(idx_neg, size=min(labeled_per_class, len(idx_neg)), replace=False)
    labeled = np.concatenate([l_pos, l_neg])
    unlabeled = np.setdiff1d(np.arange(N), labeled)
    return {'Xl': X[labeled], 'Yl': Y[labeled],
            'Xu': X[unlabeled], 'Xt': Xt, 'Yt': Yt,
            'informative_idx': informative_idx, 'n_noisy': n_noisy,
            'p': X.shape[1], 'seed': seed}