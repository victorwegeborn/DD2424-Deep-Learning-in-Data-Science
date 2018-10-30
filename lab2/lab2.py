import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import matplotlib.pyplot as plt
import sys

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

class Dataloader:
    def __init__(self):
        pass

    def load_batch(self, filename):
        A = sio.loadmat('../datasets/cifar-10-batches-mat/'+filename)
        X = A['data'].T;
        X = X.astype('float64')/ 255

        y = A['labels']
        y = y.reshape(-1,1).T

        Y = np.zeros((10,y.size))
        Y[y, np.arange(y.size)] = 1;
        Y = Y.astype('int')
        return X, Y, y

    def get_data(self):
        train = Bunch()
        valid = Bunch()
        test = Bunch()
        train.X, train.Y, train.y = self.load_batch('data_batch_1.mat')
        valid.X, valid.Y, valid.y = self.load_batch('data_batch_2.mat')
        test.X, test.Y, test.y = self.load_batch('test_batch.mat')

        #preproccess
        X_mean = np.mean(train.X, axis=1)
        train.X -=  np.expand_dims(X_mean, axis=1)
        valid.X -= np.expand_dims(X_mean, axis=1)
        test.X -= np.expand_dims(X_mean, axis=1)

        print(train.X.shape)
        print(train.Y.shape)
        print(train.y.shape)

        return train, valid, test



    def get_all_batch_data(self, validation_size=1000):

        train = Bunch()
        valid = Bunch()
        test = Bunch()

        train.X, train.Y, train.y = self.load_batch('data_batch_1.mat')
        for i in range(1,5):
            path = 'data_batch_' + str(i+1) + '.mat'
            X, Y, y = self.load_batch(path)
            train.X = np.concatenate((train.X, X), axis=1)
            train.Y = np.concatenate((train.Y, Y), axis=1)
            train.y = np.concatenate((train.y, y), axis=1)

        split = train.X.shape[1] - 1000
        valid.X = train.X[:,split:]
        valid.Y = train.Y[:,split:]
        valid.y = train.y[:,split:]
        train.X = train.X[:,:split]
        train.Y = train.Y[:,:split]
        train.y = train.y[:,:split]

        # load test batch
        test.X, test.Y, test.y = self.load_batch('test_batch.mat')

        #preproccess
        X_mean = np.mean(train.X, axis=1)
        train.X -=  np.expand_dims(X_mean, axis=1)
        valid.X -= np.expand_dims(X_mean, axis=1)
        test.X -= np.expand_dims(X_mean, axis=1)

        print(train.X.shape)
        print(train.Y.shape)
        print(train.y.shape)

        return train, valid, test





class Network:
    def __init__(self, train, valid, test, eta=0.001, lamb=0, seed=400):

        # hyperparameters
        self.ETA = eta
        self.LAMBDA = lamb

        self.RHO = 0.8
        self.DECAY = 0.95

        # store data
        self.train = train
        self.valid = valid
        self.test = test

        # extract dimensions
        self.M = 50
        self.K = train.Y.shape[0]
        self.D = train.X.shape[0]

        #np.random.seed(seed)

        # setup weights and bias terms
        self.b = [np.zeros((self.M,1)),
                  np.zeros((self.K,1))]
        self.W = [np.random.normal(0, 0.001, (self.M,self.D)),
                  np.random.normal(0, 0.001, (self.K,self.M))]


        # momentum terms
        self.momentum_b = [np.zeros(self.b[0].shape),
                           np.zeros(self.b[1].shape)]
        self.momentum_W = [np.zeros(self.W[0].shape),
                           np.zeros(self.W[1].shape)]

    def softmax(self,x):
        e_x = np.exp(x - np.max(x)).astype('float64')
        return e_x / e_x.sum(axis=0)

    def evaluate_classifier(self, X, W, b):
        s = np.dot(W[0],X) + b[0]
        H = np.maximum(0, s)
        s1 = np.dot(W[1],H) + b[1]
        return self.softmax(s1), H

    def compute_cost(self, X, Y, W, b):
        P, H = self.evaluate_classifier(X, W, b)
        N = X.shape[1]
        entropy = 0
        for i in range(N):
            y = Y[:,i,None]
            p = P[:,i,None]
            entropy += -np.log(np.dot(y.T,p))
        return (entropy/np.float64(N) + self.LAMBDA*(np.sum(W[0]*W[0])+np.sum(W[1]*W[1])))

    def compute_accuracy(self):
        P, _ = self.evaluate_classifier(self.test.X, self.W, self.b)
        argmax = np.argmax(P, axis=0)
        return (np.sum(argmax==self.test.y)/self.test.X.shape[1])

    def compute_gradients(self, X, Y, P, H, W):
        N = X.shape[1]

        gradb = [np.zeros(self.b[0].shape),
                 np.zeros(self.b[1].shape)]
        gradW = [np.zeros(self.W[0].shape),
                 np.zeros(self.W[1].shape)]

        for i in range(N):
            h = H[:,i,None]
            y = Y[:,i,None]
            p = P[:,i,None]
            x = X[:,i,None]

            g = -(y-p).T

            gradb[1] += g.T
            gradW[1] += np.dot(g.T, h.T)

            g = np.dot(g,W[1])
            h[h > 0] = np.float64(1)
            g = np.dot(g,np.diagflat(h))

            gradb[0] += g.T
            gradW[0] += np.dot(g.T,x.T)


        gradb[0] /= np.float64(N)
        gradb[1] /= np.float64(N)
        gradW[0] /= np.float64(N)
        gradW[1] /= np.float64(N)
        gradW[0] += 2*self.LAMBDA*W[0]
        gradW[1] += 2*self.LAMBDA*W[1]

        return gradb, gradW



    def gradient_check_helper(self, cost, xm, a_grad):
        h_step = np.float64(1e-5)
        eps = np.float64(0.001)

        for m in range(len(xm)):
            x = xm[m]
            grad = a_grad[m]
            max_relative_error = -999
            print('Checking gradients..',end="", flush=True)
            for i in range(x.shape[0]):
                print('.',end="", flush=True)
                for j in range(x.shape[1]):
                    _x = np.copy(x[i,j])
                    x[i,j] = _x + h_step
                    c1 = cost(x)
                    x[i,j] = _x - h_step
                    c2 = cost(x)
                    x[i,j] = _x

                    n_grad = np.divide(c1-c2, np.float64(2*h_step))

                    relative_error = np.abs(n_grad-grad[i,j]) / np.maximum(eps,(np.abs(n_grad) + np.abs(grad[i,j])))
                    max_relative_error = np.maximum(max_relative_error, relative_error)
            print("Done\nRelative error:" , max_relative_error)

    def gradient_check(self, batch_size=100, dim_reduction=100, m_reduction=10):
        xbatch = self.train.X[:dim_reduction,:batch_size]
        ybatch = self.train.Y[:dim_reduction,:batch_size]

        # reduce dimensions for precision
        self.W = [np.copy(self.W[0][:m_reduction,:dim_reduction]), np.copy(self.W[1][:,:m_reduction])]
        self.b[0] = np.copy(self.b[0][:m_reduction,:])

        # compute analytic gradients
        P, H = self.evaluate_classifier(xbatch, self.W, self.b)
        a_gradb, a_gradW = self.compute_gradients(xbatch, ybatch, P, H, self.W)

        self.gradient_check_helper(lambda b: self.compute_cost(xbatch, ybatch, self.W, self.b), self.b, a_gradb)
        self.gradient_check_helper(lambda W: self.compute_cost(xbatch, ybatch, self.W, self.b), self.W, a_gradW)


    def train_network(self, n_epochs=10, batch_size=100, momentum=True):
        pre_train_cost = self.compute_cost(self.train.X, self.train.Y, self.W, self.b)
        pre_valid_cost = self.compute_cost(self.train.X, self.train.Y, self.W, self.b)
        train_cost = [np.asscalar(pre_train_cost)]
        valid_cost = [np.asscalar(pre_valid_cost)]

        print("epoch: 0 train loss:", pre_train_cost, "valid loss:", pre_valid_cost)

        for i in range(n_epochs):
            for j in range(1, int(self.train.X.shape[1]/batch_size)+1):
                j_start = (j-1)*batch_size
                j_end = j*batch_size
                xbatch = self.train.X[:, j_start:j_end];
                ybatch = self.train.Y[:, j_start:j_end];

                P, H = self.evaluate_classifier(xbatch, self.W, self.b)
                gradb, gradW = self.compute_gradients(xbatch, ybatch, P, H, self.W)

                if momentum:
                    self.momentum_b[0] = self.RHO*self.momentum_b[0]+self.ETA*gradb[0]
                    self.momentum_b[1] = self.RHO*self.momentum_b[1]+self.ETA*gradb[1]
                    self.momentum_W[0] = self.RHO*self.momentum_W[0]+self.ETA*gradW[0]
                    self.momentum_W[1] = self.RHO*self.momentum_W[1]+self.ETA*gradW[1]

                    self.W[0] -= self.momentum_W[0]
                    self.W[1] -= self.momentum_W[1]
                    self.b[0] -= self.momentum_b[0]
                    self.b[1] -= self.momentum_b[1]
                else:
                    self.W[0] -= self.ETA*gradW[0]
                    self.W[1] -= self.ETA*gradW[1]
                    self.b[0] -= self.ETA*gradb[0]
                    self.b[1] -= self.ETA*gradb[1]

            # cost over time
            t_cost = self.compute_cost(self.train.X, self.train.Y, self.W, self.b)
            v_cost = self.compute_cost(self.valid.X, self.valid.Y, self.W, self.b)
            print("epoch:", i+1 ,"train loss:", t_cost, "valid loss:", v_cost)
            train_cost.append(np.asscalar(t_cost))
            valid_cost.append(np.asscalar(v_cost))

            # decay rate
            if momentum:
                self.ETA = self.ETA*self.DECAY
        return train_cost, valid_cost


def plotCost(train_cost, valid_cost):
    epochs = np.arange(0,len(train_cost))
    train_cost = np.array(train_cost)
    valid_cost = np.array(valid_cost)

    plt.plot(epochs, train_cost, 'r-', label='training loss')
    plt.plot(epochs, valid_cost, 'b-', label='validation loss')
    plt.legend(loc='upper right', shadow='true')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def parameter_search(settings):
    LAMBDA = np.random.uniform(settings['LAMBDA_MIN'], settings['LAMBDA_MAX'], (settings['SIZE'],1))
    ETA = np.random.uniform(settings['ETA_MIN'], settings['ETA_MAX'], (settings['SIZE'],1))
    results = []

    train, valid, test = Dataloader().get_data()
    for i in range(settings['SIZE']):
        _ETA = pow(10,ETA[i])
        _LAMBDA = pow(10,LAMBDA[i])
        network = Network(train, valid, test, eta=_ETA, lamb=_LAMBDA)
        _, v = network.train_network(n_epochs=settings['EPOCHS'])
        results.append((v[-1],[ETA[i], _ETA, LAMBDA[i], _LAMBDA]))

    file = open(settings['FILENAME'], 'w')
    results.sort()
    for r in results:
        file.write('loss: %s ' % r[0])
        file.write('%s\n' % r[1])
    file.close()



def main():
    '''
    # Gradient checks
    train, valid, test = Dataloader().get_data()
    network.gradient_check()
    '''

    '''
    # Coarse search
    parameter_search({'ETA_MIN': -4,
                      'ETA_MAX': -1,
                      'LAMBDA_MIN': -6,
                      'LAMBDA_MAX': -1,
                      'EPOCHS': 5,
                      'SIZE': 100,
                      'FILENAME': 'search_1.txt'})

    # fine search 1
    parameter_search({'ETA_MIN': -1.9,
                      'ETA_MAX': -1.1,
                      'LAMBDA_MIN': -5.9,
                      'LAMBDA_MAX': -3.5,
                      'EPOCHS': 7,
                      'SIZE': 80,
                      'FILENAME': 'search_2.txt'})
    '''

    '''
    #Tested found fine searched hyper-parameters
    train, valid, test = Dataloader().get_data()
    network = Network(train, valid, test, eta=0.02632556, lamb=3.39446014e-06)
    t,v = network.train_network()
    print(network.compute_accuracy())
    plotCost(t,v)
    '''

    # Final training on all batches
    train, valid, test = Dataloader().get_all_batch_data()
    network = Network(train, valid, test, eta=0.018, lamb=8.0e-04)
    t,v = network.train_network(n_epochs=30)
    print(network.compute_accuracy())
    plotCost(t,v)

if __name__ == "__main__":
    main()
