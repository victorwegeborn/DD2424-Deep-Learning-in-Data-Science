import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from copy import deepcopy
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

    def get_small_data(self, sample):
        train, valid, test = self.get_data()
        train.X = train.X[:,:sample]
        train.Y = train.Y[:,:sample]
        train.y = train.y[:sample]
        valid.X = valid.X[:,:sample]
        valid.Y = valid.Y[:,:sample]
        valid.y = valid.y[:sample]
        test.X = test.X[:,:sample]
        test.Y = test.Y[:,:sample]
        test.y = test.y[:sample]
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

        # split training into training and validation set
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
    def __init__(self, train, valid, test, eta, lamb, rho=0.8, seed=400, BN=False, He_init=False, gradient_check=False):

        # hyperparameters
        self.ETA = eta
        self.LAMBDA = lamb

        self.RHO = rho
        self.DECAY = 0.99

        # store data
        self.train = train
        self.valid = valid
        self.test = test

        self.model = []
        self.BN = BN

        np.random.seed(seed)
        self.He_init = He_init
        self.gradients_check = gradient_check

    def add_layer_to_model(self, nodes, input_dim):
        std = 0.001
        if self.He_init:
            std = np.sqrt(2/(nodes + input_dim))

        W = np.random.normal(0, std, (nodes, input_dim))
        #b = np.random.normal(0, std, (nodes,1))
        b = np.zeros((nodes,1))
        W_v = np.zeros(W.shape)
        b_v = np.zeros(b.shape)
        self.model.append({'W': W, 'b': b,
                           'momentum_W': W_v,
                           'momentum_b': b_v,
                           's': None,
                           's^': None,
                           'X': None,
                           'mean': None,
                           'mean_avg': None,
                           'variance': None,
                           'variance_avg': None})


    def build_model(self):
        if not self.model:
            self.add_layer_to_model(self.train.Y.shape[0], self.train.X.shape[0])
        else:
            self.add_layer_to_model(self.train.Y.shape[0], self.model[-1]['W'].shape[0])

    def print_model(self):
        for layer in self.model:
            print("model W:", layer['W'].shape, "b:", layer['b'].shape)


    def softmax(self, score):
        e_x = np.exp(score - np.max(score)).astype('float64')
        return e_x / e_x.sum(axis=0)


    def linear(self, W, X, b):
        return np.dot(W,X) + b

    def batch_normalize(self, score, mode, layer):
        alpha = 0.99

        N = np.float64(score.shape[1])
        mean = score.sum(axis=1, keepdims=True)/N
        variance = np.sum(np.square(score-mean), axis=1, keepdims=True)/N

        if not self.gradients_check:
            if mode == 'test':
                if layer['mean_avg'] is not None:
                    mean = layer['mean_avg']

                if layer['variance_avg'] is not None:
                    variance = layer['variance_avg']
            else:
                if layer['mean_avg'] is None:
                    layer['mean_avg'] = mean
                else:
                    layer['mean_avg'] = alpha*layer['mean_avg'] + (1-alpha)*mean

                if layer['variance_avg'] is None:
                    layer['variance_avg'] = variance
                else:
                    layer['variance_avg'] = alpha*layer['variance_avg'] + (1-alpha)*variance

                layer['mean'] = mean
                layer['variance'] = variance
        else:
            layer['mean'] = mean
            layer['variance'] = variance


        r1 = score-mean
        r2 = np.power(variance + 1e-12, -0.5)
        res = r1 * r2
        assert score.shape == res.shape
        return res

    def batch_norm_back_pass(self, g, layer):
        N = g.shape[0]
        g = g.T
        s = layer['s'] - layer['mean']
        v_root_inverse = np.power(layer['variance'] + 1e-12, -0.5)
        djdv = -np.power(layer['variance'] + 1e-12, -1.5) * np.sum(np.multiply(g, s), axis=1, keepdims=True)
        djdm = -np.multiply(v_root_inverse, np.sum(g, axis=1, keepdims=True))
        g = np.multiply(g, v_root_inverse) + np.multiply(djdv, s) / N + djdm / N
        return g.T


    def relu_activation(self, score):
        return np.maximum(0, score)


    def add_layer(self, nodes):
        if not self.model:
            input_dim = self.train.X.shape[0]
        else:
            input_dim = self.model[-1]['W'].shape[0]
        self.add_layer_to_model(nodes, input_dim)


    def evaluate_classifier(self, X, mode='test'):
        input = X
        for i in range(len(self.model)-1):
            layer = self.model[i]
            layer['s'] = self.linear(layer['W'], input, layer['b'])
            if self.BN:
                layer['s^'] = self.batch_normalize(layer['s'], mode, layer)
                layer['X'] = self.relu_activation(layer['s^'])
            else:
                layer['X'] = self.relu_activation(layer['s'])
            input = layer['X']
        self.model[-1]['s'] = self.linear(self.model[-1]['W'], input, self.model[-1]['b'])
        return self.softmax(self.model[-1]['s'])



    def compute_cost(self, X, Y):
        P = self.evaluate_classifier(X)
        entropy = -np.mean(np.apply_along_axis(np.log, 0,np.multiply(Y, P).sum(axis=0)), axis=0)
        regularization = 0
        if self.LAMBDA > 0:
            for m in self.model:
                regularization += np.sum(m['W']*m['W'])
        return (entropy + self.LAMBDA*regularization)


    def compute_accuracy(self, set):
        P = self.evaluate_classifier(set.X)
        prediction_argmax = np.argmax(P, axis=0)
        true_argmax = np.argmax(set.Y, axis=0)
        return (np.sum(prediction_argmax==true_argmax)/set.Y.shape[1])


    def compute_gradients(self, X, Y, P):
        gradW = []
        gradb = []

        N = np.float64(X.shape[1])
        K = len(self.model)-1

        g = -(Y-P).T

        gradb.append(g.sum(axis=0, keepdims=True).T/N)
        assert gradb[-1].shape == self.model[K]['b'].shape

        if K == 0:
            gradW.append(np.dot(g.T, X.T)/N + 2*self.LAMBDA*self.model[K]['W'])
            assert gradW[-1].shape == self.model[K]['W'].shape
            return gradb, gradW
        gradW.append(np.dot(g.T, self.model[K-1]['X'].T)/N + 2*self.LAMBDA*self.model[K]['W'])
        assert gradW[-1].shape == self.model[K]['W'].shape

        g = np.dot(g, self.model[K]['W'])

        _s = self.model[K-1]['X']
        _s[_s > 0] = 1
        g = np.multiply(g, _s.T)


        for l in reversed(range(0,K)):
            if self.BN:
                g = self.batch_norm_back_pass(g, self.model[l])

            gradb.insert(0, g.sum(axis=0, keepdims=True).T/N)
            assert gradb[0].shape == self.model[l]['b'].shape

            if l > 0:
                gradW.insert(0, np.dot(g.T, self.model[l-1]['X'].T)/N + 2*self.LAMBDA*self.model[l]['W'])
                assert gradW[0].shape == self.model[l]['W'].shape

                g = np.dot(g, self.model[l]['W'])

                _s = self.model[l-1]['X']
                _s[_s > 0] = np.float64(1)
                g = np.multiply(g, _s.T)

            else:
                gradW.insert(0, np.dot(g.T, X.T)/N + 2*self.LAMBDA*self.model[l]['W'])
                assert gradW[0].shape == self.model[l]['W'].shape

        return gradb, gradW

    def grad_check_store(self):
        self.model_copy = deepcopy(self.model)

    def grad_check_restore(self):
        self.model = deepcopy(self.model_copy)

    def gradient_check(self, batch_size=100, dim_reduction=100):
        xbatch = self.train.X[:dim_reduction,:batch_size]
        ybatch = self.train.Y[:dim_reduction,:batch_size]

        # reduce dimensions for precision
        for m in self.model:
            m['W'] = np.copy(m['W'][:,:dim_reduction])

        # compute analytic gradients
        P = self.evaluate_classifier(xbatch, mode='train')
        a_gradb, a_gradW = self.compute_gradients(xbatch, ybatch, P)

        h_step = np.float64(1e-5)
        eps = np.float64(1e-5)

        # bias terms
        for g in range(len(a_gradb)):
            errors = []
            for m in range(self.model[g]['b'].shape[0]):
                self.grad_check_store()
                _b = np.copy(self.model[g]['b'][m])
                self.model[g]['b'][m] = _b + h_step
                c1 = self.compute_cost(xbatch, ybatch)
                self.grad_check_restore()
                self.model[g]['b'][m] = _b - h_step
                c2 = self.compute_cost(xbatch, ybatch)
                self.grad_check_restore()

                n_grad = np.divide(c1-c2, np.float64(2*h_step))

                relative_error = np.abs(n_grad-a_gradb[g][m]) / np.maximum(eps, (np.abs(n_grad) + np.abs(a_gradb[g][m])))
                errors.append(relative_error)
            print("$b_%d$" % (g+1), "& {:.3e}".format(np.amax(errors)), "& {:.3e}".format(np.mean(errors)), "& {:.3e} \\\\   \hline".format(np.var(errors)))

        for g in range(len(a_gradW)):
            errors = []
            for i in range(self.model[g]['W'].shape[0]):
                for j in range(self.model[g]['W'].shape[1]):
                    self.grad_check_store()
                    _W = np.copy(self.model[g]['W'][i,j])
                    self.model[g]['W'][i,j] = _W + h_step
                    c1 = self.compute_cost(xbatch, ybatch)
                    self.grad_check_restore()
                    self.model[g]['W'][i,j] = _W - h_step
                    c2 = self.compute_cost(xbatch, ybatch)
                    self.grad_check_restore()

                    n_grad = np.divide(c1-c2, np.float64(2*h_step))

                    relative_error = np.abs(n_grad-a_gradW[g][i,j]) / np.maximum(eps, (np.abs(n_grad) + np.abs(a_gradW[g][i,j])))
                    errors.append(relative_error)
            print("$W_%d$" % (g+1), "& {:.3e}".format(np.amax(errors)), "& {:.3e}".format(np.mean(errors)), "& {:.3e} \\\\   \hline".format(np.var(errors)))

    def update_weights(self, gradb, gradW, momentum):
        for i in range(len(self.model)):
            m = self.model[i]
            if momentum:
                m['momentum_W'] = self.RHO*m['momentum_W'] + self.ETA*gradW[i]
                m['momentum_b'] = self.RHO*m['momentum_b'] + self.ETA*gradb[i]
                m['W'] -= m['momentum_W']
                m['b'] -= m['momentum_b']
            else:
                m['W'] -= self.ETA*gradW[i]
                m['b'] -= self.ETA*gradb[i]


    def train_network(self, n_epochs=10, batch_size=100, momentum=True):
        pre_train_cost = self.compute_cost(self.train.X, self.train.Y)
        pre_valid_cost = self.compute_cost(self.valid.X, self.valid.Y)
        pre_train_acc  = self.compute_accuracy(self.train)
        pre_valid_acc  = self.compute_accuracy(self.valid)
        train_cost = [np.asscalar(pre_train_cost)]
        valid_cost = [np.asscalar(pre_valid_cost)]
        train_acc  = [np.asscalar(pre_train_acc)]
        valid_acc  = [np.asscalar(pre_valid_acc)]

        print("epoch: 0 train loss:", pre_train_cost, "valid loss:", pre_valid_cost)

        for i in range(n_epochs):
            for j in range(1, int(self.train.X.shape[1]/batch_size)+1):
                j_start = (j-1)*batch_size
                j_end = j*batch_size
                xbatch = self.train.X[:, j_start:j_end];
                ybatch = self.train.Y[:, j_start:j_end];

                P = self.evaluate_classifier(xbatch, mode='train')
                gradb, gradW = self.compute_gradients(xbatch, ybatch, P)
                self.update_weights(gradb, gradW, momentum)


            # cost over time
            t_cost = self.compute_cost(self.train.X, self.train.Y)
            v_cost = self.compute_cost(self.valid.X, self.valid.Y)
            t_acc  = self.compute_accuracy(self.train)
            v_acc  = self.compute_accuracy(self.valid)
            print("epoch:", i+1 ,"train loss:", t_cost, "valid loss:", v_cost)
            train_cost.append(np.asscalar(t_cost))
            valid_cost.append(np.asscalar(v_cost))
            train_acc.append(np.asscalar(t_acc))
            valid_acc.append(np.asscalar(v_acc))

            # decay rate
            if momentum:
                self.ETA = self.ETA*self.DECAY
        return train_cost, valid_cost, train_acc, valid_acc


def plotCost(train, valid, filepath):
    print(filepath)

    epochs = np.arange(0,len(train))
    train = np.array(train)
    valid = np.array(valid)

    #axes = plt.gca()
    #axes.set_ylim([1,2.4])
    plt.clf()
    plt.plot(epochs, train, 'r-', label='training loss')
    plt.plot(epochs, valid, 'b-', label='validation loss')
    plt.legend(loc='upper right', shadow='true')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    #plt.show()
    plt.savefig(filepath, bbox_inches='tight')

def plotAcc(train, valid, filepath):
    epochs = np.arange(0,len(train))
    train = np.array(train)
    valid = np.array(valid)

    plt.clf()
    plt.plot(epochs, train, 'r-', label='training')
    plt.plot(epochs, valid, 'b-', label='validation')
    plt.legend(loc='upper left', shadow='true')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    #plt.show()
    plt.savefig(filepath, bbox_inches='tight')

def parameter_search(settings):
    LAMBDA = np.random.uniform(settings['LAMBDA_MIN'], settings['LAMBDA_MAX'], (settings['SIZE'],1))
    ETA = np.random.uniform(settings['ETA_MIN'], settings['ETA_MAX'], (settings['SIZE'],1))

    results = []

    train, valid, test = Dataloader().get_data()
    for i in range(settings['SIZE']):
        print("Round %d of %d" % (i+1, settings['SIZE']))
        _ETA = pow(10,ETA[i])
        _LAMBDA = pow(10,LAMBDA[i])
        network = Network(train, valid, test, eta=_ETA, rho=0.7, lamb=_LAMBDA, BN=True, He_init=True)
        network.add_layer(50)
        network.add_layer(30)
        network.build_model()
        _, _, _, v_a = network.train_network(n_epochs=settings['EPOCHS'])
        results.append((v_a[-1],[ETA[i], _ETA, LAMBDA[i], _LAMBDA]))

    file = open(settings['FILENAME'], 'w')
    results.sort()
    for r in results:
        file.write('loss: %s ' % r[0])
        file.write('%s\n' % r[1])
    file.close()

def two_layer_testing():

    # Pairs of eta and lambda:
    settings = [{'name':'small', 'eta': 0.0005},
                {'name':'medium','eta': 0.03},
                {'name':'large', 'eta': 0.8}]

    train, valid, test = Dataloader().get_all_batch_data()
    for setting in settings:
        for bn in [True, False]:
            network = Network(train, valid, test, eta=setting['eta'], rho=0.8, lamb=0, BN=bn, He_init=False)
            network.add_layer(50)
            network.build_model()
            t_cost, v_cost, t_acc, v_acc = network.train_network(n_epochs=10, batch_size=100, momentum=True)
            print("test:", network.compute_accuracy(test))
            print("train:", network.compute_accuracy(train))
            plotCost(t_cost, v_cost, "K=2_" + setting['name'] + "_BN=" + str(bn) + "_loss.png")
            plotAcc(t_acc, t_acc, "K=2_" + setting['name'] + "_BN=" + str(bn) + "_accuracy.png")


def perform_gradchecks():
    train, valid, test = Dataloader().get_data()
    print("\n[3072 -> 10]: lambda=0")
    network = Network(train, valid, test, eta=0, lamb=0, gradient_check=True, He_init=True, BN=True)
    network.build_model()
    network.gradient_check()
    print("\n[3072 -> 10]: lambda=0.01")
    network = Network(train, valid, test, eta=0, lamb=0.01, gradient_check=True, He_init=True, BN=True)
    network.build_model()
    network.gradient_check()
    print("\n[3072 -> 50 -> 10]: lambda=0")
    network = Network(train, valid, test, eta=0, lamb=0, gradient_check=True, He_init=True, BN=True)
    network.add_layer(50)
    network.build_model()
    network.gradient_check()
    print("\n[3072 -> 50 -> 10]: lambda=0.01")
    network = Network(train, valid, test, eta=0, lamb=0.01, gradient_check=True, He_init=True, BN=True)
    network.add_layer(50)
    network.build_model()
    network.gradient_check()
    print("\n[3072 -> 50 -> 30 -> 10]: lambda=0")
    network = Network(train, valid, test, eta=0, lamb=0, gradient_check=True, He_init=True, BN=True)
    network.add_layer(50)
    network.add_layer(30)
    network.build_model()
    network.gradient_check()
    print("\n[3072 -> 50 -> 30 -> 10]: lambda=0.01")
    network = Network(train, valid, test, eta=0, lamb=0.01, gradient_check=True, He_init=True, BN=True)
    network.add_layer(50)
    network.add_layer(30)
    network.build_model()
    network.gradient_check()
    print("\n[3072 -> 50 -> 30 -> 20 -> 10]: lambda=0")
    network = Network(train, valid, test, eta=0, lamb=0, gradient_check=True, He_init=True, BN=True)
    network.add_layer(50)
    network.add_layer(30)
    network.add_layer(10)
    network.build_model()
    network.gradient_check()
    print("\n[3072 -> 50 -> 30 -> 20 -> 10]: lambda=0.01")
    network = Network(train, valid, test, eta=0, lamb=0.01, gradient_check=True, He_init=True, BN=True)
    network.add_layer(50)
    network.add_layer(30)
    network.add_layer(10)
    network.build_model()
    network.gradient_check()

def main():

    #perform_gradchecks()

    '''
    # second fine-search
    settings = { 'LAMBDA_MAX': -3.6,
                 'LAMBDA_MIN': -5.3,
                 'ETA_MAX': 0,
                 'ETA_MIN': -1.7,
                 'SIZE': 300,
                 'EPOCHS': 14,
                 'FILENAME': 'fine_search_2.txt'}

    parameter_search(settings)
    '''

    '''
    # second fine-search
    settings = { 'LAMBDA_MAX': -3.6,
                 'LAMBDA_MIN': -5.3,
                 'ETA_MAX': 0,
                 'ETA_MIN': -1.7,
                 'SIZE': 300,
                 'EPOCHS': 14,
                 'FILENAME': 'fine_search_2.txt'}

    parameter_search(settings)
    '''

    two_layer_testing()


if __name__ == "__main__":
    main()
