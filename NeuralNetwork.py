import numpy as np
import h5py
import time
#Just use torch to access GPU


def softmax(x):
    #take vector x, gives output vector same size
    out = np.exp(x)
    sum = np.sum(out)
    return out/sum


def e(y,K):
    #y scalar, K vector output size
    out = np.zeros(K)
    out[y] =1
    return out


def sigprim(z):
    #z vector, output vector same size,sigma= RELU
    out = np.zeros(len(z))
    out[z>0] = 1
    return out


def sigma(z):
    #z vector, output vector same size ,sigma= RELU
    out = np.zeros(len(z))
    return np.maximum(z,out)


def main():
    MNIST_data = h5py.File("MNISTdata.hdf5", "r")
    X_train = np.float32(MNIST_data["x_train"][:])
    y_train = np.int32(np.array(MNIST_data["y_train"][:, 0]))
    x_test = np.float32(MNIST_data['x_test'][:])
    y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))

    d = 28 * 28  # dd
    K = 10  # Kk
    dh = 50

    Z = np.zeros(dh)
    H = np.zeros(dh)
    U = np.zeros(K)
    f = np.zeros(K)

    C = np.random.rand(K,dh)-0.5*np.ones((K,dh))
    W = np.random.rand(dh,d)-0.5*np.ones((dh,d))
    b1 = np.random.rand(dh)-0.5*np.ones((dh))
    b2 = np.random.rand(K) - 0.5 * np.ones((K))

    Epochs  =500
    lr = 0.02
    for j in range(Epochs):
        lr*=0.95
        time1 = time.time()
        for i in range(len(y_train)):
            x = X_train[i,:]
            y = y_train[i]
            Z = np.matmul(W,x)+b1
            H = sigma(Z)
            U = np.matmul(C,H)+b2
            f = softmax(U)

            dRhoU = -(e(y,K)-f)
            dRhob2 = dRhoU
            dRhoC = np.outer(dRhoU,H)
            delta = np.matmul(C.T,dRhoU)
            dRhob1 = np.multiply(delta,sigprim(Z))
            dRhoW = np.outer(dRhob1,x)

            b1-=lr*dRhob1
            C-=lr*dRhoC
            b2-=lr*dRhob2
            W-=lr*dRhoW

        total_correct = 0
        time2 = time.time()
        #W/o vectorization 4.84 per epoch. with dh = 5
        print("Time for one epoch: ", time2-time1)
        for i in range(len(y_test)):
            y = y_test[i]
            x = x_test[i,:]
            Z = np.matmul(W, x) + b1
            H = sigma(Z)
            U = np.matmul(C, H) + b2
            f = softmax(U)
            prediction = np.argmax(f)
            if prediction==y:
                total_correct+=1
        print("Accuracy with ",j+1," Epochs is: ", str(total_correct/len(y_test)))
        #97.2 percent
main()