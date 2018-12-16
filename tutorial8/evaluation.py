import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.autograd import Variable

# calculate accuracy score(f(x), y) of n samples from x for DANN model
def eval_clf(model1, model2, x, y, n):
    out = model1(model2(Variable(torch.from_numpy(x[:n]))).view(n, -1))
    preds = out.max(1)[1]
    return accuracy_score(y_true=[np.argmax(i) for i in y[:n]], y_pred=preds.data.numpy().ravel())

# plots generator output 
def plot_imgs(model, xt, xs): 
    
    def norm_img(x):
        return ((x - np.min(x))) / (np.max(x) - np.min(x))
    
    rand_img = np.random.randint(0, 19, 1)[0]

    # take image from target domain and transform with generator 
    img = model(Variable(torch.from_numpy(xt[:20]))).data.numpy()[rand_img]
    
    # plot generated, target, and source images
    subplot(1,3,1)
    plt.imshow(norm_img(img.transpose(1,2,0)))
    subplot(1,3,2)
    plt.imshow(norm_img(xt[rand_img].transpose(1,2,0)))
    subplot(1,3,3)
    plt.imshow(norm_img(xs[rand_img].transpose(1,2,0)))
    
    # print plot
    plt.show()
