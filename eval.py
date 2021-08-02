import torch
import numpy as np

from models import PointCMLP
from utils import score, create_test_set


MODEL_PATH = 'pretrained_models/mlgp_clean.tar'
# MODEL_PATH = 'pretrained_models/mlgp_pi4_noisy_02.tar'
# MODEL_PATH = 'pretrained_models/baseline_pi4_noisy_02.tar'
# MODEL_PATH = 'pretrained_models/vanilla_pi4_noisy_02.tar'



if __name__ == '__main__':
    # get the data:
    print('\ncreating test data............')
    Xtest, Ytest = create_test_set(distortion=0.0)

    # or, e.g., for the noisy theta-split experiment:
    # Xtest, Ytest = create_test_set(distortion=0.2,  theta_train=[[0.0, 1/2], [1/8, 5/8]], theta_test=[[1/8, 5/8], [1/2, 1.0]])
    
    # other options (depending on the pretrained model):
    # Xtest_clean,           Ytest_clean           = create_test_set(distortion=None)
    # Xtest_noisy,           Ytest_noisy           = create_test_set(distortion=0.1)
    # Xtest_noisy_02,        Ytest_noisy_02        = create_test_set(distortion=0.2)
    # Xtest_pi4_clean,       Ytest_pi4_clean       = create_test_set(distortion=None, theta_train=[[0.0, 1/2], [1/8, 5/8]], theta_test=[[1/8, 5/8], [1/2, 1.0]])
    # Xtest_pi4_noisy,       Ytest_pi4_noisy       = create_test_set(distortion=0.1,  theta_train=[[0.0, 1/2], [1/8, 5/8]], theta_test=[[1/8, 5/8], [1/2, 1.0]])
    
    if 'baseline' in MODEL_PATH:
        sample_size = torch.tensor(Xtest[0].shape).prod().item()
        Xtest = Xtest.reshape(-1, 1, sample_size)

    # load the model:
    if not torch.cuda.is_available():
        model_dic = torch.load(MODEL_PATH, map_location ='cpu')
    else:
        model_dic = torch.load(MODEL_PATH)
    model = model_dic['model']

    if torch.cuda.is_available():
        model = model.cuda()
        Xtest, Ytest = Xtest.cuda(), Ytest.cuda()

    # evaluate on the test data:
    Ytest_pred = model(Xtest).detach_()
    test_acc = score(Ytest_pred, Ytest)

    print('\nmodel:', model_dic['name'], '\n\ntest acc:', np.round(test_acc, 5))
