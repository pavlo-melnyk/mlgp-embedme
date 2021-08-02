import torch
import torch.nn as nn
import scipy.linalg
import numpy as np

from matplotlib.legend_handler import HandlerPathCollection
from sklearn.utils import shuffle
from models import PointCMLP



EPSILON = 1e-8



def identity(x):
    # needed in this format to save the model properly
    return x



def build_mlgp(input_shape=(4, 3), output_dim=8, hidden_layer_sizes=[4], bias=False, activation=identity):
    # Multilayer Geometric Perceptron (ours)
    print('\nmodel: MLGP (ours)')
    model = PointCMLP(input_shape, output_dim, hidden_layer_sizes, activation, bias, version=1)
    return model



def build_vanilla(input_shape=(1, 12), output_dim=8, hidden_layer_sizes=[6], bias=True, activation=nn.functional.relu):
    # Vanilla Multilayer Perceptron
    print('\nmodel: Vanilla MLP')
    model = PointCMLP(input_shape, output_dim, hidden_layer_sizes, activation, bias, version=0)
    return model



def build_baseline(input_shape=(1, 12), output_dim=8, hidden_layer_sizes=[5], bias=False, activation=identity):
    # Multilayer Hypersphere Perceptron
    print('\nmodel: Baseline (MLHP)')
    model = PointCMLP(input_shape, output_dim, hidden_layer_sizes, activation, bias, version=1)
    return model



def score(y, t):
    return torch.mean((torch.argmax(y, axis=1) == t).double()).item()



def save_checkpoint(state, save_dir='pretrained_models'):
    torch.save(state, save_dir+'/'+state['name']+'.tar')



def random_rotation_matrix(low=[0.0], high=[1.0]):  
    """
    Inspired by
    https://github.com/tensorfieldnetworks/tensorfieldnetworks/blob/master/tensorfieldnetworks/utils.py

    Generates a random 3D rotation matrix.

    Args:
        low, high:  intergers, or floats, or tuples/lists;
                    the lower and upper bounds of the random rotation angle,
                    specified as fractions of 2*pi;
                    in case of tuples/lists, the intervals are formed
                    by taking the bounds from low and high pair-wise, e.g., 
                    low=[0.0, 1/4], high=[1/8, 1.0] corresponds to 
                    [0, 2*pi/8) U [2*pi/4, 2*pi) = [0, pi/4) U [pi/2, 2*pi).
                    The angle is drawn from the distribution over the joint interval.
    Returns:
        Random rotation matrix.
    """
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis) + EPSILON

    theta = 2 * np.pi * np.random.uniform(low, high)
    theta = np.random.choice(np.atleast_1d(theta))

    return rotation_matrix(axis, theta)



def rotation_matrix(axis, theta):
    return scipy.linalg.expm(np.cross(np.eye(3), axis * theta))



def get_tetris_data(total_size=10000, train_size=1000, shuffle_data=False,
                    distortion=None, theta_train=([0.0], [1.0]), theta_test=([0.0], [1.0]),  
                    only_canonical=False, only_label_names=False):
    
    '''
    Inspired by
    https://github.com/tensorfieldnetworks/tensorfieldnetworks/blob/master/shape_classification.ipynb

    Args:
        distortion:                 float; noise n~U(-distortion, distortion) applied to shape coordinates point-wise; 
        theta_train and theta_test: tuples/lists containing two tuples/lists of lower and upper bounds
                                    of the rotation angle interval(s) for train and test sets, respectively;
        only_canonical:             boolean; if True, returns only the 8 non-transformed Tetris shapes;
        only_label_names:           boolean; if True, returns only the names of the 8 shapes.
    '''
    assert train_size < total_size
    
    label_names = ['chiral_shape_1', 'chiral_shape_2', 'square', 'line', 'corner', 'L', 'T', 'zigzag']

    if only_label_names:
        return label_names

    tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
              [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)], # chiral_shape_2
              [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
              [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
              [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]  # zigzag

    dataset = [np.array(points_) for points_ in tetris]

    Xtrain = np.array(dataset)
    Ytrain = np.arange(len(dataset)) # [0, 1, ..., 7]
    
    Xtest, Ytest= [], []
    
    # augment data by applying random rigid body transformations;
    # rotation angle ranges for train and test sets are given by
    # 2*np.pi*theta_test and 2*np.pi*theta_test, respectively:
    j = 0
    for i in range(total_size//8):
        for label, shape in enumerate(dataset):

            if j < train_size:
                rotation = random_rotation_matrix(low=theta_train[0], high=theta_train[1])
            else:
                rotation = random_rotation_matrix(low=theta_test[0], high=theta_test[1])

            rotated_shape = shape @ rotation            
            translation = np.expand_dims(np.random.uniform(low=-3., high=3., size=(3)), axis=0)
            translated_shape = rotated_shape + translation 

            if distortion:
                translated_shape += np.random.uniform(low=-distortion, high=distortion, size=(4, 3))

            Xtest.append(translated_shape)
            Ytest.append(label)

            j += 1
            
    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest) 
                
    if only_canonical:
        return (torch.from_numpy(Xtrain).float(), torch.from_numpy(Ytrain).long())
    
    Xtrain, Ytrain = Xtest[:train_size], Ytest[:train_size]
    Xtest, Ytest = Xtest[train_size:], Ytest[train_size:]

    if shuffle_data:
            Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
            Xtest, Ytest = shuffle(Xtest, Ytest)
    
    return (torch.from_numpy(Xtrain).float(), torch.from_numpy(Ytrain).long()), \
           (torch.from_numpy(Xtest).float(), torch.from_numpy(Ytest).long())



def create_test_set(distortion=None, theta_train=([0.0], [1.0]), theta_test=([0.0], [1.0])):
    data_seeds = [15567, 23495, 80990, 1010394, 1996, 2020, 1969, 1986, 53250, 793254]
    
    Xs, Ys = [], []
    for data_seed in data_seeds:
        np.random.seed(data_seed)
        _, (Xtest, Ytest) = get_tetris_data(total_size=10000, train_size=1000, shuffle_data=True, 
                                            theta_train=theta_train, theta_test=theta_test,
                                            distortion=distortion)
        Xs.append(Xtest)
        Ys.append(Ytest)
        
    Xs, Ys = torch.cat(Xs, dim=0), torch.cat(Ys, dim=0)
    Xs, Ys = Xs.view(-1, 4, 3), Ys.view(-1)        
                    
    return (Xs.float(), Ys)

# usage of the above function:

# Xtest_clean,           Ytest_clean           = create_test_set(distortion=None)
# Xtest_noisy,           Ytest_noisy           = create_test_set(distortion=0.1)
# Xtest_noisy_02,        Ytest_noisy_02        = create_test_set(distortion=0.2)

# Xtest_pi4_clean,       Ytest_pi4_clean       = create_test_set(distortion=None, theta_train=[[0.0, 1/2], [1/8, 5/8]], theta_test=[[1/8, 5/8], [1/2, 1.0]])
# Xtest_pi4_noisy,       Ytest_pi4_noisy       = create_test_set(distortion=0.1,  theta_train=[[0.0, 1/2], [1/8, 5/8]], theta_test=[[1/8, 5/8], [1/2, 1.0]])
# Xtest_pi4_noisy_02,    Ytest_pi4_noisy_02    = create_test_set(distortion=0.2,  theta_train=[[0.0, 1/2], [1/8, 5/8]], theta_test=[[1/8, 5/8], [1/2, 1.0]])



def construct_isomorphism_transformation(rotation, translation):
    '''
    Given 3D rotation and translation, constructs a matrix isomorphism of the transformation in R^{5} (the matrix itself is R^{5x5})
    corresponding to a motor in R^{3+1, 1} = ME^{3}.

    Args:
        rotation:    3D rotation, an array of shape (3, 3);
        translation: 3D translation, a vector of length 3;
    Returns:
        TR: a 5x5 matrix
    '''
    rotation_isom = construct_rotation_isom(rotation)
    translation_isom = construct_translation_isom(translation)

    TR = np.matmul(translation_isom, rotation_isom)
    
    return TR



def construct_rotation_isom(rotation):
    rotation = rotation.reshape(3, 3)
    bottom_part = np.zeros((2, 3))                                     
    rotation_isom = np.concatenate((rotation, bottom_part), axis=0)    

    right_part = np.eye(5)[:, -2:]                                     
    rotation_isom = np.concatenate((rotation_isom, right_part), axis=1)

    return rotation_isom 



def construct_translation_isom(translation):
    translation = translation.reshape(1, 3)
    base = np.eye(3)                              
                                                
    translation_isom = np.concatenate((base, translation, np.zeros((1, 3))), axis=0)    

    t_sq_mag = np.sum(translation**2, axis=-1, keepdims=True)                        
    right_part = np.concatenate((translation, 0.5*t_sq_mag, [[1.]]), axis=1)  

    translation_isom = np.concatenate((translation_isom, np.transpose([[0., 0., 0., 1., 0.]]), np.transpose(right_part)), axis=1)

    return translation_isom 



def embed_points(points):
    ''' 
    Performs conformal embedding -- embeds points in the conformal space.

    Args:
        points - the 3D model points, a tensor of shape (num_points, 3).
    Returns:
        embedded points - points embedded in R^{5}, a tensor of shape (num_points, 5).
    '''

    # compute the squared magnitude for each point:
    points_sq_mag = np.sum(points**2, axis=-1, keepdims=True)
    embedded_points = np.concatenate([points, 0.5*points_sq_mag, np.ones_like(points_sq_mag)], axis=-1)

    return embedded_points



def transform_points(points, transformation):
    ''' 
    Applies the isomorphism transformation to embedded points.

    Args:
        points:              points embedded in R^{5}, an array of shape (num_points, 5);
        transformation:      an array of shape (5, 5).
    Returns:
        transformed points:  a tensor of the same shape as the input points.
    '''

    # reshape to (1, 5, 5) to perform matmul properly:
    T = np.reshape(transformation, (-1, 5, 5))
    
    # expand dims to make a tensor of shape(num_points, 5, 1) to perform matmul properly:
    X = np.expand_dims(points, -1)

    # transform each point:
    transformed_points = np.matmul(T, X)
    
    # reshape to the input points size -- squeeze the last dimension:
    transformed_points = np.squeeze(transformed_points, -1)
        
    return transformed_points



def unembed_points(embedded_points):
    ''' 
    Performs a mapping that is inverse to conformal embedding.

    Args:
        embedded_points: points embedded in R^{5}, an array of shape (num_points, 5).
    Returns:
        points:          3D points, an array of shape (num_points, 3).

    '''

    # p-normalize points, i.e., divide by the last element:
    normalized_points = embedded_points / np.expand_dims(embedded_points[:,-1], axis=-1)

    # the first three elements are now Euclidean R^{3} coordinates:
    points = normalized_points[:,:3]

    return points



def set_axes_equal(ax):    
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



def draw_sphere(c, r, ax, color='b', ind=1, pole_marker_size=33, marker='s', alpha=0.025, arrow_length=4.5, draw_normal=False):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = r * np.outer(np.cos(u), np.sin(v)) + c[0]
    y = r * np.outer(np.sin(u), np.sin(v)) + c[1]
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + c[2]

    pole = ax.scatter(c[0], c[1], c[2]+r, s=pole_marker_size, marker=marker, alpha=1, label='sphere_'+str(ind), color=color)
    ax.plot_wireframe(x, y, z, alpha=alpha, color=color)

    if draw_normal:        
        if color=='r':
            ax.quiver(c[0]-r/np.sqrt(3), c[1]-r/np.sqrt(3), c[2]-r/np.sqrt(3),  -1, -1, -1, color=color, length=arrow_length, arrow_length_ratio=0.25)
        else:
            ax.quiver(c[0]-r/np.sqrt(3), c[1]-r/np.sqrt(3), c[2]-r/np.sqrt(3), 1, 1, 1, color=color, length=arrow_length, arrow_length_ratio=0.25)
    return pole



class HandlerMultiPathCollection(HandlerPathCollection):
    """
    Handler for PathCollections, which are used by scatter.
    """
    def create_collection(self, orig_handle, sizes, offsets, transOffset):
        p = type(orig_handle)(orig_handle.get_paths(), sizes=sizes,
                              offsets=offsets,
                              transOffset=transOffset,
                              )
        return p