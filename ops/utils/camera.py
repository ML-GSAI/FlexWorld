import numpy as np
from scipy.linalg import logm,expm

def SE3_to_se3(T):
    se3_mat = logm(T) 
    v = se3_mat[0:3, 3] 
    w_hat = se3_mat[0:3, 0:3] 
    
    w = np.array([
        w_hat[2, 1],
        w_hat[0, 2],
        w_hat[1, 0]
    ])
    
    se3_vec = np.concatenate((v, w))
    return se3_vec

def se3_to_SE3(se3_vec):
    v = se3_vec[0:3]  
    w = se3_vec[3:6]  

    w_hat = np.array([
        [ 0,     -w[2],  w[1]],
        [ w[2],   0,    -w[0]],
        [-w[1],  w[0],   0   ]
    ])

    se3_mat = np.zeros((4, 4))
    se3_mat[0:3, 0:3] = w_hat
    se3_mat[0:3, 3] = v

    T = expm(se3_mat)
    return T


def rotate_matrix(angle, axis, R):
    if axis == 'x':
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angle), -np.sin(angle)],
                       [0, np.sin(angle), np.cos(angle)]])
        return Rx @ R
    elif axis == 'y':
        Ry = np.array([[np.cos(angle), 0, np.sin(angle)],
                       [0, 1, 0],
                       [-np.sin(angle), 0, np.cos(angle)]])
        return Ry @ R
    elif axis == 'z':
        Rz = np.array([[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle), np.cos(angle), 0],
                       [0, 0, 1]])
        return Rz @ R
    else:
        raise ValueError("axis error")
