import numpy as np

def spatial_conv(input, kernels, padding = 0, stride = 1):
        padded_inp  = np.pad(input[0, :, :, :],((padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=(0))
        tup = (np.subtract(padded_inp.shape, (kernels.shape[0], kernels.shape[1], kernels.shape[2])))
        submat_shape = tuple((int(tup[0]/stride+1), int(tup[1]/stride+1), int(tup[2]/stride+1)))
        window_shape = (kernels.shape[0], kernels.shape[1], kernels.shape[2]) + submat_shape
        sub_matrices = np.lib.stride_tricks.as_strided(padded_inp, window_shape, padded_inp.strides+(padded_inp.strides[0]*stride, padded_inp.strides[1]*stride, padded_inp.strides[2]), writeable=False)
        conv_result = np.einsum('hijn,hijklm->kln', kernels, sub_matrices)
        return conv_result.reshape(1, conv_result.shape[0], conv_result.shape[1], conv_result.shape[2])
