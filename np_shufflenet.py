import spatial_conv as conv2d
import numpy as np

class Shufflenet:
	def __init__(self, model_loc):
		self.trained_model = np.load(model_loc, encoding='latin1')
		print('Pre-trained npz model loaded')

	def pw_gconv(self, activations, stage, block, layer, num_groups):
		layer_name = str(stage) + '/' + str(block) + '/' + str(layer) + '/W:0'
		kernels = self.trained_model[layer_name]
		ch_per_group = activations.shape[3] // num_groups
		act_split = np.split(activations, indices_or_sections = num_groups, axis = 3)
		kernels_split = np.split(kernels, indices_or_sections = num_groups, axis = 3)
		convs = []
		print(act_split[0].shape, kernels_split[0].shape)
		for grp in range(0, num_groups):
			convs.append(conv2d.spatial_conv(act_split[grp], kernels_split[grp], padding = 0, stride= 1))
		return np.concatenate(convs, axis=3)

	def batch_normalization(self, activations, stage, block, layer):
		layer_name = str(stage) + '/' + str(block) + '/' if stage is not '' else ''
		layer_name = layer_name + 'conv1/bn/' if layer == 'conv1' else layer_name + layer + '_bn/'
		bn_out = self.trained_model[layer_name + 'gamma:0']*(activations - self.trained_model[layer_name + 'mean/EMA:0']) / (self.trained_model[layer_name + 'variance/EMA:0'] + 0.001)**0.5 + self.trained_model[layer_name + 'beta:0']
		return bn_out

	def channel_shuffle(self, activations, num_groups = 8):
		activations = np.transpose(activations, (0, 3, 1, 2))
		in_shape = activations.shape
		in_channel = in_shape[1]
		l = np.reshape(activations, [-1, in_channel // num_groups, num_groups] + in_shape[-2:])
		l = np.transpose(l, [0, 2, 1, 3, 4])
		l = l.reshape(([-1, in_channel] + in_shape[-2:]))
		l = l.transpose((0, 2, 3, 1))
		return l

def main():
	act = np.float32(np.arange(28*28*384).reshape(1, 28, 28, 384))
	act2 = np.float32(np.arange(28*28*192).reshape(1, 28, 28, 192))
	arch = Shufflenet('./../ShuffleNetV1-1x-8g.npz')
	res = arch.pw_gconv(act, 'stage3', 'block0', 'conv1', 8)
	res2 = arch.batch_normalization(act2, 'stage3', 'block0', 'conv1')
	print(res2[0, 10:14, 10:14, 128:132])

if __name__ == '__main__':
	main()

