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

	def dw_conv(self, activations, stage, block, padding = 'SAME', stride = 1):
		in_ch = activations.shape[3]
		layer_name = str(stage) + '/' + str(block) + '/dconv/W:0'
		kernels = self.trained_model[layer_name]
		kernel_size = kernels.shape[0]
		act_split = np.split(activations, indices_or_sections = in_ch, axis = 3)
		kern_split = np.split(kernels, indices_or_sections = kernels.shape[2], axis = 2)
		conv_res = []
		for ch in range(0, len(act_split)):
			conv_res.append(conv2d.spatial_conv(act_split[ch], kern_split[ch], padding = 1, stride = stride))
		return np.concatenate(conv_res, axis=3)

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

	def relu(self, activations):
		activations[activations < 0] = 0
		return activations

	def sub_sample(data, pool, stride, type='MAX'):                     #Input and Ouput both in format [1, Rows, Cols, C]
		numTilesR = np.floor((data.shape[2] - pool)/stride) + 1
		numTilesC = np.floor((data.shape[1] - pool)/stride) + 1
		numTiles = numTilesR * numTilesC
		C = data.shape[3]
		ro = 0
		co = 0
		output = np.zeros((1, int(numTilesR), int(numTilesC), C))
		if type == 'MAX':
			pool = lambda a, ro, co, pool, ch: np.amax(a[0, ro:ro+pool, co:co+pool, ch])
		elif type == 'AVG':
			pool = lambda a, ro, co, pool, ch: np.average(a[0, ro:ro+pool, co:co+pool, ch])
		for c in range(0, C):
			co = 0
			ro = 0
			for t in range(0, int(numTiles)):
				col = int((t % numTilesR) * stride)
				row = int(np.floor(t/numTilesR) * stride)
				output[0, ro, co, c] = pool(data, row, col, pool, c)
				co = co+1
				if(co >= numTilesR):
					co = 0
					ro = ro + 1
		return output

	def softmax(self, activations):
		return (np.exp(activations) / np.sum(np.exp(activations), axis = 1))

def main():
	act = np.float32(np.arange(28*28*384).reshape(1, 28, 28, 384))
	act2 = np.float32(np.arange(28*28*192).reshape(1, 28, 28, 192))
	arch = Shufflenet('./../ShuffleNetV1-1x-8g.npz')
	res = arch.pw_gconv(act, 'stage3', 'block0', 'conv1', 8)
	res2 = arch.batch_normalization(act2, 'stage3', 'block0', 'conv1')
	res3 = arch.dw_conv(act2, 'stage3','block0', padding = 'SAME', stride = 1)
	print(res3[0, 10:14, 10:14, 128:132])

if __name__ == '__main__':
	main()

