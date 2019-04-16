import spatial_conv as conv2d
import numpy as np
import utils

SHUFFLENET_MEAN = [103.939, 116.779, 123.68]
NORMALIZER = 0.017

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
#		print(act_split[0].shape, kernels_split[0].shape)
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

	def shufflenet_unit(self, activations, stage, block, stride, num_groups=8):
		residual = activations
		num_split = num_groups if activations.shape[3] > 24 else 1
		pwgconv1 = self.pw_gconv(activations, stage, block, 'conv1', num_split)
		bnconv1 = self.batch_normalization(pwgconv1, stage, block, 'conv1')
		reluconv1 = self.relu(bnconv1)
		ch_sh = self.channel_shuffle(reluconv1, num_groups)
		dconv = self.dw_conv(ch_sh, stage, block, padding = 'SAME', stride = stride)
		bndconv = self.batch_normalization(dconv, stage, block, 'dconv')
		pwgconv2 = self.pw_gconv(bndconv, stage, block, 'conv2', num_groups)
		bnconv2 = self.batch_normalization(pwgconv2, stage, block, 'conv2')

		if stride == 1:
			return self.relu(bnconv2 + residual)
		elif stride == 2:
			residual = self.sub_sample(residual, pool = 3, stride = 2, padding = 1, type = 'AVG')
			return np.concatenate([residual, self.relu(bnconv2)], axis = 3)
		else:
			raise ValueError("Stride value can only be 1 or 2 for Shufflenet")

	def shufflenet_stage(self, activations, stage, repeat, num_groups=8):
		first_block = self.shufflenet_unit(activations, stage, 'block0', stride = 2, num_groups = 8)
		res = first_block
		for b in range(1, repeat+1):
			res = self.shufflenet_unit(res, stage, 'block' + str(b), stride = 1, num_groups = 8)
		return res

	def shufflenet_stage1(self, activations):
		kernels = self.trained_model['conv1/W:0']
		res = conv2d.spatial_conv(activations, kernels, padding = 1, stride = 2)
		res = self.batch_normalization(res, '', '', 'conv1')
		res = self.sub_sample(res, pool = 3, stride = 2, padding = 1, type = 'MAX')
		return res

	def channel_shuffle(self, activations, num_groups = 8):
		activations = np.transpose(activations, (0, 3, 1, 2))
		in_shape = activations.shape
		in_channel = in_shape[1]
#		print(type(in_shape))
		l = np.reshape(activations, (-1, in_channel // num_groups, num_groups) + in_shape[-2:])
		l = np.transpose(l, [0, 2, 1, 3, 4])
		l = l.reshape(((-1, in_channel) + in_shape[-2:]))
		l = l.transpose((0, 2, 3, 1))
		return l

	def forward_pass(self, image):
		red, green, blue = np.split(image, axis=3, indices_or_sections=3)
		bgr = np.concatenate([(blue - SHUFFLENET_MEAN[0])*NORMALIZER, (green - SHUFFLENET_MEAN[1])*NORMALIZER, (red - SHUFFLENET_MEAN[2])*NORMALIZER], axis = 3)
		stage1 = self.shufflenet_stage1(bgr)
#		print(stage1[0, 10:14, 10:14, 10:14])
		stage2 = self.shufflenet_stage(stage1, 'stage2', repeat = 3, num_groups = 8)
		stage3 = self.shufflenet_stage(stage2, 'stage3', repeat = 7, num_groups = 8)
		stage4 = self.shufflenet_stage(stage3, 'stage4', repeat = 3, num_groups = 8)
		g_pool = self.sub_sample(stage4, pool = 7, stride = 1, padding = 0, type = 'AVG')
		logits = self.fc_layer(g_pool)
		logits = self.softmax(logits)
		return logits

	def relu(self, activations):
		activations[activations < 0] = 0
		return activations

	def sub_sample(self, data, pool, stride, padding = 0, type='MAX'):                     #Input and Ouput both in format [1, Rows, Cols, C]
		data  = np.pad(data[:, :, :, :],((0, 0),(padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=(0))
		numTilesR = np.floor((data.shape[2] - pool)/stride) + 1
		numTilesC = np.floor((data.shape[1] - pool)/stride) + 1
		numTiles = numTilesR * numTilesC
		C = data.shape[3]
		ro = 0
		co = 0
		output = np.zeros((1, int(numTilesR), int(numTilesC), C))
		if type == 'MAX':
			_pool = lambda a, ro, co, wind, ch: np.amax(a[0, ro:ro+wind, co:co+wind, ch])
		elif type == 'AVG':
			_pool = lambda a, ro, co, wind, ch: np.average(a[0, ro:ro+wind, co:co+wind, ch])
		for c in range(0, C):
			co = 0
			ro = 0
			for t in range(0, int(numTiles)):
				col = int((t % numTilesR) * stride)
				row = int(np.floor(t/numTilesR) * stride)
				output[0, ro, co, c] = _pool(data, row, col, pool, c)
				co = co+1
				if(co >= numTilesR):
					co = 0
					ro = ro + 1
		return output

	def fc_layer(self, activations):    #Input output data in format [Rows, Cols, Ch]
		layer_name = 'linear'
		weights = self.trained_model[layer_name + '/W:0']
		biases = self.trained_model[layer_name + '/b:0']
		numElm = 1
		for i in range(0, activations.ndim):
			numElm = numElm*activations.shape[i]
		activations = activations.reshape((1, numElm))
		mul1 = np.dot(activations, weights)
		return np.add(mul1, biases)

	def softmax(self, activations):
		return (np.exp(activations) / np.sum(np.exp(activations), axis = 1))

def main():
	img = utils.load_image('./../tf_shufflenet/test_data/32.JPEG')
	img = img.reshape((1, 224, 224, 3))
	img = np.float32(img) * 255.0

	act = np.float32(np.arange(28*28*384).reshape(1, 28, 28, 384))
	act2 = np.float32(np.arange(28*28*192).reshape(1, 28, 28, 192))
	arch = Shufflenet('./../ShuffleNetV1-1x-8g.npz')
	res = arch.pw_gconv(act, 'stage3', 'block0', 'conv1', 8)
	res2 = arch.batch_normalization(act2, 'stage3', 'block0', 'conv1')
	res3 = arch.dw_conv(act2, 'stage3','block0', padding = 'SAME', stride = 1)
#	print(res3[0, 10:14, 10:14, 128:132])

	prob = arch.forward_pass(img)
	utils.print_prob(prob[0], '../tf_shufflenet/synset.txt')

if __name__ == '__main__':
	main()

