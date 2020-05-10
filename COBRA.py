import torch
import torch.nn as nn
import numpy as np

from torch import optim
import utils
import torch.nn.functional as F
import torch.nn as nn
import dataloader
from torch.autograd import Variable
import scipy.io as sio
import copy
import math
import time

# parts of code referred from https://github.com/penghu-cs/SDML/blob/master/SDML.py

class Solver(object):
	def __init__(self, config):
		wv_matrix = None
		self.output_shape = config.output_shape
		data = dataloader.load_deep_features(config.datasets)
		self.datasets = config.datasets
		(self.train_data, self.train_labels, self.valid_data, self.valid_labels, self.test_data, self.test_labels, self.MAP) = data

		# number of modalities
		self.n_view = len(self.train_data)

		self.use_nce = config.use_nce

		# softmax contrastive loss implementation of Eq. 4
		self.cpc_loss_func = nn.BCEWithLogitsLoss()

		if torch.cuda.is_available():
			self.cpc_loss_func.cuda()

		for v in range(self.n_view):
			if min(self.train_labels[v].shape) == 1:
				self.train_labels[v] = self.train_labels[v].reshape([-1])
			if min(self.valid_labels[v].shape) == 1:
				self.valid_labels[v] = self.valid_labels[v].reshape([-1])
			if min(self.test_labels[v].shape) == 1:
				self.test_labels[v] = self.test_labels[v].reshape([-1])

		if len(self.train_labels[0].shape) == 1:
			self.classes = np.unique(np.concatenate(self.train_labels).reshape([-1]))
			self.classes = self.classes[self.classes >= 0]
			self.num_classes = len(self.classes)
		else:
			self.num_classes = self.train_labels[0].shape[1]

		if self.output_shape == -1:
			self.output_shape = self.num_classes

		self.dropout_prob = 0.5
		if wv_matrix is not None:
			self.vocab_size = wv_matrix.shape[0] - 2

		self.input_shape = [self.train_data[v].shape[1] for v in range(self.n_view)]

		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2
		self.batch_size = config.batch_size
		self.alpha = config.alpha
		self.beta = config.beta
		self.gamma = config.gamma
		self.delta = config.delta
		self.view_id = config.view_id

		self.epochs = config.epochs
		self.sample_interval = config.sample_interval

		self.compute_all = config.compute_all
		self.just_valid = config.just_valid

		if self.batch_size < 0:
			self.batch_size = 100 if self.num_classes < 100 else 500

		W_FileName = 'O_Transform_'+str(self.datasets)+'.mat'
		try:
			self.W = sio.loadmat(W_FileName)['W']
		except Exception as e:
			W = torch.Tensor(self.output_shape, self.output_shape)
			W = torch.nn.init.orthogonal(W, gain=1)[:, 0: self.num_classes]
			self.W = self.to_data(W)
			sio.savemat(W_FileName, {'W': self.W})

	def to_var(self, x):
		"""Converts numpy to variable."""
		if torch.cuda.is_available():
			x = x.cuda()
		return Variable(x) #torch.autograd.Variable

	def to_data(self, x):
		"""Converts variable to numpy."""
		try:
			if torch.cuda.is_available():
				x = x.cpu()
			return x.data.numpy()
		except Exception as e:
			return x

	def reset_grad(self):
		"""Zeros the gradient buffers."""
		self.g_optimizer.zero_grad()
		self.d_optimizer.zero_grad()
		# self.i_optimizer.zero_grad()

	def to_one_hot(self, x):
		if len(x.shape) == 1 or x.shape[1] == 1:
			one_hot = (self.classes.reshape([1, -1]) == x.reshape([-1, 1])).astype('float32')
			labels = one_hot
			y = self.to_var(torch.tensor(labels))
		else:
			y = self.to_var(torch.tensor(x.astype('float32')))
		return y

	def view_result(self, _acc):
		res = ''
		if type(_acc) is not list:
			res += ((' - mean: %.5f' % (np.sum(_acc) / (self.n_view * (self.n_view - 1)))) + ' - detail:')
			for _i in range(self.n_view):
				for _j in range(self.n_view):
					if _i != _j:
						res += ('%.5f' % _acc[_i, _j]) + ','
		else:
			R = [50, 'ALL']
			for _k in range(len(_acc)):
				res += (' R = ' + str(R[_k]) + ': ')
				res += ((' - mean: %.5f' % (np.sum(_acc[_k]) / (self.n_view * (self.n_view - 1)))) + ' - detail:')
				for _i in range(self.n_view):
					for _j in range(self.n_view):
						if _i != _j:
							res += ('%.5f' % _acc[_k][_i, _j]) + ','
		return res

	def shuffleTrain(self):
		for v in range(self.n_view):
			for ci in range(self.num_classes):
				inx = np.arange(len(self.train_view_list[1][v][ci]))
				np.random.shuffle(inx)
				self.train_view_list[0][v][ci] = self.train_view_list[0][v][ci][inx]
				self.train_view_list[1][v][ci] = self.train_view_list[1][v][ci][inx]
			self.train_data[v] = np.concatenate(self.train_view_list[0][v])
			self.train_labels[v] = np.concatenate(self.train_view_list[1][v])

	def train(self):
		self.train_self_supervised()

		if not self.just_valid:
			valid_fea, valid_lab, test_fea, test_lab = [], [], [], []
			for v in range(self.n_view):
				tmp = sio.loadmat('features/' + self.datasets + '_' + str(v) + '.mat')
				valid_fea.append(tmp['valid_fea'])
				valid_lab.append(tmp['valid_lab'].reshape([-1,]) if min(tmp['valid_lab'].shape) == 1 else tmp['valid_lab'])
				test_fea.append(tmp['test_fea'])
				test_lab.append(tmp['test_lab'].reshape([-1,]) if min(tmp['test_lab'].shape) == 1 else tmp['test_lab'])

			valid_results = utils.multi_test(valid_fea, valid_lab, self.MAP)
			test_results = utils.multi_test(test_fea, test_lab, self.MAP)
			print("valid results: " + self.view_result(valid_results) + ",\t test resutls:" + self.view_result(test_results))
			sio.savemat('features/' + self.datasets + '_SDML_test_feature_results.mat', {'test': test_fea, 'test_labels': test_lab})
			return valid_results, test_results
		else:
			return np.concatenate([np.array(loss).reshape([1, -1]) for loss in self.val_d_loss], axis=0), np.concatenate([np.array(loss).reshape([1, -1]) for loss in self.tr_d_loss], axis=0), np.concatenate([np.array(loss).reshape([1, -1]) for loss in self.tr_ae_loss], axis=0)

	def sample_anchor_points(self, features_list):

		anchor_samples = []

		import random

		while(len(anchor_samples) != self.num_anchors):
			view_id = random.randint(0, self.n_view-1)
			sample_id = random.randint(0, features_list[0].shape[0]-1)

			if([view_id, sample_id] not in anchor_samples):
				anchor_samples.append([view_id, sample_id])

		return anchor_samples

	def sample_positives(self, anchors, features_list):

		positives = []

		import random

		for index in range(len(anchors)):
			sampling_list = [i for i in range(self.n_view)]
			sampling_list.remove(anchors[index][0])
			positives.append([random.choice(sampling_list), anchors[index][1]])

		return positives

	def create_negative_samples(self, anchor, features_list):
		anchor_view_id = anchor[0]
		anchor_sample_id = anchor[1]

		neg_samples_list = []

		for view in range(self.n_view):
			for sample in range(features_list[0].shape[0]):
				if(sample != anchor_sample_id):
					neg_samples_list.append([view, sample])

		return neg_samples_list 

	def sample_negatives(self, anchors, features_list):

		negatives = []

		import random

		for index in range(len(anchors)):
			sampling_list=self.create_negative_samples(anchors[index], features_list)
			negative_samples = random.sample(sampling_list, self.num_negative_samples)
			negatives.append(negative_samples)

		return negatives

	def train_self_supervised(self):
		seed = 0
		import numpy as np
		np.random.seed(seed)
		import random as rn
		rn.seed(seed)
		import os
		os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
		import torch
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

		from networks import Dense_Net, Dense_Net_with_softmax
		Nets = []
		AEs = []
		
		for view_id in range(self.n_view):
			Net = Dense_Net(input_dim=self.input_shape[view_id], out_dim=self.output_shape)
			Nets.append(Net)
			AE = Dense_Net(input_dim=self.output_shape, out_dim=self.input_shape[view_id])
			AEs.append(AE)

		if torch.cuda.is_available():
			for view_id in range(self.n_view):
				Nets[view_id].cuda()
				AEs[view_id].cuda()

		W = torch.tensor(self.W)
		W = Variable(W.cuda(), requires_grad=False)
		get_grad_params = lambda model: [x for x in model.parameters() if x.requires_grad]
		
		params = []
		optims = []
		for view_id in range(self.n_view):
			params.append(get_grad_params(Nets[view_id]) + get_grad_params(AEs[view_id]))
			optims.append(optim.Adam(params[view_id], self.lr[view_id], [self.beta1, self.beta2]))

		discriminator_losses, losses, valid_results = [], [], []

		for view_id in range(self.n_view):
			discriminator_losses.append([])
			losses.append([])
			valid_results.append([])

		criterion = lambda x, y: (((x - y) ** 2).sum(1).sqrt()).mean()
		tr_d_loss, tr_ae_loss, val_d_loss, val_ae_loss = [], [], [], []
		for view_id in range(self.n_view):		
			tr_d_loss.append([])
			tr_ae_loss.append([])
			val_d_loss.append([])
			val_ae_loss.append([])

		valid_loss_min = 1e9
		for epoch in range(self.epochs):
			rand_idx = np.arange(self.train_data[view_id].shape[0])
			np.random.shuffle(rand_idx)
			batch_count = int(self.train_data[view_id].shape[0] / float(self.batch_size))

			k = 0
			mean_loss = []
			mean_tr_d_loss, mean_tr_ae_loss = [], []

			for view_id in range(self.n_view):
				mean_loss.append([])
				mean_tr_d_loss.append([])
				mean_tr_ae_loss.append([])

			for batch_idx in range(batch_count):

				self_supervised_features = []

				ae_loss = 0

				for view_id in range(self.n_view):
					print(('ViewID: %d, Epoch %d/%d') % (view_id, epoch + 1, self.epochs))

					idx = rand_idx[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
					train_y = self.to_one_hot(self.train_labels[view_id][idx])
					train_x = self.to_var(torch.tensor(self.train_data[view_id][idx]))
					
					optimizer = optims[view_id]
					Net = Nets[view_id]
					AE = AEs[view_id]

					optimizer.zero_grad()

					network_outputs = Net(train_x)
					ae_input = network_outputs[-1]
					pred = ae_input.view([ae_input.shape[0], -1]).mm(W)
					self_supervised_features.append(pred)

					ae_data = train_x
					ae_pred = AE(ae_input)[-1]
					ae_loss += criterion(ae_pred, ae_data)
				
				labeled_inx = train_y.sum(1) > 0

				d_loss = 0

				if(not self.use_nce):
					identity_mat=torch.eye(self_supervised_features[0].shape[0]).reshape(1, self_supervised_features[0].shape[0], self_supervised_features[0].shape[0]).repeat(self_supervised_features[0].shape[1], 1, 1)

					if torch.cuda.is_available():
						identity_mat = identity_mat.cuda()

					f_1 = self_supervised_features[0].view(self_supervised_features[0].shape[1], self_supervised_features[0].shape[0], -1)
					f_2 = self_supervised_features[1].view(self_supervised_features[1].shape[1], -1, self_supervised_features[0].shape[0])

					loss_pred=torch.bmm(f_1, f_2)			
					cpc_loss = self.cpc_loss_func(loss_pred, identity_mat)

				else:
					# NCE CPC loss implementation (Eq. 6) [parts of code referred from https://github.com/HobbitLong/CMC/blob/master/NCE/NCECriterion.py]
					eps = 1e-7

					anchor_samples_indices = self.sample_anchor_points(self_supervised_features)
					pos_samples_indices = self.sample_positives(anchor_samples_indices, self_supervised_features)
					neg_samples_indices = self.sample_negatives(anchor_samples_indices, self_supervised_features)

					anchor_samples = torch.zeros((self.num_anchors, self_supervised_features[0].shape[1]))
					pos_samples = torch.zeros((self.num_anchors, self_supervised_features[0].shape[1]))
					neg_samples = torch.zeros((self.num_anchors, self.num_negative_samples, self_supervised_features[0].shape[1]))

					for anchor_id in range(len(anchor_samples_indices)):
						anchor_point = anchor_samples_indices[anchor_id]
						anchor_samples[anchor_id] = self_supervised_features[anchor_point[0]][anchor_point[1]]

					for pos_id in range(len(pos_samples_indices)):
						pos_point = pos_samples_indices[pos_id]
						pos_samples[pos_id] = self_supervised_features[pos_point[0]][pos_point[1]]

					for neg_id in range(len(neg_samples_indices)):
						for neg_sample_id in range(len(neg_samples_indices[neg_id])):
							neg_point = neg_samples_indices[neg_id][neg_sample_id]
							neg_samples[neg_id][neg_sample_id] = self_supervised_features[neg_point[0]][neg_point[1]]

					if(torch.cuda.is_available()):
						anchor_samples = anchor_samples.cuda()
						pos_samples = pos_samples.cuda()
						neg_samples = neg_samples.cuda()

					# noise distribution
					Pn = 1 / float(self_supervised_features[0].shape[0])

					# number of noise samples
					m = self_supervised_features[0].shape[1]

					D1 = torch.div(pos_samples, torch.zeros_like(pos_samples).fill_(eps) + pos_samples.add(m*Pn + eps))
					D1 = torch.clamp(D1, min=eps)
					log_D1 = D1.log()
					
					D0 = torch.div(neg_samples.clone().fill_(m*Pn), torch.zeros_like(neg_samples).fill_(eps) + neg_samples.add(m*Pn + eps))
					D0 = torch.clamp(D0, min=eps)
					log_D0 = D0.log()

					cpc_loss = torch.mean(-(log_D1.sum(0) + log_D0.view(-1, 1).sum(0))/self_supervised_features[0].shape[0]**2)

				for f_ind in range(len(self_supervised_features)):
					train_y = self.to_one_hot(self.train_labels[f_ind][idx])
					feature = self_supervised_features[f_ind]
					
					curr_loss = criterion(feature[labeled_inx], train_y[labeled_inx])
					d_loss += curr_loss

				# cross modal loss (Eq. 2)
				cross_loss = 0

				for f_ind1 in range(len(self_supervised_features)):
					for f_ind2 in range(len(self_supervised_features)):
						if(f_ind1 == f_ind2):
							continue
						else:
							feature_1 = self_supervised_features[f_ind1]
							feature_2 = self_supervised_features[f_ind2]
							train_y1 = self.to_one_hot(self.train_labels[f_ind1][idx])
							train_y2 = self.to_one_hot(self.train_labels[f_ind2][idx])

							curr_loss = criterion(feature_1[labeled_inx], feature_2[labeled_inx])
							cross_loss += curr_loss 

				ae_loss *= self.alpha
				cross_loss *= self.beta
				cpc_loss *= self.gamma
				d_loss *= self.delta

				loss = ae_loss + cpc_loss + d_loss + cross_loss
				loss.backward()

				for view_id in range(self.n_view):
					optims[view_id].step()
					mean_loss[view_id].append(self.to_data(loss))
					mean_tr_d_loss[view_id].append(self.to_data(d_loss))
					mean_tr_ae_loss[view_id].append(self.to_data(ae_loss))

				if ((epoch + 1) % self.sample_interval == 0) and (batch_idx == batch_count - 1):
					valid_pres = []
					test_pres = []

					for view_id in range(self.n_view):
						losses[view_id].append(np.mean(mean_loss[view_id]))
						utils.show_progressbar([batch_idx, batch_count], mean_loss=np.mean(mean_loss[view_id]))

						pre_labels = utils.predict(lambda x: Nets[view_id](x)[-1].view([x.shape[0], -1]).mm(W).view([x.shape[0], -1]), self.valid_data[view_id], self.batch_size).reshape([self.valid_data[view_id].shape[0], -1])
						valid_labels = self.to_one_hot(self.valid_labels[view_id])
						valid_d_loss = self.to_data(criterion(self.to_var(torch.tensor(pre_labels)), valid_labels))
						if valid_loss_min > valid_d_loss and not self.just_valid:
							valid_loss_min = valid_d_loss
							valid_pre = utils.predict(lambda x: Nets[view_id](x)[-1].view([x.shape[0], -1]), self.valid_data[view_id], self.batch_size).reshape([self.valid_data[view_id].shape[0], -1])
							test_pre = utils.predict(lambda x: Nets[view_id](x)[-1].view([x.shape[0], -1]), self.test_data[view_id], self.batch_size).reshape([self.test_data[view_id].shape[0], -1])
							valid_pres.append(valid_pre)
							test_pres.append(test_pre)
						elif self.just_valid:
							tr_d_loss[view_id].append(np.mean(mean_tr_d_loss[view_id]))
							val_d_loss[view_id].append(valid_d_loss[view_id])
							tr_ae_loss[view_id].append(np.mean(mean_tr_ae_loss[view_id]))
				elif batch_idx == batch_count - 1:
					utils.show_progressbar([batch_idx, batch_count], mean_loss=np.mean(mean_loss[view_id]))
					losses[view_id].append(np.mean(mean_loss[view_id]))
				else:
					utils.show_progressbar([batch_idx, batch_count], loss=loss)
				k += 1

		torch.save(Nets[0].state_dict(), './features/'+self.datasets+'_image_encoder_weights')
		torch.save(Nets[1].state_dict(), './features/'+self.datasets+'_text_encoder_weights')
		torch.save(AEs[0].state_dict(), './features/'+self.datasets+'_image_decoder_weights')
		torch.save(AEs[1].state_dict(), './features/'+self.datasets+'_text_decoder_weights')

		if not self.just_valid:
			for view_id in range(self.n_view):
				sio.savemat('features/' + self.datasets + '_' + str(view_id) + '.mat', {'valid_fea': valid_pre, 'valid_lab': self.valid_labels[view_id], 'test_fea': test_pre, 'test_lab': self.test_labels[view_id]})
			return [valid_pre, test_pre]
		else:
			self.tr_d_loss[view_id] = tr_d_loss[view_id]
			self.tr_ae_loss[view_id] = tr_ae_loss[view_id]
			self.val_d_loss[view_id] = val_d_loss[view_id]