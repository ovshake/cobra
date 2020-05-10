import scipy.io as sio
import h5py
import pickle
import numpy as np

# parts of code referred from https://github.com/penghu-cs/SDML/blob/master/data_loader.py

def load_deep_features(data_name):
	import numpy as np
	valid_data, req_rec, b_wv_matrix = True, True, True
	unlabels, zero_shot, doc2vec, split = False, False, False, False

	if data_name.find('_doc2vec') > -1:
		doc2vec = True
		req_rec, b_wv_matrix = False, False

	if data_name == 'wiki_doc2vec':
		path = './datasets/wiki_data/wiki_deep.h5py'
		valid_len = 231
		MAP = -1
	elif data_name == 'nus_tfidf':
		path = './datasets/NUS-WIDE/tfidf'
		req_rec, b_wv_matrix = False, False
		MAP = -1

		# load dict for img feats
		test_img_feats_dict = pickle.load(open(path+'/img_test_id_feats.pkl', 'rb'), encoding='latin1')
		train_img_feats_dict = pickle.load(open(path+'/img_train_id_feats.pkl', 'rb'), encoding='latin1')

		# load dict for text feats
		test_text_feats_dict = pickle.load(open(path+'/test_id_bow.pkl', 'rb'), encoding='latin1')
		train_text_feats_dict = pickle.load(open(path+'/train_id_bow.pkl', 'rb'), encoding='latin1')

		# load dict keys/ids
		test_ids = pickle.load(open(path+'/test_ids.pkl', 'rb'), encoding='latin1')
		train_ids = pickle.load(open(path+'/train_ids.pkl', 'rb'), encoding='latin1')

		# load dict for one-hot encoded labels
		test_labels_one_hot = pickle.load(open(path+'/test_id_label_map.pkl', 'rb'), encoding='latin1')
		train_labels_one_hot = pickle.load(open(path+'/train_id_label_map.pkl', 'rb'), encoding='latin1')

		# load dict for integer labels
		test_labels_single = pickle.load(open(path+'/test_id_label_single.pkl', 'rb'), encoding='latin1')
		train_labels_single = pickle.load(open(path+'/train_id_label_single.pkl', 'rb'), encoding='latin1')

		# make feature and label arrays
		train_img_feats = np.zeros((len(train_ids), train_img_feats_dict[train_ids[0]].shape[0]))
		test_img_feats = np.zeros((len(test_ids), test_img_feats_dict[test_ids[0]].shape[0]))

		train_text_feats = np.zeros((len(train_ids), train_text_feats_dict[train_ids[0]].shape[0]))
		test_text_feats = np.zeros((len(test_ids), test_text_feats_dict[test_ids[0]].shape[0]))

		train_imgs_labels = np.zeros(len(train_ids))
		test_imgs_labels = np.zeros(len(test_ids))

		train_texts_labels = np.zeros(len(train_ids))
		test_texts_labels = np.zeros(len(test_ids))

		for i, key in enumerate(train_ids):
			train_img_feats[i] = train_img_feats_dict[key]
			train_text_feats[i] = train_text_feats_dict[key]
			train_imgs_labels[i] = train_labels_single[key]
			train_texts_labels[i] = train_labels_single[key]

		for i, key in enumerate(test_ids):
			test_img_feats[i] = test_img_feats_dict[key]
			test_text_feats[i] = test_text_feats_dict[key]
			test_imgs_labels[i] = test_labels_single[key]
			test_texts_labels[i] = test_labels_single[key]

		train_img_feats = train_img_feats.astype('float32')
		test_img_feats = test_img_feats.astype('float32')
		train_text_feats = train_text_feats.astype('float32')
		test_text_feats = test_text_feats.astype('float32')

		train_imgs_labels = train_imgs_labels.astype('int64')
		test_imgs_labels = test_imgs_labels.astype('int64')
		train_texts_labels = train_texts_labels.astype('int64')
		test_texts_labels = test_texts_labels.astype('int64')

		train_data = [train_img_feats, train_text_feats]
		test_data = [test_img_feats[:1000], test_text_feats[:1000]]
		valid_data = [test_img_feats[1000::], test_text_feats[1000::]]

		train_labels = [train_imgs_labels, train_texts_labels]
		test_labels = [test_imgs_labels[:1000], test_texts_labels[:1000]]
		valid_labels = [test_imgs_labels[1000::], test_texts_labels[1000::]]

	# named it as Doc2Vec to bypass _doc2vec condition
	elif data_name == 'pascal_Doc2Vec':
		path = './datasets/pascal_sentence'
		req_rec, b_wv_matrix = False, False
		MAP = -1

		# load dict for img feats
		test_img_feats_dict = pickle.load(open(path+'/pascal_test_img_feats.pickle', 'rb'), encoding='latin1')
		train_img_feats_dict = pickle.load(open(path+'/pascal_train_img_feats.pickle', 'rb'), encoding='latin1')

		# load dict for text feats
		test_text_feats_dict = pickle.load(open(path+'/pascal_test_text_feats.pickle', 'rb'), encoding='latin1')
		train_text_feats_dict = pickle.load(open(path+'/pascal_train_text_feats.pickle', 'rb'), encoding='latin1')

		# load dict keys/ids
		test_ids = pickle.load(open(path+'/pascal_test_ids.pickle', 'rb'), encoding='latin1')
		train_ids = pickle.load(open(path+'/pascal_train_ids.pickle', 'rb'), encoding='latin1')

		# load dict for integer labels
		test_labels_single = pickle.load(open(path+'/pascal_test_img_labels.pickle', 'rb'), encoding='latin1')
		train_labels_single = pickle.load(open(path+'/pascal_train_img_labels.pickle', 'rb'), encoding='latin1')

		# make feature and label arrays
		train_img_feats = np.zeros((len(train_ids), train_img_feats_dict[train_ids[0]].shape[1]))
		test_img_feats = np.zeros((len(test_ids), test_img_feats_dict[test_ids[0]].shape[1]))

		train_text_feats = np.zeros((len(train_ids), train_text_feats_dict[train_ids[0]].shape[0]))
		test_text_feats = np.zeros((len(test_ids), test_text_feats_dict[test_ids[0]].shape[0]))

		train_imgs_labels = np.zeros(len(train_ids))
		test_imgs_labels = np.zeros(len(test_ids))

		train_texts_labels = np.zeros(len(train_ids))
		test_texts_labels = np.zeros(len(test_ids))

		for i, key in enumerate(train_ids):
			train_img_feats[i] = train_img_feats_dict[key].reshape((-1))
			train_text_feats[i] = train_text_feats_dict[key]
			train_imgs_labels[i] = train_labels_single[key]
			train_texts_labels[i] = train_labels_single[key]

		for i, key in enumerate(test_ids):
			test_img_feats[i] = test_img_feats_dict[key].reshape((-1))
			test_text_feats[i] = test_text_feats_dict[key]
			test_imgs_labels[i] = test_labels_single[key]
			test_texts_labels[i] = test_labels_single[key]

		train_img_feats = train_img_feats.astype('float32')
		test_img_feats = test_img_feats.astype('float32')
		train_text_feats = train_text_feats.astype('float32')
		test_text_feats = test_text_feats.astype('float32')

		train_imgs_labels = train_imgs_labels.astype('int64')
		test_imgs_labels = test_imgs_labels.astype('int64')
		train_texts_labels = train_texts_labels.astype('int64')
		test_texts_labels = test_texts_labels.astype('int64')

		train_data = [train_img_feats, train_text_feats]
		test_data = [test_img_feats[:100], test_text_feats[:100]]
		valid_data = [test_img_feats[100::], test_text_feats[100::]]

		train_labels = [train_imgs_labels, train_texts_labels]
		test_labels = [test_imgs_labels[:100], test_texts_labels[:100]]
		valid_labels = [test_imgs_labels[100::], test_texts_labels[100::]]

	elif data_name == 'mscoco':
		path = './datasets/mscoco'
		req_rec, b_wv_matrix = False, False
		MAP = -1

		# load dict for img feats
		test_img_feats_dict = pickle.load(open(path+'/test_coco_img_feats.pickle', 'rb'))
		train_img_feats_dict = pickle.load(open(path+'/train_coco_img_feats.pickle', 'rb'))

		# load dict for text feats
		test_text_feats_dict = pickle.load(open(path+'/test_coco_text_feats.pickle', 'rb'))
		train_text_feats_dict = pickle.load(open(path+'/train_coco_text_feats.pickle', 'rb'))

		# load img dict keys/ids
		test_img_ids = pickle.load(open(path+'/test_coco_img_access_keys.pickle', 'rb'))
		train_img_ids = pickle.load(open(path+'/train_coco_img_access_keys.pickle', 'rb'))

		# load text dict keys/ids
		test_text_ids = pickle.load(open(path+'/test_coco_text_access_keys.pickle', 'rb'))
		train_text_ids = pickle.load(open(path+'/train_coco_text_access_keys.pickle', 'rb'))

		# load dict for img integer labels
		test_img_labels_single = pickle.load(open(path+'/test_coco_img_labels.pickle', 'rb'))
		train_img_labels_single = pickle.load(open(path+'/train_coco_img_labels.pickle', 'rb'))

		# load dict for text integer labels
		test_text_labels_single = pickle.load(open(path+'/test_coco_text_labels.pickle', 'rb'))
		train_text_labels_single = pickle.load(open(path+'/train_coco_text_labels.pickle', 'rb'))

		# make feature and label arrays
		train_img_feats = np.zeros((len(train_img_ids), train_img_feats_dict[train_img_ids[0]].shape[0]))
		test_img_feats = np.zeros((len(test_img_ids), test_img_feats_dict[test_img_ids[0]].shape[0]))

		train_text_feats = np.zeros((len(train_img_ids), train_text_feats_dict[train_text_ids[0]].shape[0]))
		test_text_feats = np.zeros((len(test_img_ids), test_text_feats_dict[test_text_ids[0]].shape[0]))

		train_imgs_labels = np.zeros(len(train_img_ids))
		test_imgs_labels = np.zeros(len(test_img_ids))

		train_texts_labels = np.zeros(len(train_img_ids))
		test_texts_labels = np.zeros(len(test_img_ids))

		for i, key in enumerate(train_img_ids):
			train_img_feats[i] = train_img_feats_dict[key]
			train_imgs_labels[i] = train_img_labels_single[key]

		for i, key in enumerate(test_img_ids):
			test_img_feats[i] = test_img_feats_dict[key]
			test_imgs_labels[i] = test_img_labels_single[key]

		completed_test_text_ids = []
		completed_train_text_ids = []

		train_text_idx = 0
		test_text_idx = 0

		for i, key in enumerate(train_text_ids):
			pos = key.find('_')
			img_id = key[:pos]

			if(img_id in completed_train_text_ids):
				continue

			else:
				completed_train_text_ids.append(img_id)
				train_text_feats[train_text_idx] = train_text_feats_dict[key]
				train_texts_labels[train_text_idx] = train_text_labels_single[key]
				train_text_idx+=1

		for i, key in enumerate(test_text_ids):
			pos = key.find('_')
			img_id = key[:pos]

			if(img_id in completed_test_text_ids):
				continue

			else:
				completed_test_text_ids.append(img_id)
				test_text_feats[test_text_idx] = test_text_feats_dict[key]
				test_texts_labels[test_text_idx] = test_text_labels_single[key]
				test_text_idx+=1

		train_img_feats = train_img_feats.astype('float32')
		test_img_feats = test_img_feats.astype('float32')
		train_text_feats = train_text_feats.astype('float32')
		test_text_feats = test_text_feats.astype('float32')

		train_imgs_labels = train_imgs_labels.astype('int64')
		test_imgs_labels = test_imgs_labels.astype('int64')
		train_texts_labels = train_texts_labels.astype('int64')
		test_texts_labels = test_texts_labels.astype('int64')

		train_data = [train_img_feats, train_text_feats]
		test_data = [test_img_feats[:10000], test_text_feats[:10000]]
		valid_data = [test_img_feats[10000::], test_text_feats[10000::]]

		train_labels = [train_imgs_labels, train_texts_labels]
		test_labels = [test_imgs_labels[:10000], test_texts_labels[:10000]]
		valid_labels = [test_imgs_labels[10000::], test_texts_labels[10000::]]
		
	elif data_name == 'xmedia':
		path = './datasets/XMedia&Code/XMediaFeatures.mat'
		MAP = -1
		req_rec, b_wv_matrix = False, False
		all_data = sio.loadmat(path)
		A_te = all_data['A_te'].astype('float32')           # Features of test set for audio data, MFCC feature
		A_tr = all_data['A_tr'].astype('float32')           # Features of training set for audio data, MFCC feature
		d3_te = all_data['d3_te'].astype('float32')         # Features of test set for 3D data, LightField feature
		d3_tr = all_data['d3_tr'].astype('float32')         # Features of training set for 3D data, LightField feature
		I_te_CNN = all_data['I_te_CNN'].astype('float32')	# Features of test set for image data, CNN feature
		I_tr_CNN = all_data['I_tr_CNN'].astype('float32')	# Features of training set for image data, CNN feature
		T_te_BOW = all_data['T_te_BOW'].astype('float32')	# Features of test set for text data, BOW feature
		T_tr_BOW = all_data['T_tr_BOW'].astype('float32')	# Features of training set for text data, BOW feature
		V_te_CNN = all_data['V_te_CNN'].astype('float32')	# Features of test set for video(frame) data, CNN feature
		V_tr_CNN = all_data['V_tr_CNN'].astype('float32')	# Features of training set for video(frame) data, CNN feature
		te3dCat = all_data['te3dCat'].reshape([-1]).astype('int64')   # category label of test set for 3D data
		tr3dCat = all_data['tr3dCat'].reshape([-1]).astype('int64')   # category label of training set for 3D data
		teAudCat = all_data['teAudCat'].reshape([-1]).astype('int64') # category label of test set for audio data
		trAudCat = all_data['trAudCat'].reshape([-1]).astype('int64') # category label of training set for audio data
		teImgCat = all_data['teImgCat'].reshape([-1]).astype('int64') # category label of test set for image data
		trImgCat = all_data['trImgCat'].reshape([-1]).astype('int64') # category label of training set for image data
		teVidCat = all_data['teVidCat'].reshape([-1]).astype('int64') # category label of test set for video(frame) data
		trVidCat = all_data['trVidCat'].reshape([-1]).astype('int64') # category label of training set for video(frame) data
		teTxtCat = all_data['teTxtCat'].reshape([-1]).astype('int64') # category label of test set for text data
		trTxtCat = all_data['trTxtCat'].reshape([-1]).astype('int64') # category label of training set for text data

		train_data = [I_tr_CNN, T_tr_BOW, A_tr, d3_tr, V_tr_CNN]
		test_data = [I_te_CNN[0: 500], T_te_BOW[0: 500], A_te[0: 100], d3_te[0: 50], V_te_CNN[0: 87]]
		valid_data = [I_te_CNN[500::], T_te_BOW[500::], A_te[100::], d3_te[50::], V_te_CNN[87::]]
		train_labels = [trImgCat, trTxtCat, trAudCat, tr3dCat, trVidCat]
		test_labels = [teImgCat[0: 500], teTxtCat[0: 500], teAudCat[0: 100], te3dCat[0: 50], teVidCat[0: 87]]
		valid_labels = [teImgCat[500::], teTxtCat[500::], teAudCat[100::], te3dCat[50::], teVidCat[87::]]

	elif data_name == 'gossip':
		path = './datasets/gossip/'
		MAP = -1
		req_rec, b_wv_matrix = False, False

		train_img_feats = pickle.load(open(path+'final_train_image_features.pickle', 'rb'))
		test_img_feats = pickle.load(open(path+'final_test_image_features.pickle', 'rb'))

		train_text_feats = pickle.load(open(path+'final_train_text_features.pickle', 'rb'))
		test_text_feats = pickle.load(open(path+'final_test_text_features.pickle', 'rb'))

		train_labels = pickle.load(open(path+'final_train_labels.pickle', 'rb'))
		test_labels = pickle.load(open(path+'final_test_labels.pickle', 'rb'))

		train_img_feats = train_img_feats.astype('float32')
		test_img_feats = test_img_feats.astype('float32')
		train_text_feats = train_text_feats.astype('float32')
		test_text_feats = test_text_feats.astype('float32')

		train_labels = train_labels.astype('int64')
		test_labels = test_labels.astype('int64')

		print(test_text_feats.shape)

		train_data = [train_img_feats, train_text_feats]
		test_data = [test_img_feats[:1000], test_text_feats[:1000]]
		valid_data = [test_img_feats[1000:], test_text_feats[1000:]]

		print(test_data[0].shape, test_data[1].shape)
		print(valid_data[0].shape, valid_data[1].shape)

		test_l = test_labels[:1000]
		val_l = test_labels[1000:]

		train_labels = [train_labels, train_labels]
		test_labels = [test_l, test_l]
		valid_labels = [val_l, val_l]

		print(train_labels[0].shape, train_labels[1].shape)
		print(test_labels[0].shape, test_labels[1].shape)
		print(valid_labels[0].shape, valid_labels[1].shape)

	elif data_name == 'politifact':
		path = './datasets/politifact/'
		MAP = -1
		req_rec, b_wv_matrix = False, False

		train_img_feats = pickle.load(open(path+'final_train_image_features.pickle', 'rb'))
		test_img_feats = pickle.load(open(path+'final_test_image_features.pickle', 'rb'))

		train_text_feats = pickle.load(open(path+'final_train_text_features.pickle', 'rb'))
		test_text_feats = pickle.load(open(path+'final_test_text_features.pickle', 'rb'))

		train_labels = pickle.load(open(path+'final_train_labels.pickle', 'rb'))
		test_labels = pickle.load(open(path+'final_test_labels.pickle', 'rb'))

		train_img_feats = train_img_feats.astype('float32')
		test_img_feats = test_img_feats.astype('float32')
		train_text_feats = train_text_feats.astype('float32')
		test_text_feats = test_text_feats.astype('float32')

		train_labels = train_labels.astype('int64')
		test_labels = test_labels.astype('int64')

		train_data = [train_img_feats, train_text_feats]
		test_data = [test_img_feats[:50], test_text_feats[:50]]
		valid_data = [test_img_feats[50:], test_text_feats[50:]]

		test_l = test_labels[:50]
		val_l = test_labels[50:]

		train_labels = [train_labels, train_labels]
		test_labels = [test_l, test_l]
		valid_labels = [val_l, val_l]

	elif data_name == 'metoo':
		path = './datasets/metoo/'
		MAP = -1
		req_rec, b_wv_matrix = False, False

		train_img_feats_dict = pickle.load(open(path+'metoo_train_img_feats.pickle', 'rb'))
		test_img_feats_dict = pickle.load(open(path+'metoo_test_img_feats.pickle', 'rb'))

		train_text_feats_dict = pickle.load(open(path+'metoo_train_text_feats.pickle', 'rb'))
		test_text_feats_dict = pickle.load(open(path+'metoo_test_text_feats.pickle', 'rb'))

		train_keys = pickle.load(open(path+'metoo_train_keys.pickle', 'rb'))
		test_keys = pickle.load(open(path+'metoo_test_keys.pickle', 'rb'))

		train_labels_dict = pickle.load(open(path+'metoo_train_labels_dictionary.pickle', 'rb'))
		test_labels_dict = pickle.load(open(path+'metoo_test_labels_dictionary.pickle', 'rb'))

		train_img_feats = np.zeros((len(train_keys), train_img_feats_dict[train_keys[0]].shape[0]))
		test_img_feats = np.zeros((len(test_keys), test_img_feats_dict[test_keys[0]].shape[0]))

		train_text_feats = np.zeros((len(train_keys), train_text_feats_dict[train_keys[0]].shape[0]))
		test_text_feats = np.zeros((len(test_keys), test_text_feats_dict[test_keys[0]].shape[0]))

		train_labels = np.zeros(len(train_keys))
		test_labels = np.zeros(len(test_keys))

		# diff label indices are: text_only_inform, image_only_inform, dir_hate, gen_hate, sarcasm, allegation, justification, refutation, support, oppose

		for i, key in enumerate(train_keys):
			train_img_feats[i] = train_img_feats_dict[key]
			train_text_feats[i] = train_text_feats_dict[key]
			train_labels[i] = train_labels_dict['text_only_inform'][key]

		for i, key in enumerate(test_keys):
			test_img_feats[i] = test_img_feats_dict[key]
			test_text_feats[i] = test_text_feats_dict[key]
			test_labels[i] = test_labels_dict['text_only_inform'][key]

		train_img_feats = train_img_feats.astype('float32')
		test_img_feats = test_img_feats.astype('float32')
		train_text_feats = train_text_feats.astype('float32')
		test_text_feats = test_text_feats.astype('float32')

		train_labels = train_labels.astype('int64')
		test_labels = test_labels.astype('int64')

		train_data = [train_img_feats, train_text_feats]
		test_data = [test_img_feats[:1000], test_text_feats[:1000]]
		valid_data = [test_img_feats[1000:], test_text_feats[1000:]]

		test_l = test_labels[:1000]
		val_l = test_labels[1000:]

		train_labels = [train_labels, train_labels]
		test_labels = [test_l, test_l]
		valid_labels = [val_l, val_l]

		print(train_data[0].shape, train_data[1].shape)
		print(valid_data[0].shape, valid_data[1].shape)
		print(test_data[0].shape, test_data[1].shape)

		print(train_labels[0].shape, train_labels[1].shape)
		print(valid_labels[0].shape, valid_labels[1].shape)
		print(test_labels[0].shape, test_labels[1].shape)

	elif data_name == 'crisis':
		path = './datasets/CrisisMMD/'
		MAP = -1
		req_rec, b_wv_matrix = False, False

		train_img_feats_dict = pickle.load(open(path+'crisis_train_img_feats.pickle', 'rb'))
		test_img_feats_dict = pickle.load(open(path+'crisis_test_img_feats.pickle', 'rb'))

		train_text_feats_dict = pickle.load(open(path+'crisis_train_text_feats.pickle', 'rb'))
		test_text_feats_dict = pickle.load(open(path+'crisis_test_text_feats.pickle', 'rb'))

		train_keys = pickle.load(open(path+'crisis_train_keys.pickle', 'rb'))
		test_keys = pickle.load(open(path+'crisis_test_keys.pickle', 'rb'))

		train_labels_dict = pickle.load(open(path+'crisis_train_labels_dict.pickle', 'rb'))
		test_labels_dict = pickle.load(open(path+'crisis_test_labels_dict.pickle', 'rb'))

		train_img_feats = np.zeros((len(train_keys), train_img_feats_dict[train_keys[0]].shape[0]))
		test_img_feats = np.zeros((len(test_keys), test_img_feats_dict[test_keys[0]].shape[0]))

		train_text_feats = np.zeros((len(train_keys), train_text_feats_dict[train_keys[0]].shape[0]))
		test_text_feats = np.zeros((len(test_keys), test_text_feats_dict[test_keys[0]].shape[0]))

		train_labels = np.zeros(len(train_keys))
		test_labels = np.zeros(len(test_keys))

		# diff label indices are: text_inform, image_inform, text_human, image_human

		for i, key in enumerate(train_keys):
			train_img_feats[i] = train_img_feats_dict[key]
			train_text_feats[i] = train_text_feats_dict[key]
			train_labels[i] = train_labels_dict['text_human'][key]

		for i, key in enumerate(test_keys):
			test_img_feats[i] = test_img_feats_dict[key]
			test_text_feats[i] = test_text_feats_dict[key]
			test_labels[i] = test_labels_dict['text_human'][key]

		train_img_feats = train_img_feats.astype('float32')
		test_img_feats = test_img_feats.astype('float32')
		train_text_feats = train_text_feats.astype('float32')
		test_text_feats = test_text_feats.astype('float32')

		train_labels = train_labels.astype('int64')
		test_labels = test_labels.astype('int64')

		train_data = [train_img_feats, train_text_feats]
		test_data = [test_img_feats[:500], test_text_feats[:500]]
		valid_data = [test_img_feats[500:], test_text_feats[500:]]

		test_l = test_labels[:500]
		val_l = test_labels[500:]

		train_labels = [train_labels, train_labels]
		test_labels = [test_l, test_l]
		valid_labels = [val_l, val_l]

		print(train_data[0].shape, train_data[1].shape)
		print(valid_data[0].shape, valid_data[1].shape)
		print(test_data[0].shape, test_data[1].shape)

		print(train_labels[0].shape, train_labels[1].shape)
		print(valid_labels[0].shape, valid_labels[1].shape)
		print(test_labels[0].shape, test_labels[1].shape)

	elif data_name == 'crisis_damage':
		path = './datasets/CrisisMMD/'
		MAP = -1
		req_rec, b_wv_matrix = False, False

		train_img_feats_dict = pickle.load(open(path+'crisis_train_img_damage_feats.pickle', 'rb'))
		test_img_feats_dict = pickle.load(open(path+'crisis_test_img_damage_feats.pickle', 'rb'))

		train_text_feats_dict = pickle.load(open(path+'crisis_train_text_damage_feats.pickle', 'rb'))
		test_text_feats_dict = pickle.load(open(path+'crisis_test_text_damage_feats.pickle', 'rb'))

		train_keys = pickle.load(open(path+'crisis_train_image_damage_keys.pickle', 'rb'))
		test_keys = pickle.load(open(path+'crisis_test_image_damage_keys.pickle', 'rb'))

		train_labels_dict = pickle.load(open(path+'crisis_train_labels_dict.pickle', 'rb'))
		test_labels_dict = pickle.load(open(path+'crisis_test_labels_dict.pickle', 'rb'))

		train_img_feats = np.zeros((len(train_keys), train_img_feats_dict[train_keys[0]].shape[0]))
		test_img_feats = np.zeros((len(test_keys), test_img_feats_dict[test_keys[0]].shape[0]))

		train_text_feats = np.zeros((len(train_keys), train_text_feats_dict[train_keys[0]].shape[0]))
		test_text_feats = np.zeros((len(test_keys), test_text_feats_dict[test_keys[0]].shape[0]))

		train_labels = np.zeros(len(train_keys))
		test_labels = np.zeros(len(test_keys))

		for i, key in enumerate(train_keys):
			train_img_feats[i] = train_img_feats_dict[key]
			train_text_feats[i] = train_text_feats_dict[key]
			train_labels[i] = train_labels_dict['image_damage'][key]

		for i, key in enumerate(test_keys):
			test_img_feats[i] = test_img_feats_dict[key]
			test_text_feats[i] = test_text_feats_dict[key]
			test_labels[i] = test_labels_dict['image_damage'][key]

		train_img_feats = train_img_feats.astype('float32')
		test_img_feats = test_img_feats.astype('float32')
		train_text_feats = train_text_feats.astype('float32')
		test_text_feats = test_text_feats.astype('float32')

		train_labels = train_labels.astype('int64')
		test_labels = test_labels.astype('int64')

		train_data = [train_img_feats, train_text_feats]
		test_data = [test_img_feats[:500], test_text_feats[:500]]
		valid_data = [test_img_feats[500:], test_text_feats[500:]]

		test_l = test_labels[:500]
		val_l = test_labels[500:]

		train_labels = [train_labels, train_labels]
		test_labels = [test_l, test_l]
		valid_labels = [val_l, val_l]

		print(train_data[0].shape, train_data[1].shape)
		print(valid_data[0].shape, valid_data[1].shape)
		print(test_data[0].shape, test_data[1].shape)

		print(train_labels[0].shape, train_labels[1].shape)
		print(valid_labels[0].shape, valid_labels[1].shape)
		print(test_labels[0].shape, test_labels[1].shape)
			

	if doc2vec:
		print(path)	
		h = h5py.File(path)
		train_imgs_deep = h['train_imgs_deep'][()].astype('float32')
		train_imgs_labels = h['train_imgs_labels'][()]
		train_imgs_labels -= np.min(train_imgs_labels)
		train_texts_idx = h['train_text'][()].astype('float32')
		train_texts_labels = h['train_texts_labels'][()]
		train_texts_labels -= np.min(train_texts_labels)
		train_data = [train_imgs_deep, train_texts_idx]
		train_labels = [train_imgs_labels, train_texts_labels]

		test_imgs_deep = h['test_imgs_deep'][()].astype('float32')
		test_imgs_labels = h['test_imgs_labels'][()]
		test_imgs_labels -= np.min(test_imgs_labels)
		test_texts_idx = h['test_text'][()].astype('float32')
		test_texts_labels = h['test_texts_labels'][()]
		test_texts_labels -= np.min(test_texts_labels)
		test_data = [test_imgs_deep, test_texts_idx]
		test_labels = [test_imgs_labels, test_texts_labels]

		valid_data = [test_data[0][0: valid_len], test_data[1][0: valid_len]]
		valid_labels = [test_labels[0][0: valid_len], test_labels[1][0: valid_len]]

		test_data = [test_data[0][valid_len::], test_data[1][valid_len::]]
		test_labels = [test_labels[0][valid_len::], test_labels[1][valid_len::]]

	if valid_data:
		if b_wv_matrix:
			return train_data, train_labels, valid_data, valid_labels, test_data, test_labels, wv_matrix, MAP
		else:
			return train_data, train_labels, valid_data, valid_labels, test_data, test_labels, MAP
	else:
		if b_wv_matrix:
			return train_data, train_labels, test_data, test_labels, wv_matrix, MAP
		else:
			return train_data, train_labels, test_data, test_labels, MAP

if __name__ == '__main__':
	load_deep_features('wiki_doc2vec')
	load_deep_features('nus_tfidf')
	load_deep_features('mscoco')
	load_deep_features('xmedia')
	load_deep_features('gossip')
	load_deep_features('politifact')
	load_deep_features('metoo')
	load_deep_features('crisis')
	load_deep_features('crisis_damage')