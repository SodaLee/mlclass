import os
import random
import numpy as np
import pickle

simi = dict()


def train_init(tr_t, tr_l, te_t, te_l, types):
	weibo = "new_weibo_13638"
	dir_name = os.listdir(weibo)

	for d in dir_name:
		t_name = os.listdir(weibo+"/"+d)
		types.append(d)
		random.shuffle(t_name)
		pos = int(np.floor(len(t_name) * 0.9))
		
		for txt in t_name[:pos]:
			f = open(weibo+"/"+d+"/"+txt, "r")
			t = f.read().split()
			if (len(tr_l) > 0) and (d == tr_l[-1]):
				tr_t[-1].append(t)
			else:
				tr_l.append(d)
				tr_t.append(list(t))

		for txt in t_name[pos:]:
			f = open(weibo+"/"+d+"/"+txt, "r")
			t = f.read().split()
			te_t.append(t)
			te_l.append(d)


def extract_feature(tr_t, features):
	global simi
	txt_words = [[set(txt) for txt in topic] for topic in tr_t]
	topic_words = []
	for topic in txt_words:
		es = set()
		for txt in topic:
			es = es | txt
		topic_words.append(es)
	tot_words = set()
	for topic in topic_words:
		tot_words = tot_words | topic
	
	print("Computing TF-IDF...")
	TF_IDF_in_topic = []
	for i in range(len(topic_words)):
		TF_IDF = []
		for word in topic_words[i]:
			F, TTW = 0, 0
			for doc in tr_t[i]:
				F += doc.count(word)
				TTW += len(doc)
			IDF = 0
			for topic in topic_words:
				if word in topic:
					IDF += 1
			TF_IDF.append([word, F/TTW, np.log2(9/IDF)])
		TF_IDF_in_topic.append(TF_IDF)
	TF_IDF_SAVE = open("TF_IDF.pkl", "wb")
	pickle.dump(TF_IDF_in_topic, TF_IDF_SAVE)
	print("Finished")

	# print("Loading TF-IDF...")
	# TF_IDF_LOAD = open("TF_IDF.pkl", "rb")
	# TF_IDF_in_topic = pickle.load(TF_IDF_LOAD)
	# print("Finished")

	TOP10 = [sorted(topic, key=lambda e: e[1]*e[2], reverse=True)[0:2] for topic in TF_IDF_in_topic]
	for topic in TOP10:
		for e in topic:
			features.append(e[0])

	print("Computing similarity...")
	docs = 0
	for topic in txt_words:
		docs += len(topic)
	cnt = 0
	for i in tot_words:
		cnt += 1
		if (cnt % 1000 == 0):
			print(cnt/57436)
		simi[i] = dict()
		for j in features:
			pi, pj, pij = 0, 0, 0
			for topic in txt_words:
				for doc in topic:
					if i in doc:
						pi += 1
					if j in doc:
						pj += 1
						if i in doc:
							pij += 1
			for topic in topic_words:
				if (i in topic) and (j in topic):
					pij += 1
			if (pij == 0):
				simi[i][j] = -1
			else:
				simi[i][j] = np.log2(pij * docs / (pi * pj))
	simi_save = open("simi.pkl", "wb")
	pickle.dump(simi, simi_save)
	print("Done")

	# print("Loading similarity...")
	# simi_load = open("simi.pkl", "rb")
	# simi = pickle.load(simi_load)
	# print("Finished")


def count_entropy(l, ans):
	label = list(l)
	N = len(label)
	if N == 0:
		return 0
	s1 = label.count(ans)
	s2 = N - s1
	if (s1 == 0) or (s2 == 0):
		return 0
	return s1/N*np.log2(N/s1)+s2/N*np.log2(N/s2)


def bulid_tree(txt, label, tree_name):
	N = len(txt)
	if count_entropy(label, tree_name) == 0:
		l = list(label)
		if (l.count(tree_name) > 0):
			return "Yes"
		else:
			return "No"
	choice = [0, 0, 100]
	for i in range(9*2):
		n_min = txt.min(0)[i]
		n_max = txt.max(0)[i]
		step = (n_max - n_min) / 10
		if step < 1e-9:
			continue
		for s in np.arange(n_min + step, n_max, step):
			left = label[txt[:, i] < s]
			right = label[txt[:, i] >= s]
			N1, N2 = len(left), len(right)
			entropy = N1/N*count_entropy(left, tree_name) + N2/N*count_entropy(right, tree_name)
			if entropy < choice[2]:
				choice = [i, s, entropy]
	l_txt = txt[txt[:, choice[0]] < choice[1]]
	l_label = label[txt[:, choice[0]] < choice[1]]
	r_txt = txt[txt[:, choice[0]] >= choice[1]]
	r_label = label[txt[:, choice[0]] >= choice[1]]
	if (len(l_label) == 0):
		l = list(r_label)
		if (l.count(tree_name) * 2 > len(l)):
			return "Yes"
		else:
			return "No"
	if (len(r_label) == 0):
		l = list(l_label)
		if (l.count(tree_name) * 2 > len(l)):
			return "Yes"
		else:
			return "No"
	return [choice[0:2], bulid_tree(l_txt, l_label, tree_name), bulid_tree(r_txt, r_label, tree_name)]


def train_classifier(tr_t, tr_l, fea, cla):
	ss, l = [], []
	for i in range(len(tr_t)):
		for doc in tr_t[i]:
			l.append(tr_l[i])
			s = np.zeros(len(fea))
			for word in doc:
				for j in range(len(fea)):
					s[j] += simi[word][fea[j]]
			s /= len(doc)
			ss.append(s)
	txt_f = np.array(ss)
	txt_l = np.array(l)
	
	print("Building...")
	for label in tr_l:
		print("Working on",label)
		cla.append(bulid_tree(txt_f, txt_l, label))


def tree_test(s, cla):
	p = cla[0]
	if (p == "Y"):
		return 1
	if (p == "N"):
		return 0
	if (s[p[0]] < p[1]):
		return tree_test(s, cla[1])
	else:
		return tree_test(s, cla[2])


def test(txt, label, fea, cla, types):
	ans = []
	for doc in txt:
		del_word = 0
		s = np.zeros(len(fea))
		for word in doc:
			if word not in simi:
				del_word += 1
				continue
			for j in range(len(fea)):
				s[j] += simi[word][fea[j]]
		s /= len(doc) - del_word
		l = [tree_test(s, classifier) for classifier in cla]
		if (l.count(1) == 1):
			ans.append(types[l.index(1)])
		else:
			rea = []
			while l.count(1):
				rea.append(types[l.index(1)])
				l[l.index(1)] = 0
			random.shuffle(rea)
			if len(rea) == 0:
				ans.append(types[np.random.randint(9)])
			else:
				ans.append(rea[0])

	cnt = 0
	for i in range(len(label)):
		if ans[i] == label[i]:
			cnt += 1
	print("correctness is", cnt / len(label))
	return cnt / len(label)


def train():
	global simi
	simi.clear()
	train_text, train_label, test_text, test_label, types = [], [], [], [], []
	print("Initializing...")
	train_init(train_text, train_label, test_text, test_label, types)
	print("Extracting features")
	features = []
	extract_feature(train_text, features)
	print("Training classifiers...")
	classifier = []
	train_classifier(train_text, train_label, features, classifier)
	print("Start testing...")
	return test(test_text, test_label, features, classifier, types)


if __name__ == "__main__":
	train()