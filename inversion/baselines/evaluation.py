import sys
sys.path.append('~/P2I_MI/inversion/baselines')
from utils import *
from classify import *
from generator import *
from discri import *
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import grad
import torchvision.transforms as transforms
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
# import inception
# import fid
import lpips
import statistics

def save_knn(E, trainloader, dataset):
	os.makedirs('./feat', exist_ok=True)
	feat_path = './feat/feat_{}.npy'.format(dataset)
	info_path = "./feat/info_{}.npy".format(dataset)


	feat_list = []
	label_list = []
	with torch.no_grad():
		for i, (img, iden) in enumerate(trainloader):
			img = img.to(device)
			bs = img.size(0)
			feat = E(img)[-2]
			feat_list.append(feat)
			label_list.append(iden)
			# save_tensor_images(img[0].detach(), './test.png')
	img_feat = torch.cat(feat_list, dim=0)
	img_label = torch.cat(label_list, dim=0)
	# print(img_feat.shape)
	info = torch.LongTensor(img_label)
	# print(info)
	np.save(feat_path, img_feat.detach().cpu().numpy())
	np.save(info_path, info.cpu().numpy())
	print("Success!")

def save_imgs(trainloader, dataset):
	os.makedirs('./feat', exist_ok=True)
	img_path = './feat/imgs_{}.npy'.format(dataset)
	info_path = "./feat/info2_{}.npy".format(dataset)

	img_true = []
	label_true = []
	with torch.no_grad():
		for i, (img, iden) in enumerate(trainloader):
			img = img.to(device)
			img_true.append(img)
			label_true.append(iden)
			# save_tensor_images(img[0].detach(), './test.png')
	true_img = torch.cat(img_true, dim=0)
	true_label = torch.cat(label_true, dim=0)
	# print(img_feat.shape)
	true_info = torch.LongTensor(true_label)
	np.save(img_path, true_img.detach().cpu().numpy())
	np.save(info_path, true_info.cpu().numpy())
	print("Success!")


def get_knn_dist(E, testloader, dataset):
	feat_list = []
	label_list = []
	with torch.no_grad():
		for i, (img, iden) in enumerate(testloader):
			img = img.to(device)
			bs = img.size(0)
			feat = E(img)[-2]
			# print(feat.shape)
			feat_list.append(feat)
			label_list.append(iden)
			# save_tensor_images(img[0].detach(), './test1.png')
	img_feat = torch.cat(feat_list, dim=0)
	img_label = torch.cat(label_list, dim=0)
	# print(img_feat.shape)
	info = torch.LongTensor(img_label)
	dist = calc_knn(img_feat.detach(), info, dataset, path='./feat')

	return dist, feat.detach()

def get_cos_dist(E, testloader, dataset):
	feat_list = []
	label_list = []
	with torch.no_grad():
		for i, (img, iden) in enumerate(testloader):
			img = img.to(device)
			bs = img.size(0)
			feat = E(img)[-2]
			# print(feat.shape)
			feat_list.append(feat)
			label_list.append(iden)
	img_feat = torch.cat(feat_list, dim=0)
	img_label = torch.cat(label_list, dim=0)
	# print(img_feat.shape)
	info = torch.LongTensor(img_label)
	dist = calc_cos(img_feat.detach(), info, dataset, path='./feat')

	return dist, feat.detach()


def get_feat_dist(E, testloader, dataset):
	feat_list = []
	label_list = []
	with torch.no_grad():
		for i, (img, iden) in enumerate(testloader):
			img = img.to(device)
			bs = img.size(0)
			feat = E(img)[-2]
			feat_list.append(feat)
			label_list.append(iden)
	img_feat = torch.cat(feat_list, dim=0)
	img_label = torch.cat(label_list, dim=0)
	# print(img_feat.shape)
	info = torch.LongTensor(img_label)
	dist = calc_feat_dist(img_feat.detach(), info, dataset, path='./feat')

	return dist, feat.detach()

def psnr(img1, img2):
	mse = torch.mean((img1 - img2) ** 2)
	return 20 * torch.log10(255.0 / torch.sqrt(mse))


def get_lpips_score (device, testloader, dataset):
	loss_fn_alex = lpips.LPIPS(net='alex')

	img_reco = []
	label_reco = []
	with torch.no_grad():
		for i, (img, iden) in enumerate(testloader):
			img = img.to(device)
			# print(feat.shape)
			img_reco.append(img)
			label_reco.append(iden)
	reco_img = torch.cat(img_reco, dim=0)
	reco_label = torch.cat(label_reco, dim=0)
	# print(img_feat.shape)
	reco_info = torch.LongTensor(reco_label)

	reco_info = reco_info.cpu().long()
	reco_img = reco_img.detach().cpu()
	true_img = torch.from_numpy(np.load("./feat/imgs_{}.npy".format(dataset))).float()
	true_info = torch.from_numpy(np.load("./feat/info2_{}.npy".format(dataset))).view(-1).long()

	# info2 = reco_img.cpu().long()
	# feat = img_img2.detach().cpu()
	# true_feat = img_img.detach().cpu().float()
	# info = info.cpu().long()
	bs = reco_img.size(0)
	tot = true_img.size(0)

	mean_value = []
	for i in range(bs):
		id_img = []
		for j in range(tot):
			if true_info[j] == reco_info[i]:
				ss = loss_fn_alex(reco_img[i, :], true_img[j, :])
				id_img.append(ss)
		centroid = torch.mean(torch.tensor(id_img))
		mean_value.append(centroid)
	return torch.mean(torch.tensor(mean_value)).item()


def get_fid_score (device, trainloader, testloader):

	inception_model = inception.InceptionV3().to(device)

	recovery_list, private_list = [], []
	for i, (img, iden) in enumerate(trainloader):
		private_list.append(img.numpy())
	for i, (img, iden) in enumerate(testloader):
		recovery_list.append(img.numpy())


	recovery_images = np.concatenate(recovery_list)
	private_images = np.concatenate(private_list)

	mu_fake, sigma_fake = fid.calculate_activation_statistics(
		recovery_images, inception_model, 1, device=device
	)
	mu_real, sigma_real = fid.calculate_activation_statistics(
		private_images, inception_model, 50, device=device
	)
	fid_score = fid.calculate_frechet_distance(
		mu_fake, sigma_fake, mu_real, sigma_real
	)

	return fid_score

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = '1'
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	# dataset_name = 'pubfig'
	# dataset_name = 'facescrub'
	dataset_name = 'celeba'

	# E = FaceNet(50)
	# ckp = torch.load('~/P2I_MI/inversion/checkpoints/evaluate_model/FaceNet_99.56_allclass.tar')
	# E = torch.nn.DataParallel(E).cuda()
	# E.load_state_dict(ckp['state_dict'], strict=False)
	# E.eval()

	# E = FaceNet(200)
	# ckp = torch.load('~/P2I_MI/inversion/checkpoints/evaluate_model/FaceNet_97.82_allclass.tar')
	# E = torch.nn.DataParallel(E).cuda()
	# E.load_state_dict(ckp['state_dict'], strict=False)
	# E.eval()

	E = FaceNet(1000)
	# ckp = torch.load('~/P2I_MI/inversion/checkpoints/evaluate_model/FaceNet_97.74_allclass.tar')
	ckp = torch.load('~/P2I_MI/inversion/checkpoints/evaluate_model/FaceNet_95.88.tar')
	E = torch.nn.DataParallel(E).cuda()
	E.load_state_dict(ckp['state_dict'], strict=False)
	E.eval()
	
	print("Loading Backbone Checkpoint ")


	class dataset_pubfig(torch.utils.data.Dataset):
		def __init__(self, root='~/P2I_MI/inversion/data/pubfig50_train', transform=None,
					 feat=False):
			super(dataset, self).__init__()
			self.root = root
			self.transform = transform
			self.images = []
			self.path = self.root
			self.feat = feat

			for foldername in os.listdir(self.path):
				FolderName = os.path.join(self.path, foldername)
				if self.feat == True:
					self.images.append(FolderName)
				else:
					for filename in os.listdir(FolderName):
						self.images.append(os.path.join(FolderName, filename))
			print("Load " + str(len(self.images)) + " images")

		def __getitem__(self, index):
			img_path = self.images[index]
			if self.feat == True:
				id = int(os.path.basename(img_path).strip().split('_')[1])
			else:
				dic = {}
				f = open("~/P2I_MI/inversion/data_files/pubfig_train.txt", "r")
				for line in f.readlines():
					name1, name2, iden = line.strip().split(' ')
					dic[name1 + " " + name2] = int(iden)

				id = dic[os.path.basename(os.path.dirname(img_path))]
			img = Image.open(img_path)
			if self.transform != None:
				img = self.transform(img)
			return img, id

		def __len__(self):
			return len(self.images)
		
	class dataset_facescrub(torch.utils.data.Dataset):
		def __init__(self, root='~/P2I_MI/inversion/data/facescrub-train2', transform=None,
					 feat=False):
			super(dataset, self).__init__()
			self.root = root
			self.transform = transform
			self.images = []
			self.path = self.root
			self.feat = feat

			for foldername in os.listdir(self.path):
				FolderName = os.path.join(self.path, foldername)
				if self.feat == True:
					self.images.append(FolderName)
				else:
					for filename in os.listdir(FolderName):
						self.images.append(os.path.join(FolderName, filename))
			print("Load " + str(len(self.images)) + " images")

		def __getitem__(self, index):
			img_path = self.images[index]
			if self.feat == True:
				id = int(os.path.basename(img_path).strip().split('_')[1])
			else:
				dic = {}
				f = open("~/P2I_MI/inversion/data_files/facescrub_train.txt", "r")
				for line in f.readlines():
					name, iden = line.strip().split(' ')
					dic[name] = int(iden)

				id = dic[os.path.basename(os.path.dirname(img_path))]
			img = Image.open(img_path)
			if self.transform != None:
				img = self.transform(img)
			return img, id

		def __len__(self):
			return len(self.images)

	class dataset_celeba(torch.utils.data.Dataset):
		def __init__(self, root='~/P2I_MI/inversion/data_files/celeba_trainset.txt', transform=None,
					 feat=False):
			super(dataset_celeba, self).__init__()
			self.img_path = '~/P2I_MI/data/img_align_celeba_png'
			self.transform = transform
			self.images = []
			self.path = root
			self.feat = feat

			self.name_list, self.label_list = [], []
			f = open(self.path, "r")
			for line in f.readlines():
				img_name, iden = line.strip().split(' ')
				self.label_list.append(int(iden))
				self.name_list.append(img_name)

			for i, img_name in enumerate(self.name_list):
				if img_name.endswith(".png"):
					path = self.img_path + "/" + img_name
					img = Image.open(path)
					img = img.convert('RGB')
					self.images.append(img)
			print("Load " + str(len(self.label_list)) + " images")

		def __getitem__(self, index):
			image = self.images[index]
			id = self.label_list[index]
			if self.transform != None:
				image = self.transform(image)
			return image, id

		def __len__(self):
			return len(self.images)


	if dataset_name == 'pubfig':
		re_size = 112
		# crop_size1 = 704  # 568
		# crop_size2 = 704  # 644
		# offset_height = (1024 - crop_size1) // 2
		# offset_width = (1024 - crop_size2) // 2
		# crop = lambda x: x[:, offset_height:offset_height + crop_size1, offset_width:offset_width + crop_size2]
		pubfig_transform = transforms.Compose([
			# transforms.ToTensor(),
			# transforms.Lambda(crop),
			# transforms.ToPILImage(),
			transforms.Resize((re_size, re_size)),
			transforms.ToTensor()
		])

		crop0 = lambda x: x[:, 10:80, 15:85]
		pubfig_transform0 = transforms.Compose([
			transforms.ToTensor(),
			transforms.Lambda(crop0),
			transforms.ToPILImage(),
			transforms.Resize((re_size, re_size)),
			transforms.ToTensor()
		])
	if dataset_name == 'facescrub':
		re_size = 112
		# crop_size = 80
		# offset_height = (100 - crop_size) // 2
		# offset_width = (100 - crop_size) // 2
		crop_size1 = 704  # 568
		crop_size2 = 704  # 644
		offset_height = (1024 - crop_size1) // 2
		offset_width = (1024 - crop_size2) // 2
		crop = lambda x: x[:, offset_height:offset_height + crop_size1, offset_width:offset_width + crop_size2]
		facescrub_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Lambda(crop),
			transforms.ToPILImage(),
			transforms.Resize((re_size, re_size)),
			transforms.ToTensor()
		])


		facescrub_transform0 = transforms.Compose([
			transforms.Resize((re_size, re_size)),
			transforms.ToTensor()
		])
	if dataset_name == 'celeba':
		re_size = 112
		# crop_size1 = 704  # 568
		# crop_size2 = 704  # 644
		# offset_height = (1024 - crop_size1) // 2
		# offset_width = (1024 - crop_size2) // 2
		# crop = lambda x: x[:, offset_height:offset_height + crop_size1, offset_width:offset_width + crop_size2]
		celeba_transform = transforms.Compose([
			# transforms.ToTensor(),
			# transforms.Lambda(crop),
			# transforms.ToPILImage(),
			transforms.Resize((re_size, re_size)),
			transforms.ToTensor()
		])

		crop_size0 = 108
		offset_height0 = (218 - crop_size0) // 2
		offset_width0 = (178 - crop_size0) // 2
		crop0 = lambda x: x[:, offset_height0:offset_height0 + crop_size0, offset_width0:offset_width0 + crop_size0]
		celeba_transform0 = transforms.Compose([
			transforms.ToTensor(),
			transforms.Lambda(crop0),
			transforms.ToPILImage(),
			transforms.Resize((re_size, re_size)),
			transforms.ToTensor()
		])

	# traindata_set = dataset_pubfig(root='~/P2I_MI/inversion/data/pubfig50_train', transform=pubfig_transform0)
	# traindata_set = dataset_facescrub(root='~/P2I_MI/inversion/data/facescrub-train2', transform=facescrub_transform0)
	traindata_set = dataset_celeba(root='~/P2I_MI/inversion/data_files/celeba_trainset.txt', transform=celeba_transform0)
	trainloader = data.DataLoader(traindata_set, batch_size=50, shuffle=True)


	testdata_set = dataset(root='~/P2I_MI/inversion_success-celeba-vgg', transform=celeba_transform, feat=True)
	# testdata_set = dataset(root='~/P2I_MI/success_images-celeba-plgfacenet', transform=celeba_transform, feat=True)

	testloader = data.DataLoader(testdata_set, batch_size=1, shuffle=True)

	if not os.path.exists("./feat/feat_{}.npy".format(dataset_name)):
		save_knn(E, trainloader, dataset_name)
	if not os.path.exists("./feat/imgs_{}.npy".format(dataset_name)):
		save_imgs(trainloader, dataset_name)


	knn_dist, _ = get_knn_dist(E, testloader, dataset_name)
	print("knn_dist:", knn_dist)

	cos_dist, _ = get_cos_dist(E, testloader, dataset_name)
	print("cos_dist:", cos_dist)

	feat_dist, _ = get_feat_dist(E, testloader, dataset_name)
	print("feat_dist:", feat_dist)

	# fid_score = get_fid_score(device, trainloader, testloader)
	# print("fid_score:", fid_score)

	# lpips_score = get_lpips_score(device, testloader, dataset_name)
	# print("lpips_score:", lpips_score)
	

