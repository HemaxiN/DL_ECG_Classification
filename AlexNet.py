import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import configure_seed, configure_device, plot, ECGImageDataset


class AlexNet(nn.Module):
	def __init__(self, n_classes, **kwargs):
		"""
		Define the layers of the model

		Args:
			n_classes (int): Number of classes in our classification problem
		"""
		super(AlexNet, self).__init__()
		self.n_classes = n_classes
		self.conv2d_1 = nn.Conv2d(3,96,11,stride=4)
		#nn.Conv2d(in_channels, out_channels, kernel_size)
		self.conv2d_2 = nn.Conv2d(96, 256, 5, padding=2)
		self.conv2d_3 = nn.Conv2d(256, 384, 3, padding=1)
		self.conv2d_4 = nn.Conv2d(384, 384, 3, padding=1)
		self.conv2d_5 = nn.Conv2d(384, 256, 3, padding=1)
		self.linear_1 = nn.Linear(9216, 4096)
		self.linear_2 = nn.Linear(4096, 4096)
		self.linear_3 = nn.Linear(4096, n_classes)
		#nn.MaxPool2d(kernel_size)
		self.maxpool2d = nn.MaxPool2d(3, stride=2)
		self.relu = nn.ReLU()

	def forward(self, X, **kwargs):
		"""
		Forward Propagation

		Args:
			X: batch of training examples with dimension (batch_size, 3, 227, 227)
		"""
		x1 = self.relu(self.conv2d_1(X))
		maxpool1 =  self.MaxPool2d(x1)
		x2 = self.relu(self.conv2d_2(maxpool1))
		maxpool2 = self.MaxPool2d(x2)
		x3 = self.relu(self.conv2d_3(maxpool2))
		x4 = self.relu(self.conv2d_4(x3))
		x5 = self.relu(self.conv2d_5(x4))
		x6 = self.maxpool2d(x5)
		x6 = x6.reshape(x.shape[0],-1) #flatten (batch_size,)
		x7 = self.relu(self.linear_1(x6))
		x8 = self.relu(self.linear_2(x7))
		x9 = self.linear_3(x8)
		return x9

def train_batch(X, y, model, optimizer, criterion, gpu_id=None, **kwargs):
	"""
	X (batch_size, 3, 227, 227): batch of examples
	y (batch_size): ground truth labels
	model: Pytorch model
	optimizer: optimizer for the gradient step
	criterion: loss function
	"""
	X, y = X.to(gpu_id), y.to(gpu_id)
	optimizer.zero_grad()
	out = model(X, **kwargs)
	loss = criterion(out, y)
	loss.backward()
	optimizer.step()
	return loss.item()

def predict(model, X):
	"""
	Make label predictions for "X" (batch_size, 3, 227, 227) 
	given the trained model "model"
	"""
	probabilities = model(X) # (batch_size, n_classes)
	pred_labels = scores.argmax(dim=-1) # (batch_size)
	return pred_labels

def evaluate(model,dataloader, gpu_id=None):
	"""
	model: Pytorch model
	X (batch_size, 3, 227, 227): batch of examples
	y (batch_size): ground truth labels
	"""

	model.eval()
	with torch.no_grad():
		y_pred_all = []
		y_true_all = []
		for i, (x_batch, y_batch) in enumerate(dataloader):
			y_true_all.append(y_batch)
			print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
			x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
			y_pred = model(x_batch)
			y_pred_all.append(y_pred)

		n_correct = (y==ypred).sum().item()
		n_possible = float(y.shape[0])
		model.train()
	return n_correct/n_possible





def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-data', default=None,
						help="Path to the dataset.")
	parser.add_argument('-epochs', default=100, type=int,
						help="""Number of epochs to train the model.""")
	parser.add_argument('-batch_size', default=4, type=int,
						help="Size of training batch.")
	parser.add_argument('-learning_rate', type=float, default=0.01)
	parser.add_argument('-l2_decay', type=float, default=0)
	parser.add_argument('-optimizer',
						choices=['sgd', 'adam'], default='sgd')
	parser.add_argument('-gpu_id', type=int, default=None)
	opt = parser.parse_args()

	configure_seed(seed=42)
	configure_device(opt.gpu_id)

	print("Loading data...")
	train_dataset = ECGImageDataset(opt.data, 'train')
	dev_dataset = ECGImageDataset(opt.data, 'dev')
	test_dataset = ECGImageDataset(opt.data, 'test')

	train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
	dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
	test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


	n_classes = torch.unique(dataset.y).shape[0]  # 5

	# initialize the model
	model = AlexNet(n_classes)
	model = model.to(gpu_id)

	# get an optimizer
	optims = {
		"adam": torch.optim.Adam,
		"sgd": torch.optim.SGD}

	optim_cls = optims[opt.optimizer]
	optimizer = optim_cls(
		model.parameters(),
		lr=opt.learning_rate,
		weight_decay=opt.l2_decay)

	# get a loss criterion
	criterion = nn.CrossEntropyLoss()

	# training loop
	epochs = torch.arange(1, opt.epochs + 1)
	train_mean_losses = []
	valid_accs = []
	train_losses = []
	for ii in epochs:
		print('Training epoch {}'.format(ii))
		for i, (X_batch, y_batch) in enumerate(train_dataloader):
			print('{} of {}'.format(i + 1, len(train_dataloader)), end='\r')
			loss = train_batch(
				X_batch, y_batch, model, optimizer, criterion, gpu_id=opt.gpu_id)
			train_losses.append(loss)

		mean_loss = torch.tensor(train_losses).mean().item()
		print('Training loss: %.4f' % (mean_loss))

		train_mean_losses.append(mean_loss)
		valid_accs.append(evaluate(model, dev_dataloader, gpu_id=opt.gpu_id))
		print('Valid acc: %.4f' % (valid_accs[-1]))

	print('Final Test acc: %.4f' % (evaluate(model, test_dataloader, gpu_id=opt.gpu_id)))
	# plot
	plot(epochs, train_mean_losses, ylabel='Loss', name='{}-training-loss-{}-{}-{}-{}-{}'.format(opt.model, opt.learning_rate, opt.hidden_sizes[0], opt.dropout, opt.activation, opt.optimizer))
	plot(epochs, valid_accs, ylabel='Accuracy', name='{}-validation-accuracy-{}-{}-{}-{}-{}'.format(opt.model, opt.learning_rate, opt.hidden_sizes[0], opt.dropout, opt.activation, opt.optimizer))


if __name__ == '__main__':
	main()
