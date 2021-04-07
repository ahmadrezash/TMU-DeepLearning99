import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


class Trainer:

	def __init__(self, model, epoch=None, criterion=None, optimizer=None) -> None:
		super().__init__()
		self.model = model
		self.epoch = epoch or 1000
		self.criterion = criterion or torch.nn.MSELoss()
		self.optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=0.01)
		self.loss_history = []

	def fit(self, x_train, y_train):
		x_train = Variable(torch.tensor(x_train, dtype=torch.float))
		y_train = Variable(torch.tensor(y_train, dtype=torch.float))

		epoch = self.epoch
		model = self.model
		criterion = self.criterion
		optimizer = self.optimizer

		self.loss_history = []
		for i in range(epoch):
			optimizer.zero_grad()

			# Forward
			y_pred = model(x_train)

			# Loss
			loss = criterion(y_pred.squeeze(), y_train)
			self.loss_history.append(loss)
			if not i % 50:
				print(f'Epoch {i}: train loss: {loss.item()}')

			# Backward pass
			loss.backward()
			optimizer.step()

	def plot_loss(self):
		plt.plot(self.loss_history)
