import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.autograd import Variable


class Trainer:

	def __init__(self, model, epoch=None, criterion=None, optimizer=None) -> None:
		super().__init__()
		self.model = model
		self.epoch = epoch or 1000
		self.criterion = criterion or torch.nn.MSELoss()
		self.optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=0.01)
		self.loss_history = []
		self.valid_loss_history = []

	def fit(self, X, Y):
		x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=.5)

		x_train = Variable(torch.tensor(x_train, dtype=torch.float))
		y_train = Variable(torch.tensor(y_train, dtype=torch.float))

		x_test = Variable(torch.tensor(x_test, dtype=torch.float))
		y_test = Variable(torch.tensor(y_test, dtype=torch.float))

		epoch = self.epoch
		model = self.model
		criterion = self.criterion
		optimizer = self.optimizer

		self.loss_history = []
		self.valid_loss_history = []

		# Validation Step1
		y_pred = model(x_test)
		loss = criterion(y_pred.squeeze(), y_test)
		self.valid_loss_history.append(loss)

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

			# Validation
			y_pred = model(x_test)
			loss = criterion(y_pred.squeeze(), y_test)
			self.valid_loss_history.append(loss)

		self.plot_loss()

	def plot_loss(self):
		plt.plot(self.loss_history, label="train_data")
		plt.plot(self.valid_loss_history, label="test_data")
		plt.legend()
