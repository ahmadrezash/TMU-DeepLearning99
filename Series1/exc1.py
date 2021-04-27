if __name__ == "__main__":
	# print(12)
	# from sklearn.datasets import load_iris
	#
	# data = load_iris()
	# print(data['target'])
	import pandas as pd

	data = pd.read_excel('./datasets/titanic3.xls')
	print(data.describe)


class NN(torch.nn.Module):
	def __init__(self,n_input=4, n_output=1):
		super(NN, self).__init__()
		self.fc = torch.nn.Linear(n_input,n_output)
	def forward(self, x):
		x = self.fc(x)

		return x

net = NN(n_input=4, n_output=1)


model = NN(4, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
epoch=1000

X = data['data']
Y = data['target']
x_train = Variable(torch.tensor(X,dtype=torch.float))
y_train = Variable(torch.tensor(Y,dtype=torch.float))

loss_history = []

for epoch in range(epoch):
	optimizer.zero_grad()

	# Forward
	y_pred = model(x_train)

	# Loss
	loss = criterion(y_pred.squeeze(), y_train)
	loss_history.append(loss)

	print(f'Epoch {epoch}: train loss: {loss.item()}')

	# Backward pass
	loss.backward()
	optimizer.step()
#%%
# plt.plot(loss_history)