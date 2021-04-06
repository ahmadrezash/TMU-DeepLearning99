if __name__ == "__main__":
	# print(12)
	# from sklearn.datasets import load_iris
	#
	# data = load_iris()
	# print(data['target'])
	import pandas as pd

	data = pd.read_excel('./datasets/titanic3.xls')
	print(data.describe)
