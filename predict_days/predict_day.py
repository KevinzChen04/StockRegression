import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import sys
import pickle

def usage():
	print(f"Usage:\n{sys.executable} predict_day.py number_of_days")
	print("EXIT FAILURE")

class LSTM(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
		super(LSTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
		self.fc = torch.nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		out, (hn, cn) = self.lstm(x)
		last_time_step = out[:, -1, :]
		predictions = self.fc(last_time_step)
		return predictions

def pred_next_day(model, X):
	return model(X[-1].unsqueeze(0))

def pred_N_day(model, N, X, first):
	next = pred_next_day(model, X)
	res = [next.detach().squeeze().tolist()]
	for i in range(N-1):
		last = X[-1]
		last = torch.cat((last[1:], next))
		X = torch.cat((X, last.unsqueeze(0)))
		next = pred_next_day(model, X)
		res.append(next.detach().squeeze().tolist())
	dates = pd.date_range(start="2023-12-29", periods=N + 1, freq='D')[1:]
	res = pd.DataFrame(res, index=dates, columns=first.index)
	res = first + res
	return res


def main(num_days: int):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	hidden_dim = 128
	num_layers = 3
	dropout = 0.3

	model = LSTM(
		input_dim=5,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		output_dim=5,
		dropout=dropout
	)
	model = model.to(device)
	model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
	first_series = pd.read_pickle('first.pkl')
	with open('X.pkl', 'rb') as f:
		X = pickle.load(f)
	X = X.unsqueeze(0)
	
	print(pred_N_day(model, num_days, X, first_series))


if __name__=='__main__':
	if len(sys.argv) != 2:
		usage()
		sys.exit()
	else:
		num_days = sys.argv[1]
	try:
		num_days = int(num_days)
	except Exception as e:
		print(f"Invalid num_days: {num_days}")
		usage()
		sys.exit()
	main(num_days)
	sys.exit()
