import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import argparse
import pickle

from custom_lstm import CustomLSTM as LSTMRaw
from plstm_cell import PLSTM

PI = torch.acos(torch.zeros(1)).item() * 2

def sin_wave_iterator(min_period=1, max_period=100, 
                      min_spec_period=5, max_spec_period=6, 
                      batch_size=32, num_examples=10000, 
                      min_duration=15, max_duration=125, 
                      min_num_points=15, max_num_points=125,
                      sample_res=0.5, async_sample=True):

    # Calculate constants
    num_batches = int(math.ceil(float(num_examples)/batch_size))
    min_log_period, max_log_period = torch.log(torch.tensor(min_period)), torch.log(torch.tensor(max_period))

    b = 0
    while b < num_batches:
        # Choose curve and sampling parameters
        num_points = (max_num_points - min_num_points) * torch.rand(batch_size) + min_num_points
        duration = (max_duration - min_duration) * torch.rand(batch_size) + min_duration
        start = (max_duration - duration) * torch.rand(batch_size)
        periods = torch.exp((max_log_period - min_log_period) * torch.rand(batch_size) + min_log_period)
        shifts = duration * torch.rand(batch_size)

        # Ensure always at least half is special class
        periods[:int(batch_size/2)] = (max_spec_period - min_spec_period) * torch.rand(int(batch_size/2)) + min_spec_period

        # Define arrays of data to fill in
        all_t = []
        all_masks = []
        all_wavs = []
        for idx in range(batch_size):
            if async_sample:
                # Asynchronous condition
                sorted_ts, _ = torch.sort(torch.rand(num_points[idx].type(torch.int)))
                t = sorted_ts*duration[idx]+start[idx]
            else:
                # Synchronous condition, evenly sampled
                t = torch.arange(start[idx], start[idx]+duration[idx], step=sample_res)                
            wavs = torch.sin(2.*PI/periods[idx]*t-shifts[idx])
            mask = torch.ones(wavs.size())
            all_t.append(t)
            all_masks.append(mask)
            all_wavs.append(wavs)

        # Now pack all the data down into masked matrices
        lengths = torch.tensor([len(item) for item in all_masks])
        max_length = torch.max(lengths)
        bXt = torch.zeros(batch_size, max_length)
        bXmask = torch.zeros(batch_size, max_length)
        bX = torch.zeros(batch_size, max_length, 1)
        for idx in range(batch_size):
            bX[idx, max_length-lengths[idx]:, 0] = all_wavs[idx]
            bXmask[idx, max_length-lengths[idx]:] = all_masks[idx]
            bXt[idx, max_length-lengths[idx]:] = all_t[idx]

        # Define and calculate labels
        bY = torch.zeros(batch_size)
        bY[(periods>=min_spec_period)*(periods<=max_spec_period)] = 1

        # Yield data
        yield bX.type(torch.float), bXmask.type(torch.bool), bXt.type(torch.float), bY.type(torch.LongTensor)
        b += 1    


class Net(nn.Module):

    def __init__(self, inp_dim=1, hidden_dim=20, use_lstm=True):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_lstm = use_lstm

        if use_lstm:
            pass
            # One extra vector for time            
            self.rnn = LSTMRaw(inp_dim + 1, hidden_dim)
        else:
            self.rnn = PLSTM(inp_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 2)

    def forward(self, points, times):         
        if self.use_lstm:
            combined_input = torch.cat((points, torch.unsqueeze(times, dim=-1)), -1)      
            lstm_out, _ = self.rnn(combined_input)
        else:
            lstm_out, _ = self.rnn(points, times)
        linear_out = self.linear(lstm_out)
        final_logits = linear_out[:, -1, :]
        classes = F.log_softmax(final_logits, dim=1)
        return classes


def train(args, model, device, train_loader, optimizer, epoch, clip_value=100):
    model.train()
    for batch_idx, (bX, bXmask, bT, bY) in enumerate(train_loader):
        bX, bXmask = bX.to(device), bXmask.to(device)
        bT, bY = bT.to(device), bY.to(device)
        optimizer.zero_grad()
        output = model(bX, bT)
        loss = F.nll_loss(output, bY)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                batch_idx, loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for b_idx, (bX, bXmask, bT, bY) in enumerate(test_loader):
            total += int(bXmask[:,-1].sum())
            bX, bXmask = bX.to(device), bXmask.to(device)
            bT, bY = bT.to(device), bY.to(device)
            output = model(bX, bT)
            test_loss += F.nll_loss(output, bY, reduction='sum').item() 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(bY.view_as(pred)).sum().item()
    test_loss /= total

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))
    return correct / total


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--async_sample', action='store_true', default=True,
                        help='Sample waves asynchronously')    
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    print('Sample data asynchronously? {}'.format(args.async_sample))

    lstm_test_acc, plstm_test_acc = [], []
    print('Training LSTM.')
    model = Net(use_lstm=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    test(model, device, sin_wave_iterator(num_examples=1000, async_sample=args.async_sample))
    for epoch in range(1, args.epochs + 1):   
        train_data = sin_wave_iterator(async_sample=args.async_sample)
        test_data = sin_wave_iterator(num_examples=1000, async_sample=args.async_sample)
        train(args, model, device, train_data, optimizer, epoch)
        test_acc = test(model, device, test_data)
        lstm_test_acc.append(test_acc)

    print('Training Phased LSTM.')
    model = Net(use_lstm=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    test(model, device, sin_wave_iterator(num_examples=1000, async_sample=args.async_sample))
    if args.save_model:
        torch.save(model.state_dict(), "plstm_0.pt")    
    for epoch in range(1, args.epochs + 1):   
        train_data = sin_wave_iterator(async_sample=args.async_sample)
        test_data = sin_wave_iterator(num_examples=1000, async_sample=args.async_sample)
        train(args, model, device, train_data, optimizer, epoch)
        test_acc = test(model, device, test_data)
        plstm_test_acc.append(test_acc)
        if args.save_model:
            torch.save(model.state_dict(), "plstm.pt")

    pickle.dump( {'lstm':lstm_test_acc, 'plstm': plstm_test_acc}, 
        open( "acc.pkl", "wb" ) )

if __name__ == '__main__':
    main()