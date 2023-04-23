import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy
from torch.utils.data import Dataset
import argparse
from parallel_nets.spikeLayer import *
import os

N = 500
r_in = 4
T = 500
w_mean = 0.01
threshold = 1
v_rest = 0
h = 0.0001
tau = 3
g_leak = 0.001


def my_cross_entropy(input, target, reduction="mean"):
    exp = torch.exp(input)
    tmp1 = exp.gather(1, target.unsqueeze(-1)).squeeze()
    tmp2 = exp.sum(1)
    softmax = tmp1 / tmp2
    log = -torch.log(softmax)
    if reduction == "mean": return log.mean()
    else: return log.sum()

def my_cross_entropy_d(input, target, reduction="mean"):
    for k in range(len(target)):
        if target[k] < 5:
            input[k] = -input[k]

    exp = torch.exp(input)
    tmp1 = exp.gather(1, target.unsqueeze(-1)).squeeze()
    tmp2 = exp.sum(1)
    softmax = tmp1 / tmp2
    log = -torch.log(softmax)
    if reduction == "mean": return log.mean()
    else: return log.sum()

def data_model_load(args, model, kwargs):
    path = os.path.join(os.path.join(os.getcwd(), 'data'), 'Ti46')
    train_dataset = Ti46_dataset(path, train=True, transform=Compose([Normalize_ToTensor()]))
    test_dataset = DVSGestureDataset(path, train=False, transform=Compose([Normalize_ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    if args.pretrained:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print('Pretrained model loaded.')
    else:
        start_epoch = 0
        print('Model loaded.')
    return train_loader, test_loader, start_epoch

def train(args, model, train_loader, optimizer, device, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data_temp, target = data.to(device), target.to(device)
        bs = data_temp.shape[0]
        data = torch.zeros((TimeStep*bs,) + data_temp.shape[1:], device=data_temp.device)
        for t in range(TimeStep):
            data[t*bs:(t+1)*bs, ...] = data_temp
        output = model(data)
        loss = my_cross_entropy_d(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * args.batch_size, len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, test_loader, device):
    model.eval()
    total_loss = 0.
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data_temp, target = data.to(device), target.to(device)
            bs = data_temp.shape[0]

            data = torch.zeros((TimeStep*bs,) + data_temp.shape[1:], device=data_temp.device)
            for t in range(TimeStep):
                data[t * bs:(t + 1) * bs, ...] = data_temp

            output = model(data)
            total_loss += my_cross_entropy_d(output, target, reduction='sum').item()
            pre_result = output.argmax(dim=1, keepdim=True)
            correct += pre_result.eq(target.view_as(pre_result)).sum().item()

    total_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        total_loss, correct, len(test_loader.dataset),
        accuracy))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--batch-size', type=int, default=1, help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=32, help='input batch size for testing')
    parser.add_argument('--total-epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--use-cuda', action='store_true', default=True, help='use CUDA training')
    parser.add_argument('--save', action='store_true', default=True, help='save model')
    parser.add_argument('--pretrained', action='store_true', default=False, help='use pre-trained model')
    parser.add_argument('--log-interval', type=int, default=40,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model-interval', type=int, default=10,
                        help='save model every save_model_interval')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoint/cifar10/checkpoint.pth',
                        help='use CUDA training')
    args = parser.parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    model = resnet20().to(device)

    train_loader, test_loader, start_epoch = data_model_load(args, model, kwargs)
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=momentum_SGD, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, 0.1)
    for _ in range(start_epoch):
        scheduler.step()
    for epoch in range(start_epoch + 1, args.total_epochs + 1):
        start_time = time.time()
        train(args, model, train_loader, optimizer, device, epoch)
        test(args, model, test_loader, device)
        waste_time = time.time() - start_time
        print('One epoch wasting time:{:.0f}s, learning rate:{:.8f}\n'.format(
            waste_time, optimizer.state_dict()['param_groups'][0]['lr']))
        if epoch % args.save_model_interval == 0:
            if args.save:
                state = {'model': model.state_dict(), 'epoch': epoch}
                torch.save(state, args.checkpoint_path)
                print("saved")
        scheduler.step()

