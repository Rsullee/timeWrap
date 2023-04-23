import argparse
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from neu_dynamics import param_init
from datetime import datetime
from utils import *
import numpy as np
import random

parser = argparse.ArgumentParser(description='A Deep Spike Learning through Critical Time Points')
parser.add_argument('--dataset', default='MNIST', help='dataset, MNIST, FashionMNIST')
parser.add_argument('--model', default='MNISTDNN',
                    help='models including MNIST(MNISTDNN, MNISTCNN), Fashion(FashionDNN, FashionCNN)')
parser.add_argument('--batch-size', type=int, default=100, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--lr-decay-epoch', type=int, default=60, help='learning rate decays after this epochs')
parser.add_argument('--thresh', type=float, default=0.5, help='neuronal threshold')
parser.add_argument('--thresh-pooling', type=float, default=0.25, help='threshold in pooling layer')
parser.add_argument('--time-window', type=int, default=50, help='time steps')
parser.add_argument('--save-path', default='', type=str, help='the directory used to save the trained models')
parser.add_argument('--resume', action='store_true', default=False, help='load trained model')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--tau', type=int, default=1, help='tau factor')
parser.add_argument('--optim', type=str, default='adam', help='optimizer including adam and sgd')
parser.add_argument('--encoding', type=str, default='poisson',
                    help='encoding schemes including real, poisson, latency')
parser.add_argument('--spike', type=int, default=10, help='desired spike number for target neuron')



args = parser.parse_args()
if args.save_path is '':
    save_path = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
else:
    save_path = args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

setup_logging(os.path.join(save_path, 'log.txt'))
logging.info("saving to %s", save_path)
logging.info('args:' + str(args))

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if is_cuda:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(args.seed)

num_class = 10

best_acc = 0.

data_path = '../../data/'
if args.dataset == 'MNIST':
    train_dataset = torchvision.datasets.MNIST(root=data_path+'MNIST',
                                               train=True,
                                               transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root=data_path+'MNIST', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False)
elif args.dataset == 'FashionMNIST':

    train_dataset = torchvision.datasets.FashionMNIST(root=data_path+'FashionMNIST',download=False,
                                               train=True,
                                              transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.FashionMNIST(root=data_path+'FashionMNIST', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False)
else:
    raise Exception('No valid dataset is specified.')




param_init(args.thresh, args.thresh_pooling, torch.exp(torch.tensor(-1 / args.tau, device=device)))
params = {
    'thresh': args.thresh,
    'encoding': args.encoding,
    'device': device,
    'inf_mem': 9999
}


if args.model == 'FashionCNN':
    from parallel_nets.FashionCNN import FashionCNN
    model = FashionCNN(params)
else:
    raise Exception('No valid model is specified.')



model = model().to(device)


best_acc = 0.0
start_epoch = 1
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(os.path.join(save_path, 'checkpoint.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    print(best_acc)

if args.optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
else:
    raise Exception('No valid optimizer is specified.')

def evaluate(test_loader, model):
    with torch.no_grad():
        correct = 0
        total = 0
        model.eval()
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            outputs, _, _ = model(images, args.time_window)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return 100.0 * correct / total

for epoch in range(start_epoch, args.epochs+1):

    total_step = len(train_loader)
    running_loss = 0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)

        outputs, max_mem, min_mem = model(images, args.time_window)
        labels_ = torch.zeros(images.size(0), 10).scatter_(1, labels.view(-1, 1), args.spike).to(device)
        membrane = torch.where(outputs.detach() < labels_, -max_mem, torch.tensor(0., device=device))
        membrane = torch.where(outputs.detach() > labels_, min_mem, membrane)

        # Backward
        optimizer.zero_grad()
        membrane.mean().backward()
        optimizer.step()

    optimizer = lr_scheduler(optimizer, epoch, args.lr_decay_epoch)
    acc = evaluate(test_loader, model)
    logging.info('Iters: %s', epoch)
    logging.info('Test Accuracy of the model on the  test images: %.5f', acc)

    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
    save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_acc': best_acc,
                     'optimizer': optimizer.state_dict(), },
                    is_best, save_path)
logging.info('best acc' + str(best_acc))



