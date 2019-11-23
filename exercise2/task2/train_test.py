import torch
import torch.optim as optim
from torch.autograd import Variable
import config as cf
import time
import numpy as np
from torch.optim import lr_scheduler

use_cuda = torch.cuda.is_available()

best_acc = 0

def train(epoch, net, trainloader, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.Adam(net.parameters(), lr=cf.lr, weight_decay=5e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    train_loss_stacked = np.array([])
    exp_lr_scheduler.step()

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.lr))
    for batch_idx, (inputs_value, targets) in enumerate(trainloader):
        if use_cuda:
            inputs_value, targets = inputs_value.cuda(), targets.cuda() # GPU settings
        #inputs_value, targets = Variable(inputs_value), Variable(targets)
        outputs = net(inputs_value)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        optimizer.zero_grad()
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        train_loss_stacked = np.append(train_loss_stacked, loss.item())
    print ('| Epoch [%3d/%3d] \t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, cf.num_epochs, loss.item(), 100.*correct/total))

    return train_loss_stacked


def test(epoch, net, testloader, criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_loss_stacked = np.array([])
    for batch_idx, (inputs_value, targets) in enumerate(testloader):
        if use_cuda:
            inputs_value, targets = inputs_value.cuda(), targets.cuda()
        with torch.no_grad():
            inputs_value, targets = Variable(inputs_value), Variable(targets)
        outputs = net(inputs_value)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        test_loss_stacked = np.append(test_loss_stacked, loss.item())


    # Save checkpoint when best model
    acc = 100. * correct / total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" % (epoch, loss.item(), acc))



    if acc > best_acc:
        best_acc = acc
    print('* Test results : Acc@1 = %.2f%%' % (best_acc))

    return test_loss_stacked

def start_train_test(net,trainloader, testloader, criterion):
    elapsed_time = 0

    for epoch in range(cf.start_epoch, cf.start_epoch + cf.num_epochs):
        start_time = time.time()

        train_loss = train(epoch, net, trainloader, criterion)
        test_loss = test(epoch, net, testloader, criterion)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

    return train_loss.tolist(), test_loss.tolist()

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
