import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def mnist(train_batch_size, test_batch_size, amp=False):
    if amp:
        transform = transforms.Compose([
                                        transforms.RandomAffine(10, (0.05, 0.05)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)), ])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)), ])

    trainset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

    testset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True)

    demo_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    return train_loader, test_loader, demo_loader

def fashionmnist(train_batch_size, test_batch_size, amp=False):
    if amp:
        transform = transforms.Compose([
                                        transforms.RandomAffine(10, (0.05, 0.05)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)), ])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)), ])

    trainset = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

    testset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True)

    demo_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    return train_loader, test_loader, demo_loader

def cifar10(train_batch_size, test_batch_size, amp=False):
    if amp:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True)

    demo_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    return train_loader, test_loader, demo_loader

def cifar100(train_batch_size, test_batch_size, amp=False):
    if amp:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True)

    demo_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    return train_loader, test_loader, demo_loader