import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from nwd import NuclearWassersteinDiscrepancy
from test_model import resnet34
import analyse
from test_dataloader import GetLoader



def run_successful(flag):
    print("*********run " + flag + " OK !!*******")





def main(args):
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    if args.dataset == 'home':
        data_root = "./office-home"
        data_root_s = "./office-home/Art"
        data_root_t = "./office-home/Clipart"
        data_root_test = "./office-home/Clipart"
        n_class = 65
    elif args.dataset == 'office31':
        data_root = "./office31"
        data_root_s = "./office31/amazon"
        data_root_t = "./office31/dslr"
        data_root_test = "./office31/dslr"
        n_class = 31
    assert os.path.exists(data_root), "{} path does not exist.".format(data_root)

    n_epoch = args.epochs
    batch_size = args.batchsize

    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224),  # 随机裁剪一个area然后再resize
         transforms.RandomHorizontalFlip(),  # 随机水平翻转
         transforms.ToTensor(),  # 归一化到[0.0,1.0]
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])
    contra_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
         transforms.RandomGrayscale(p=0.2),
         transforms.ToTensor(),
         transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
         ])
    test_transform = transforms.Compose(
        [transforms.Resize((288, 288)),
         transforms.ToTensor(),  # 归一化到[-1.0,1.0]
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])



    source_dataset = GetLoader(dataroot=data_root_s, transforms=train_transform, contra_transforms=contra_transform)
    data_loader_source = torch.utils.data.DataLoader(source_dataset, batch_size=batch_size, shuffle=True,
                                                     drop_last=True, num_workers=8)
    target_dataset = GetLoader(dataroot=data_root_t, transforms=test_transform, contra_transforms=contra_transform)
    data_loader_target = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=True,
                                                     drop_last=True, num_workers=8)
    # source_dataset = datasets.ImageFolder(root=data_root_s, transform=train_transform)
    # target_dataset = datasets.ImageFolder(root=data_root_t, transform=test_transform)
    source_num = source_dataset.n_data
    target_num = target_dataset.n_data
    # source_num = len(source_dataset)
    # target_num = len(target_dataset)

    class_list = source_dataset.class_to_idx
    # print(class_list)
    print("using {} images for training, {} images for validation.".format(source_num, target_num))


    net = resnet50()
    model_weight_path = "./pretrained/resnet34-333f7ec4.pth"
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 31)
    net.to(device)


    loss_function = nn.CrossEntropyLoss()


    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = n_epoch
    accuracy = []
    best_acc = 0.0
    save_path = './results/resNet50.pth'
    train_steps = len(data_loader_source)
    for epoch in range(epochs):
        print("start training epoch ", epoch + 1)

        net.train()
        running_loss = 0.0
        train_bar = tqdm(data_loader_source, file=sys.stdout)
        discrepancy = NuclearWassersteinDiscrepancy(classifier.head)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            discrepancy_loss = -discrepancy(f)
            transfer_loss = discrepancy_loss * trade_off_lambda
            loss = loss_function(logits, labels.to(device)) + transfer_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(data_loader_target, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)
        val_accurate = acc / target_num
        accuracy.append(val_accurate)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    with open("./results/acc.txt", "w") as fp:
        [fp.write(str(item) + '\n') for item in accuracy]
        fp.close()

    analyse.draw_accuracy("./results/acc.txt", epoch, "./results/")
    print("best accuracy is:", best_acc)
    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='office31', choices=["office31", "home", "visda"])
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of total epochs to run')
    args = parser.parse_args()
    main(args)
