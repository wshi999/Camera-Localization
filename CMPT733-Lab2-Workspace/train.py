from pyrsistent import b
import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
from models.PoseNet import PoseNet, PoseLoss
from data.DataSource import *
import os
from optparse import OptionParser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(epochs, batch_size, learning_rate, save_freq, data_dir):

    train_loss_history = []

    # train dataset and train loader
    datasource = DataSource(data_dir, train=True)
    train_loader = Data.DataLoader(
        dataset=datasource, batch_size=batch_size, shuffle=True
    )

    # load model
    posenet = PoseNet().to(device)

    # loss function
    criterion = PoseLoss(0.3, 0.3, 1.0, 300, 300, 300).to(device)

    # train the network
    optimizer = torch.optim.Adam(
        nn.ParameterList(posenet.parameters()),
        lr=learning_rate,
        eps=1,
        weight_decay=0.0625,
        betas=(0.9, 0.999),
    )

    batches_per_epoch = len(train_loader.batch_sampler)
    for epoch in range(epochs):
        epoch_loss = 0
        print("Starting epoch {}:".format(epoch))
        posenet.train()
        for step, (images, poses) in enumerate(train_loader):
            b_images = Variable(images, requires_grad=True)
            b_images = b_images.type(torch.float32).to(device)

            poses[0] = np.array(poses[0])
            poses[1] = np.array(poses[1])
            poses[2] = np.array(poses[2])
            poses[3] = np.array(poses[3])
            poses[4] = np.array(poses[4])
            poses[5] = np.array(poses[5])
            poses[6] = np.array(poses[6])
            poses = np.transpose(poses)

            b_poses = Variable(torch.Tensor(poses), requires_grad=True)
            b_poses = b_poses.type(torch.float32).to(device)

            p1_x, p1_q, p2_x, p2_q, p3_x, p3_q = posenet(b_images)
            loss = criterion(p1_x, p1_q, p2_x, p2_q, p3_x, p3_q, b_poses)
            epoch_loss += loss
            epoch_loss /= 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("{}/{}: loss = {}".format(step + 1, batches_per_epoch, loss))

        train_loss_history.append(epoch_loss)
        # Save state
        if epoch % save_freq == 0:
            save_filename = "epoch_{}.pth".format(str(epoch + 1).zfill(5))
            save_path = os.path.join("checkpoints", save_filename)
            torch.save(posenet.state_dict(), save_path)

        # test
        # posenet.eval()
        # for step, (images, poses) in enumerate(test_loader):
        #     b_images = Variable(images)
        #     b_images = b_images.type(torch.float32).to(device)
        #     poses[0] = np.array(poses[0])
        #     poses[1] = np.array(poses[1])
        #     poses[2] = np.array(poses[2])
        #     poses[3] = np.array(poses[3])
        #     poses[4] = np.array(poses[4])
        #     poses[5] = np.array(poses[5])
        #     poses[6] = np.array(poses[6])
        #     poses = np.transpose(poses)
        #     b_poses = Variable(torch.Tensor(poses))
        #     b_poses = b_poses.type(torch.float32).to(device)

        #     p_xyz, p_wpqr = posenet(b_images)


def get_args():
    parser = OptionParser()
    parser.add_option("--epochs", default=100, type="int")
    parser.add_option("--learning_rate", default=0.0001)
    parser.add_option("--batch_size", default=75, type="int")
    parser.add_option("--save_freq", default=2, type="int")
    parser.add_option("--data_dir", default="data/datasets/KingsCollege/")

    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    args = get_args()

    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_freq=args.save_freq,
        data_dir=args.data_dir,
    )
