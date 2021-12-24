import os
import random
import numpy as np
import cv2
import paddle.fluid as fluid

class BasicDataLoader():
    def __init__(self,
                 image_folder,
                 image_list_file,
                 transform=None,
                 shuffle=True):

    def read_list(self):

    def preprocess(self, data, label):

    def __len__(self):

    def __call__(self):




def main():
    batch_size = 5
    place = fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        # TODO: craete BasicDataloder instance
        # image_folder="./dummy_data"
        # image_list_file="./dummy_data/list.txt"

        # TODO: craete fluid.io.DataLoader instance

        # TODO: set sample generator for fluid dataloader


        num_epoch = 2
        for epoch in range(1, num_epoch+1):
            print(f'Epoch [{epoch}/{num_epoch}]:')
            for idx, (data, label) in enumerate(dataloader):
                print(f'Iter {idx}, Data shape: {data.shape}, Label shape: {label.shape}')

if __name__ == "__main__":
    main()
