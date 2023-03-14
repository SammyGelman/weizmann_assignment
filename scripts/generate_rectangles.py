#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import imageio


def spore_rectangles(spore_table, n_spores, max_length):
    length = len(spore_table)
    width = len(spore_table[0])

    len_range = np.linspace(0, length - 1, length).astype(int)
    len_width = np.linspace(0, width - 1, width).astype(int)

    spored = 0
    holder = 0.1
    idx_length = (max_length - 1)

    while spored < n_spores:
        spore = True
        prop_x = np.random.choice(len_range)
        prop_y = np.random.choice(len_width)

        x_range = np.linspace(prop_x - idx_length, prop_x + idx_length,
                              2 * idx_length + 1).astype(int)

        y_range = np.linspace(prop_y - idx_length, prop_y + idx_length,
                              2 * idx_length + 1).astype(int)

        for i in x_range:
            for j in y_range:
                #check to make it is a valid index
                if (i and j) >= 0 and i < length and j < width:
                    if spore_table[i, j] != 0:
                        spore = False
                        break
        if spore:
            for i in x_range:
                for j in y_range:
                    if i >= 0 and j >= 0 and i < length and j < width:
                        spore_table[i, j] = holder
            spore_table[prop_x, prop_y] += 1
            spored += 1

    spore_table -= holder
    return spore_table


def map_spores(growing_table, spore_table, spore_length, spore_width,
               max_length):
    for i in range(spore_length):
        for j in range(spore_width):
            growing_table[i + max_length, j + max_length] = spore_table[i, j]
    # print(growing_table.shape)
    # print(spore_table.shape)
    return growing_table


def grow_mushrooms(growing_table, max_length):
    length = len(growing_table)
    width = len(growing_table[0])
    output = np.zeros((length, width))
    for i in range(length):
        for j in range(width):
            if growing_table[i, j] == 1:
                h = round(np.random.random() * (max_length - 1) + 1)
                w = round(np.random.random() * (max_length - 1) + 1)
                cv2.rectangle(output, (i, j), (i + h, j + w), 1, -1)
    return output


def generate_data(max_length, max_mushrooms, length, width, n_samples):
    bank = []
    spore_length = length - 2 * max_length
    spore_width = width - 2 * max_length

    for i in range(n_samples):
        spore_table = np.zeros((spore_length, spore_width))
        growing_table = np.zeros((length, width))
        # n_spores = round(max_mushrooms*np.random.random())
        n_spores = max_mushrooms
        spore_rectangles(spore_table, n_spores, max_length)

        map_spores(growing_table, spore_table, spore_length, spore_width,
                   max_length)

        sample = grow_mushrooms(growing_table, max_length)

        #normalize sample and convert to unit8
        sample_n = cv2.normalize(src=sample,
                                 dst=None,
                                 alpha=0,
                                 beta=255,
                                 norm_type=cv2.NORM_MINMAX,
                                 dtype=cv2.CV_8U)

        #save array to jpeg image in correct folder
        imageio.imwrite('../rectangles/rectangle_' + str(i) + '.jpeg',
                        sample_n)


#     bank.append(grow_mushrooms(growing_table, max_length))
#
# with open('sticks.pkl', 'wb') as f:
#     pickle.dump(bank, f)
#
generate_data(20, 1, 64, 64, 500)
