import cv2
import os
import pygame
import time
import matplotlib.pyplot as plt
import os
import shutil
from natsort import natsorted, ns

dataset_dir = "/Users/kaustabpal/work/carla_latest/occ_map/"
plot_im_dir = "/Users/kaustabpal/work/carla_latest/plot_im/"
mean_dir = "/Users/kaustabpal/work/carla_latest/mean_controls/"



files = os.listdir(plot_im_dir)
# initialising pygame
pygame.init()
# creating display
display = pygame.display.set_mode((640, 480))

for i in range(len(files)-1):
    
    obs_pos = []
    bev_file_name = dataset_dir + "data_" + str(i).zfill(5) + ".pkl"
    im_file_name = plot_im_dir + "data_" + str(i).zfill(5) + ".png"
    mean_file_name = mean_dir + "data_" + str(i).zfill(5) + ".npy"

    img =  pygame.image.load(im_file_name).convert() #cv2.imread(file_name) #print(f[:3])

    display.blit(img, (0, 0))
    pygame.display.flip()
    
    flag = True
    while(flag):
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # checking if keydown event happened or not
            if event.type == pygame.KEYDOWN:
                # if event.key == pygame.K_t: # training set
                #     shutil.copy(bev_file_name, train_dir+bev_file_name)
                #     shutil.copy(im_file_name, train_dir+im_file_name)
                #     shutil.copy(mean_file_name, train_dir+mean_file_name)
                #     flag = False
                # if event.key == pygame.K_i: # test set
                #     shutil.copy(bev_file_name, test_dir+bev_file_name)
                #     shutil.copy(im_file_name, test_dir+im_file_name)
                #     shutil.copy(mean_file_name, test_dir+mean_file_name)
                #     flag = False
                if event.key == pygame.K_n: # training set
                    flag = False
                if event.key == pygame.K_d: # training set
                    os.remove(bev_file_name)
                    os.remove(im_file_name)
                    os.remove(mean_file_name)
                    flag = False
                if event.key == pygame.K_q:
                    flag = False
                    quit()

plt_files = os.listdir(plot_im_dir)
mean_files = os.listdir(mean_dir)
dataset_files = os.listdir(dataset_dir)

plt_files_sorted = natsorted(plt_files, key=lambda y: y.lower())
mean_files_sorted = natsorted(mean_files, key=lambda y: y.lower())
dataset_files_sorted = natsorted(dataset_files, key=lambda y: y.lower())

for i in range(len(plt_files_sorted)):
    os.replace(plot_im_dir+plt_files_sorted[i], plot_im_dir+"data_"+str(i).zfill(5) + ".png")
    os.replace(mean_dir+mean_files_sorted[i], mean_dir+"data_"+str(i).zfill(5) + ".npy")
    os.replace(dataset_dir+dataset_files_sorted[i], dataset_dir+"data_"+str(i).zfill(5) + ".pkl")
    # print("######################################################################################")

# quit()
