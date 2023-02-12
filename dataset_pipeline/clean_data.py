import cv2
import os
import pygame
import time
import matplotlib.pyplot as plt
import os
import shutil

dataset_dir = "occ_map/"
plot_im_dir = "plot_im/"
mean_dir = "mean_controls/"
train_dir = "train/"
test_dir = "test/"

files = os.listdir(plot_im_dir)
# initialising pygame
pygame.init()
# creating display
display = pygame.display.set_mode((640, 480))

for i in range(2,len(files)-1):
    
    obs_pos = []
    bev_file_name = dataset_dir + "data_" + str(i).zfill(2) + ".pkl"
    im_file_name = plot_im_dir + "data_" + str(i) + ".png"
    mean_file_name = mean_dir + "data_" + str(i) + ".npy"

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

    #         # checking if key "J" was pressed
    #         if event.key == pygame.K_j:
    #             print("Key J has been pressed")

    #         # checking if key "P" was pressed
    #         if event.key == pygame.K_p:
    #             print("Key P has been pressed")

    #         # checking if key "M" was pressed
    #         if event.key == pygame.K_m:
    #             print("Key M has been pressed")
    # cv2.destroyAllWindows() # destroys the window showing image
