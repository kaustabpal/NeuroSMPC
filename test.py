import pickle
import numpy as np

def main():
    frame = 0

    with open('data/carla_dyn2/temp/storm/data_0.pkl', 'rb') as f:
        data = pickle.load(f)

    controls = np.load('data/carla_dyn2/mean_controls/data_0.npy')
    breakpoint()


if __name__=='__main__':
    main()