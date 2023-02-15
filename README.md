# Generating data for training
- python3 dataset_pipeline/gen_data.py --dataset_dir {Path to store data directory} --plot_im_dir {Path to saving images for plotting} --mean_dir {path to save mean controls}

# Running Training Script
- python -m nn.train

With this approach we can import anything starting from the root directory.  


# Helpful Ada Commands

1. Copying stuff from local to ada
    - rsync -rv  padfoot7@10.42.0.216:/home/padfoot7/Desktop/RRC/DEB ./