# Generating data for training
- python3 dataset_pipeline/gen_data.py --dataset_dir {Path to store data directory} --plot_im_dir {Path to saving images for plotting} --mean_dir {path to save mean controls}

# Running Training Script
- python -m nn.train

With this approach we can import anything starting from the root directory.  


# Helpful Ada Commands

1. Copying stuff from local to ada
    - rsync -rv  padfoot7@10.42.0.216:/home/padfoot7/Desktop/RRC/DEB ./
# Setting up thsi repo for data collection
4. Create a python virtual-env
    ```bash
    python -m venv <**env_name**>
    ```
5. Activate the virtual-env
    ```bash
    source <**env_name**>/bin/activate
    ```
6. Install the requirements
    ```bash
    pip install -r carla/requirements.txt
    ```

# Running Carla Data Collection
1. Open 4 terminals
1. Terminal 1 - Run the following command to start the carla server
    - cd <path_to_carla>
    - ./CarlaUE4.sh

2. Terminal 2 - Start roscore
    - roscore

3. Terminal 3 - Run the teleop script
    - cd <path_to_carla>
    - python3 carla/teleop.py

4. Terminal 4 - Run the data collection script
    - cd <path_to_carla>
    - python3 carla/capture_carla_data.py
