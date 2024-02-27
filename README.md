# Fitbit Clustering

This repository contains code that clusters FitBit data and pain surveys from chronic pain patients undergoing Deep Brain Stimulation (DBS).

## Data and Results

- FitBit data and pain surveys for RCS02, RCS04, RCS06, and RCS07 can be found at `SUBNETS Dropbox/Chronic Pain - Activa and Summit 2.0/DATA ANALYSIS/Aditya Behal/clustering/data`.

- Results for clustering runs with a wide variety of parameters for both the regular script and HRV script can be found at `SUBNETS Dropbox/Chronic Pain - Activa and Summit 2.0/DATA ANALYSIS/Aditya Behal/clustering/fitbit-clustering-repo-X` where X ranges from 1 to 12.

## Clustering Scripts

The repository includes two Python scripts for clustering data using the HDBSCAN algorithm:

- `clustering-GPU-minutes.py`: Clusters MPQtotal, heart rate, and minutes asleep.

- `clustering-GPU-minutes-HRV.py`: Clusters MPQtotal, heart rate variability (rmssd), and minutes asleep.

## Spirit Server Setup

The Chang lab server spirit has been configured to run both of these scripts. Before running the scripts, you need to perform a one-time setup on the server pia.

## One Time Setup on pia:

1. Log onto pia.
   
2. Git clone this repo to your folder.

3. Copy the FitBit and pain survey data into the folder `fitbit-clustering-repo`, which was created in the previous step.

4. Clone the Anaconda environment with the following command: `/opt/anaconda3_2023/bin/conda create -p /userdata/<user_name>/myenv --clone /userdata/abehal/myenv`

## Running the Scripts on spirit:

1. Log onto spirit.
   
2. Navigate to be inside the folder `fitbit-clustering-repo`.

3. Activate the conda environment: `source /opt/anaconda3/bin/activate /userdata/<user_name>/myenv`

4. To run the scripts off the job queue on the spirit server, follow these steps:

    a) For the script `clustering-GPU-minutes.py`, run the following command:

    ```bash
    nohup python -u clustering-GPU-minutes.py --pt_ids RCS02 RCS04 RCS06 RCS07 --washout_period W --activity_bin_before_pain_survey B --activity_bin_after_pain_survey A --sleep_binning_direction D --n_jobs J 2> clustering-final-stderr.txt 1> clustering-final-stdout.txt &
    ```

    b) For the script `clustering-GPU-minutes-HRV.py`, run the following command:

    ```bash
    nohup python -u clustering-GPU-minutes-HRV.py --pt_ids RCS02 RCS04 RCS06 RCS07 --washout_period W --activity_bin_before_pain_survey B --activity_bin_after_pain_survey A --sleep_binning_direction D --n_jobs J 2> clustering-final-HRV-stderr.txt 1> clustering-final-HRV-stdout.txt &
    ```

    Note: `W`, `B`, `A`, `D`, and `J` are placeholders for the values that the user must supply. Some example values for the placeholders are given below:

    - `W` --> 60
    - `B` --> 30
    - `A` --> 30
    - `D` --> either "forward" or "backward"
    - `J` --> must be an integer and must not exceed the number of CPU cores (-1 to use all available cores)
