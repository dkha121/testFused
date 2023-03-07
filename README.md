# State Prediction
This repository is to predict the state of user turn in the dialogue

## Install
```shell
pip install -r requirements.txt
```

## Convert dataset 
```shell
python datareader/data_converter.py --ketod_dataset <path_of_ketod> --ketod_sample <out_path_of_ketod>
```
```commandline
usage: Dataset Converter [-h] [--ketod_dataset KETOD_DATASET]
                         [--ketod_sample KETOD_SAMPLE]

optional arguments:
  -h, --help            show this help message and exit
  --ketod_dataset KETOD_DATASET
                        the path of ketod dataset
  --ketod_sample KETOD_SAMPLE
                        the path of ketod sample (out file)
```

## Training
```shell
bash tune-idb0.sh
```
* **Please update your parameters in the tune-idb0.sh file**
