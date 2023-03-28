# State Prediction
This repository is to predict the state of user turn in the dialogue

## Install
```shell
pip install -r requirements.txt
```

## Convert dataset 
```shell
python datareader/data_converter.py --ketod_dataset <path_of_ketod> --ketod_sample <out_path_of_ketod>


python datareader/data_converter.py --woi_dataset <path_of_woi> --woi_sample <out_path_of_woi>


python datareader/data_converter.py --fusedchat_dataset <path_of_fused> --fused_sample <out_path_of_fused> --schema_guided <path_of_schema>

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

## Dowload link

Dataset after converted: https://drive.google.com/drive/folders/1QWqcj9vRSWrW77DJhUxndpFQFlrX2kAp



## Training
```shell
bash run_training.sh
```
* **Please update your parameters in the run_training.sh file**
