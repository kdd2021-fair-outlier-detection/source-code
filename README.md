# Deep Clustering for Fair Outlier Detection
Source code for kdd2021 submission

## Software requirement
`environment.txt` in `code` folder lists the versions for necessary packages.

Model training require at least one GPU.

## Datasets
We compare performance on eight datasets [student](https://archive.ics.uci.edu/ml/datasets/student%2Bperformance), [asd](https://archive.ics.uci.edu/ml/datasets/Autism+Screening+Adult), [obesity](https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+), [cc](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients), [german](http://archive.ics.uci.edu/ml/datasets/South+German+Credit+%28UPDATE%29), [drug](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29), [adult](https://archive.ics.uci.edu/ml/datasets/adult), [kdd](https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29).

To prepare datasets before model training, run
```
python3 getDatasets.py
```

## DCFOD 
To obtain DCFOD's performance on a speicifc dataset, run
```
python3 train.py *dataset_name* *GPU_index* *with_weight*

i.e., 

python3 train.py student 0 true
```
for `GPU_index`, if there's only one GPU, simply type `0`, if there are more than one, and you want to train on the i-th GPU, the index shall be i-1. 

### Competitive method: FairLOF
FairLOF requires the baseline result of LOF, you should first run
```
python3 LOF.py *dataset_name*
```
followed by
```
python3 get_Ws_for_FairLOF.py
```
which will retrieve all the `Ws` variable based on LOF results for all datasets, which is requried in FairLOF calculation.

If you don't have the LOF results for all datasets, tweak the `datasets` list in the `get_Ws_for_FairLOF.py` file.

then run
```
python3 FairLOF.py *dataset_name*
```
the experiment run on 4 GPUs, you can tweak line 21-25 in `FairLOF.py` to modify cuda and the associated GPUs.

### Competitive method: FairOD
We run FairOD with
```
python3 FairOD.py *dataset_name* *GPU_index* *fair_command*
```
you should first obtain a baseline result, i.e., 
```
python3 FairOD.py student 0 f
```
then train the fair model
```
python3 FairOD.py student 0 t 
```
### Conventional outlier detection methods
To save method's outlier scores and obtain metrics' values on a specific dataset, run
```
python3 pyod_results.py *dataset_name*
```
Or, if you already ran the above command, which has saved outlier scores, and you want to re-obtain the metrics' values, simply run
```
python3 Retriever.py *dataset_name*
```
