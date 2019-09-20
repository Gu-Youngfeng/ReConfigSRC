# ReConfigSRC

### 1. Introduction

ReConfigSRC implements the experiments of ReConfig approach.


### 2. Environment & Dependency

ReConfigSRC, which is designed by Python language, can be run on Windows or Linux operation system.
The dependencies include three widely-used Python libraries, i.e., **numpy**, **pandas**, and **sklearn**.


### 3. Experimental Steps

Step 1: initialize the raw dataset

Put the raw datasets (.csv format) into the folder `raw_data/`, note that the last column (performance) of each dataset should start with "$<". 
For example, the dataset of _Noc-obj1.csv_ is as follows,


|width| complexity| fifo| multiplier| $<energy|
|:--|:--|:--|:--|:--|
| 3.0| 1| 4.0| 1| 7.8351029794899985| 
| 3.0| 1| 1.0| 1| 7.836833049419999| 
| 3.0| 1| 2.0| 100| 9.965784284660002| 
| ...|...| ...| ...| ...|


Step 2: get the results of the rank-based approach

Run the rank-based approach (i.e, `src/rank_based.py`) and obtain the prediction results, which are outputted into the folder `experiment/rank_based/`.

```
>> python src/rank-based.py
```

Step 3: get the results of the other approaches

Run the other approaches (i.e., `src/outlier_detection.py`, `src/classfication.py`, `src/random_rank.py`, `src/reconfig.py`, and `src/direct_ltr.py`) 
and obtain the corresponding ranking results, which are outputted into the folder `experiment/${approach_name}`.
Note that these five approaches are based on the prediction results of the rank-based approach (outputs of Step 2).

```
>> python src/classfication.py
>> python src/outlier_detection.py
>> python src/random_rank.py
>> python src/direct_ltr.py
>> python src/reconfig.py
```

Step 4: analyze the ranking results

Run the experiments with given commands.

```
>> python src/experiment.py ${command}
```

The following table shows the given commands in `src/experiment.py`

| Command | Description |
|:--|:--|
| projInfo | Showing the basic information (e.g., options and dataset size) in each dataset.|
| projDistr {$index} | Drawing the performance distribution of specific dataset.|
| tiedNums | Drawing the number of tied configuretions in each datasets using the rank-based method.|
| calRDTie | Calculating the RDTie of each approach using different methods.|
| vsRankBased | RQ-1: Can ReConfig find better configurations than the rank-based approach?|
| vsOthers | RQ-2: Can the learning-to-rank method in ReConfig outperform comparative methods in finding configurations?|
| removeRatio | RQ-3: How many tied configurations should be filtered out in ReConfig?|
| vsRD | RQ-4: Is RDTie stable for evaluating the tied prediction?| 



