## ReConfigSRC

### 1. Introduction

Project _ReConfigSRC_ implements the experiments of ReConfig approach.


### 2. Environment & Dependency

ReConfigSRC is designed by Python language, so make sure that there is a **Python environments** on your computer.
Besides, three widely-used Python libraries (**numpy**, **pandas**, and **sklearn**) are also required.


### 3. Experimental Steps

**Step 1:** prepare the raw datasets

The input datasets (in ".csv" format) of ReConfigSRC should be saved in the folder `raw_data/`. Note that the instances in each input dataset consist of a set of options and a performance. 

Here are 3 example instances in dataset "Noc-obj1.csv", each instace has 4 options (width, complexity, fifo, multiplier) and a performance ($<energy).

|width| complexity| fifo| multiplier| $<energy|
|:--|:--|:--|:--|:--|
| 3.0| 1| 4.0| 1| 7.8351029794899985| 
| 3.0| 1| 1.0| 1| 7.836833049419999| 
| 3.0| 1| 2.0| 100| 9.965784284660002| 
| ...|...| ...| ...| ...|


**Step 2:** obtain the results of the rank-based approach

Run the rank-based approach (i.e, `src/rank_based.py`) and obtain the preliminary prediction results, which are outputted into the folder `experiment/rank_based/`. Note that `src/rank_based.py` must be executed at first.

```python
>> python src/rank-based.py
```

**Step 3:** obtain the results of the other approaches

Run the other approaches (`src/classfication_exd.py`, `src/random_rank.py`, `src/reconfig.py`, etc.) 
and obtain the corresponding ranking results.
The prediction results are outputted in the folder `experiment/${approach_name}`.

```python
>> python src/classfication_exd.py
>> python src/random_rank.py
>> python src/reconfig.py
>> ...
```

**Step 4:** analyze the ranking results

Run the `src/experiment.py` with command to analyze the results of each approach (in folder `experiment/results/`).

```python
>> python src/experiment.py calRDTie
```

The other commands of `src/experiment.py` are as follows,

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

> Note: The newly-submited `src/execute.py` is another user interface of step-2 to step-4, that is, you can only run the `src/execute.py` at once instead of running python files (step-2 to step-4) step by step. 



