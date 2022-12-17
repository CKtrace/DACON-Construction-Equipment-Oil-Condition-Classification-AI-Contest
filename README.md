# 건설기계 오일 상태 분류 AI 경진대회

```
Final Ranking : 21/517 (Top 4.1%)
```


</br>

## Introduction

</br>

__This repository is a place to share "[건설기계 오일 상태 분류 AI 경진대회](https://dacon.io/competitions/official/236013/overview/description)" solution code.__

</br>

```
주최 : 현대제뉴인
후원 : AWS
주관 : 데이콘
```
<br>

## Repository Structure

<br>

```
│  README.md
│  
├─Data_Preprocessing
│      Basic_Preprocessing.ipynb
│      
└─Models
    ├─1st_Model_Set
    │      1st_Student_Model.py
    │      1st_Teacher_Model.py
    │      
    └─2nd_Model_Set
            2nd_Student_Model.py
            2nd_Teacher_Model.py
```

<br>


## Development Environment
</br>

```
CPU : Intel i9-10900F
GPU : NVIDIA GeForce RTX 3080 Ti
RAM : 32GB
```

</br>

## Approach Method Summary
</br>

### Data Preprocessing Perspective

<br>

__Since the amount of missing values in the data and each row is data for the corresponding oil,__ 
__so failure to find out what additional data to put in & lack of knowledge about Construction Equipment oil__

    -> Aim to achieve maximum performance in the modeling part.


<br>

### Limitation

<br>

```
1st Limitation : This competition had to use a "Knowledge Distillation" technique

2nd Limitation : The number of features in the training data is 54, and the number of 
                 features in the data to be predicted is 19, which is a difference of about 3 times.
```


### Modeling Perspective

<br>

```
1. The number of training data is 14,095, and it is thouhgt that it is more appropriate 
   to use a ML model than using a DL model.

2. The "Knowledge Distillation" technique is a technique proposed for use in a DL model 
   -> Decide to building a ML model that applicated "Knowledge Distillation"
```

<br>

### Model Building Method

<br>

```
1. Save a list with each probability for target_value used in the process of classifying
   learning data with Teacher Model

2. Create a function that applies temperature to the sigmoid function proposed in the 
   "Knowledge Distillation" technique paper for the values in the list, reassign the values 
   through this function, and save as a pickle file

3. Load the training data to the Student Model, remove features except features included 
   in test data, assign the target_value as the column corresponding to the normal oil among 
   the pickle files created by the Teacher Model, and train the model

4. Proceed with regression prediction using the data to be predicted with the learned Student Model

5. Proceed with dividing into normal oil and abnormal oil by classifying a threshold 
   for the predicted values

6. Submission
```

<br>


### References

HINTON, G., VINYALS, O., AND DEAN, J. Distilling the knowledge in a neural network. _arXiv preprint arXiv:1503.02531_ (2015).


<br>

<br>

<br>



<div align=right>

_Special Thanks to My Mentor [JongHyeon Kim](https://github.com/bellhyeon)_

</div>
