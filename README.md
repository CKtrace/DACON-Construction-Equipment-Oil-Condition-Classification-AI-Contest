# 건설기계 오일 상태 분류 AI 경진대회

```
Final Ranking : 22/518 (Top 4.2%)
```

</br>

## Introduction

</br>

__This repository is a place to share "건설기계 오일 상태 분류 AI 경진대회" solution code.__
</br>

```
주최 : 현대제뉴인
후원 : AWS
주관 : 데이콘
```
</br>


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

__Since the amount of missing values in the data and each row is data for the  corresponding oil,__ 
__so failure to find out what additional data to put in & lack of knowledge about Construction Equipment oil__

    -> Aim to achieve maximum performance in the modeling part.


<br>

### Limitation

<br>

```
1st Limitations: This competition had to use a "Knowledge Distillation" technique

2nd Limitations : The number of features in the training data is 54, and the number of 
                  features in the data to be predicted is 19, which is a difference of about 3 times.
```


### Modeling Perspective

<br>

```
1. The number of training data is 14,095, and it is thouhgt that it is more appropriate 
   to use a ML model than to learn using a DL model.

2. The "Knowledge Distillation" technique is a technique proposed for use in a DL model 
   -> Decide to building a ML model that applicated Knowledge Distillation
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

3. Import the training data from the Student Model, remove features except features included 
   in test data, assign the target_value as the column corresponding to the normal oil among 
   the pickle files created by the Teacher Model, and train the model

4. Proceed with regression prediction using the data to be predicted with the learned Student Model

5. Proceed with dividing into normal oil and abnormal oil by classifying a threshold 
   for the predicted values

6. Submission
```

<br>

<br>

<br>

_Special Thanks to my mentor [JongHyeon Kim](https://github.com/bellhyeon)_
