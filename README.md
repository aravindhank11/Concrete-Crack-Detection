# Crack Detection in Buildings
Buildings tend to expand or contract based on the environment they are in and that leads to cracks on to the buildings and they can be a serious threat to the people using it and more often than not these movements are too small to be observed and so often go unnoticed. Movement can be caused by defects, movement of the ground, foundation failure, decay of the building fabric, and so on. If a structure is unable to accommodate this movement, cracking is likely to occur and its highly dangerous to the safety of the building.

Only upon the identification of the cracks, it can be subjected to treatment and the existing manual methods of sketching the crack patterns are much subjective to the person analysing them, and are often bounded by high costs, equipment and tools availability and is highly time consuming. In this work, we provide with a comparative study of various Deep Neural Networks to classify the image as one with the presence or absence of cracks and thereby suggesting a state of art binary classifying Neural Network that best suits the goal of Crack Detection.

## Dataset
Download the dataset from https://data.mendeley.com/datasets/5y9wdsg2zt/2 and rename the folder as `dataset`

## Setup the environment on all the cluster nodes with GPU
```
# HOROVOD_WITH_TENSORFLOW=1  pip3 install -v --no-cache-dir horovod
# pip3 install -r requirements.txt
```

## About Horovod
Horovod core principles are based on MPI concepts such as size, rank, local rank, allreduce, allgather and broadcast. These are best explained by example. Say we launched a training script on 4 servers, each having 4 GPUs. If we launched one copy of the script per GPU:
1) Size would be the number of processes, say 16.
2) Rank would be the unique process ID from 0 to 15 (size - 1).
3) Local rank would be the unique process ID within the server from 0 to 3.
4) Allreduce is an operation that aggregates data among multiple processes and distributes results back to them. Allreduce is used to average dense tensors.
5) Ref: https://github.com/horovod/horovod/tree/v0.9.0

## Results
We were able to predict cracks in the building upto an accuracy of 95% with CNN model. For more insights please refer to the attached AIPAPER.pdf
