# Run syne-tune on local

Based on [Run distributed hyperparameter and neural architecture tuning jobs with Syne Tune](https://aws.amazon.com/jp/blogs/machine-learning/run-distributed-hyperparameter-and-neural-architecture-tuning-jobs-with-syne-tune/).

Since it takes too much time locally, it reduces the data without guaranteeing uniformity for each class.
This script can only confirm that it can run.

```python
# Reduce data for local test
n_dataset = 1000
_, trainset = torch.utils.data.random_split(trainset, [len(trainset) - n_dataset, n_dataset])
```

## requirements
- pytorch
- autograd
- scipy
