# CLEAR Challenge | Starter Kit

## Competition Overview
[Continual LEArning on Real-World Imagery (CLEAR)](https://clear-benchmark.github.io/) is the first continual image classification benchmark dataset with a natural temporal evolution of visual concepts in the real world that spans a decade (2004-2014). This competition will be an opportunity for researchers and machine learning enthusiasts to experiment and explore state-of-the-art Continual Learning (CL) algorithms on this novel dataset. Submissions will also be evaluated with novel evaluation protocols that we have proposed for CL. 

### Competition Stages
The challenge consists of two stages:  

## Getting Started
We have provided a sample framework in ```sample_code_submission``` for participants to follow. 

A sample output:
```
Training on Bucket 1 with iid protocol
Itr: 01
Train loss: 2.70         acc : 0.45
...
Evaluate timestamp 1 model on bucket 1
Test Accuracy: 0.80
Evaluate timestamp 1 model on bucket 2
Test Accuracy: 0.78
...
Evaluate timestamp 10 model on bucket 9
Test Accuracy: 0.75
Evaluate timestamp 10 model on bucket 10
Test Accuracy: 0.74
...
Training on Bucket 1 with streaming protocol
Itr: 01
Train loss: 1.14         acc : 0.89
...
Evaluate timestamp 1 model on bucket 2
Test Accuracy: 0.81
...
Metrics:  {'in_domain': 0.766, 'next_domain_iid': 0.743, 'acc': 0.767, 'bwt': 0.765, 'fwt_iid': 0.711, 'next_domain_streaming': 0.771, 'fwt_streaming': 0.744}
Score:  0.752
```
