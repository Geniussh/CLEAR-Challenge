![image](https://user-images.githubusercontent.com/44150278/160446113-fd603153-79a3-45ad-b72a-508809e6d49f.png)
# CLEAR Challenge | Starter Kit

## Competition Overview
[Continual LEArning on Real-World Imagery (CLEAR)](https://clear-benchmark.github.io/) is the first continual image classification benchmark dataset with a natural temporal evolution of visual concepts in the real world that spans a decade (2004-2014). This competition will be an opportunity for researchers and machine learning enthusiasts to experiment and explore state-of-the-art Continual Learning (CL) algorithms on this novel dataset. In addition, submissions will be evaluated with our novel evaluation protocols that we have proposed for CL in the [paper](https://arxiv.org/abs/2201.06289). 

### Competition Stages
The challenge consists of two stages: 
- **Stage 1**  
Participants train their models locally using the public dataset following the streaming protocol, i.e. train today and test on tomorrow. Participants upload their models (10 models in total, each corresponding to a time bucket) along with their training script as one submission to AICrowd for evaluation against our private holdout set. The evaluation metrics will be the ones used for the streaming protocol, i.e. In-Domain Accuracy, Next-Domain Accuracy, Accuracy, Forward Transfer, Backward Transfer. Details about these metrics can be found in the paper. 

- **Stage 2**  
The top 10 teams on the leaderboard in Stage 1 will be selected and required to provide a dockerized environment to train their models on our own servers. We will validate each team's models submitted to the leaderboard by training their models within the specified time limit, comparing the accuracy with the baselines, as well as checking if they have finetuned any pretrained network. Teams with invalid submissions will be marked on the leaderboard, and only teams with valid submissions will be eligible for the awards.

## Getting Started
1. Download CLEAR either by ```get_data.sh``` in ```/data``` (TODO) or directly from the [website](https://clear-benchmark.github.io/).
2. Fork this starting kit and start developing your contunual supervised learning algorithms.
3. Follow the template in ```sample_code_submission``` as a reference to understand how to evaluate your models in the streaming protocol versus the traditional iid protocol. Note: the template finetunes the last classifier layer in a pretrained SqueezeNet, which is only for education purposes. Participants should never finetune a pretrained network as their submissions. But it is okay to use the performance of a pretrained SqueezeNet or ImageNet as an upper bound as they are trained on ImageNet.
4. Submit the trained models (10 in total) to AICrowd for evaluation using our private holdout set. 
  
A sample output of ```/sample_code_submission/main.py``` is shown as follows. Note: the iid protocol is kept in the sample only to serve as a comparison with the streaming protocol (in terms of different training and evaluation setup). Participants' submissions will only be evaluated in a streaming fashion. 

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

## How to develop your models
Coming soon

## Evaluation
### Accuracy Matrix
* In-Domain Accuracy
* Next-Domain Accuracy
* Accuracy
* Backward Transfer
* Forward Transfer
  
### Training Time Constraint
To prevent the participants from achieving high scores by training their models for weeks, the maximum time that participants should set when training locally should be within 48 hours. We will verify the top 10 teams' training time after Stage 1. 
  
### Ranking Criteria
In both stages, the submissions will be ranked on a weighted average of all of the 5 metrics above given the limited training time. 
