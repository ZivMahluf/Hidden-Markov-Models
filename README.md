This is my answer to the second exercise in Advanced Practical Machine Learning course.

My implementation includes Hidden Markov related models, such as Baseline, HMM and MEMM.

The objective of the models is POS tagging.

A sampling function is added to HMM, representing the distribution P(x|y).

The accuracy is as follows:
10% train, 10% test:
Baseline - 89%
HMM - 91%
MEMM - 84.65%
MEMM with extra features - 85%

25% train, 10% test:
Baseline - 91%
HMM - 94%
MEMM - 87%
MEMM with extra features - 87%

90% train, 10% test - 
Baseline - 92%
HMM - 95%
