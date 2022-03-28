'''
    This script defines the metrics used to evaluate a submission.
    The metrics are consistent with the ones used in the evaluation 
    table of streaming and iid protocols, i.e., Table 1, in the paper. 
    More information the two protocols and metrics:
    https://clear-benchmark.github.io/
'''

# Average of lower triangle + diagonal (evaluate accuracy on seen tasks)
def accuracy(M):
    r, _ = M.shape
    res = [M[i, j] for i in range(r) for j in range(i+1)]
    return sum(res) / len(res)

# Diagonal average (evaluates accuracy on the current task)
def in_domain(M):
    r, _ = M.shape
    return sum([M[i, i] for i in range(r)]) / r

# Superdiagonal average (evaluate on the immediate next time period)
def next_domain(M):
    r, _ = M.shape
    return sum([M[i, i+1] for i in range(r-1)]) / (r-1)

# Upper trianglar average (evaluate generalation)
def forward_transfer(M):
    r, _ = M.shape
    res = [M[i, j] for i in range(r) for j in range(i+1, r)]
    return sum(res) / len(res)

# Lower triangular average (evaluate learning without forgetting)
def backward_transfer(M):
    r, _ = M.shape
    res = [M[i, j] for i in range(r) for j in range(i)]
    return sum(res) / len(res)