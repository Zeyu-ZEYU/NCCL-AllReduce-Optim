# original_allreduce.py

Two nodes. Each node has four workers. All eight workers do global allreduce. Fine-tune a BERT model for sentence classification. Output the training time for 1000 iterations.

## How to run
```python
# In python code, change the variable below:
GROUP_MASTER_IP = "172.31.92.228"  # process group master's IP
DATA_PATH = "./bbc-text.csv"  # no need to modify
```

```bash
# on node 0
python3 original_allreduce.py
# on node 1
python3 original_allreduce.py --node=1
```


# allreduce_optim_v1.py

Two nodes. Each node has four workers. Workers within a node do allreduce locally. Then worker0 (rank=0) on node 0 and worker4 (rank=4) on node 1 do allreduce. Finally, worker0 and worker4 broadcast the gradients. Only one NIC gets involved on each node.

## How to run
```python
# In python code, change the variable below:
GROUP_MASTER_IP = "172.31.92.228"  # process group master's IP
DATA_PATH = "./bbc-text.csv"  # no need to modify
```

```bash
# on node 0
python3 allreduce_optim_v1.py
# on node 1
python3 allreduce_optim_v1.py --node=1
```


# allreduce_optim_v2.py

Two nodes. Each node has four workers. Workers within a node do allreduce locally. Then worker0 and worker4 do allreduce by using one pair of NICs; worker1 and worker5 do allreduce by using another pair of NICs. The same for worker2-worker6 and worker3-worker7.

## How to run
```python
# In python code, change the variable below:

# process group master's IP
GROUP_MASTER_IP = "172.31.92.228"
# the network interface each rank uses (in rank order)
NETIF = ["ens1", "ens2", "ens3", "ens4", "ens1", "ens2", "ens3", "ens4"]
# no need to modify
DATA_PATH = "./bbc-text.csv"
```

```bash
# on node 0
python3 allreduce_optim_v2.py
# on node 1
python3 allreduce_optim_v2.py --node=1
```


# allreduce_optim_v3.py

Two nodes. Each node has four workers. Workers within a node do allreduce locally. Parameters are divided into four groups (**pgroup**). Then worker0 and worker4 do allreduce by using one pair of NICs for pgroup0's gradients; worker1 and worker5 do allreduce by using another pair of NICs for pgroup1's gradients. The same for worker2-worker6 (for pgroup2) and worker3-worker7 (for pgroup3). Then four workers within a node broadcast their corresponding pgroup's gradients to other co-located workers in the same node.

## How to run
```python
# In python code, change the variable below:

# process group master's IP
GROUP_MASTER_IP = "172.31.92.228"
# the network interface each rank uses (in rank order)
NETIF = ["ens1", "ens2", "ens3", "ens4", "ens1", "ens2", "ens3", "ens4"]
# no need to modify
DATA_PATH = "./bbc-text.csv"
```

```bash
# on node 0
python3 allreduce_optim_v3.py
# on node 1
python3 allreduce_optim_v3.py --node=1
```
