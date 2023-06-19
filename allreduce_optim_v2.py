#! /usr/bin/env python3


import argparse
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as tmp

# from tqdm import tqdm
from transformers import BertModel, BertTokenizer

rand_seed = 218276150
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)
np.random.seed(rand_seed)
random.seed(rand_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


GROUP_MASTER_IP = "172.31.92.228"
# the net interface each rank uses (in rank order)
NETIF = ["ens1", "ens2", "ens3", "ens4", "ens1", "ens2", "ens3", "ens4"]
DATA_PATH = "./bbc-text.csv"


class LoggerConstructor:
    def __init__(self, logger_name, file_name, log_level=logging.INFO, mode="w"):
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        self.__fh = logging.FileHandler(filename=file_name, mode=mode)
        self.__formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        self.__fh.setFormatter(self.__formatter)
        self.__logger.addHandler(self.__fh)

    def get_logger(self):
        return self.__logger


class BertClassifier(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(768, 5)
        self.relu = torch.nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


class BBCTextDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        labels = {"business": 0, "entertainment": 1, "sport": 2, "tech": 3, "politics": 4}
        self.labels = [labels[label] for label in df["category"]]
        self.texts = [
            tokenizer(text, padding="max_length", max_length=512, truncation=True, return_tensors="pt") for text in df["text"]
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


# skip it. it never runs.
def netif_monitor(name):
    pass
    # logger = LoggerConstructor("logger", f"./{name}_netif.log").get_logger()

    # listener = znet.SocketMsger.tcp_listener("0.0.0.0", 63321)
    # connm, _ = listener.accept()
    # connm.recv()

    # recv0 = psutil.net_io_counters(pernic=True)["ens3"].bytes_recv
    # sent0 = psutil.net_io_counters(pernic=True)["ens3"].bytes_sent
    # time0 = time.time()
    # while True:
    #     time.sleep(0.01)
    #     recv1 = psutil.net_io_counters(pernic=True)["ens3"].bytes_recv
    #     sent1 = psutil.net_io_counters(pernic=True)["ens3"].bytes_sent
    #     time1 = time.time()
    #     time_diff = time1 - time0
    #     bw_in = (recv1 - recv0) / time_diff / 1048576
    #     bw_out = (sent1 - sent0) / time_diff / 1048576
    #     logger.info(f"{bw_in} {bw_out}")
    #     recv0 = recv1
    #     sent0 = sent1
    #     time0 = time1


def worker(margs):
    # logger = LoggerConstructor("logger", f"./wrk{margs['rank']}.log").get_logger()

    os.environ["MASTER_ADDR"] = GROUP_MASTER_IP
    os.environ["MASTER_PORT"] = "61234"
    os.environ["WORLD_SIZE"] = "8"
    # os.environ["NCCL_SOCKET_IFNAME"] = "ens3"
    rank = margs["rank"]
    os.environ["RANK"] = str(rank)
    gpu = margs["gpu"]

    os.environ["NCCL_SOCKET_IFNAME"] = NETIF[rank]

    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(gpu)

    model = BertClassifier()
    print(f"rank {rank} model done")
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = np.split(
        df.sample(frac=1, random_state=42),
        [
            int(0.8 * len(df)),
        ],
    )
    train_ds, test_ds = BBCTextDataset(train_df), BBCTextDataset(test_df)
    print(f"rank {rank} dataset done")
    data_loader_list = []
    train_ds_len = len(train_ds)
    wrk_ds_len = int((train_ds_len + 8) / 8)
    len_list = [wrk_ds_len for _ in range(8 - 1)]
    len_list.append(train_ds_len - (8 - 1) * wrk_ds_len)
    train_ds_list = torch.utils.data.random_split(train_ds, len_list)
    for idx in range(8):
        data_loader = torch.utils.data.DataLoader(train_ds_list[idx], batch_size=2, shuffle=True, drop_last=True)
        data_loader_list.append(data_loader)
    print(f"rank {rank} data_loader_list done")
    data_loader = data_loader_list[rank]
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=2, drop_last=True)

    device = torch.device(f"cuda:{gpu}")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    # make sure all workers start training at the same time.
    torch.cuda.synchronize()
    torch.distributed.barrier(torch.distributed.new_group(backend="nccl"))

    local_group_0 = torch.distributed.new_group([0, 1, 2, 3], backend="nccl")
    local_group_1 = torch.distributed.new_group([4, 5, 6, 7], backend="nccl")
    group_across_node_0 = torch.distributed.new_group([0, 4], backend="nccl")
    group_across_node_1 = torch.distributed.new_group([1, 5], backend="nccl")
    group_across_node_2 = torch.distributed.new_group([2, 6], backend="nccl")
    group_across_node_3 = torch.distributed.new_group([3, 7], backend="nccl")
    if rank < 4:
        local_group = local_group_0
    else:
        local_group = local_group_1
    if rank == 0 or rank == 4:
        group_across_nodes = group_across_node_0
    elif rank == 1 or rank == 5:
        group_across_nodes = group_across_node_1
    elif rank == 2 or rank == 6:
        group_across_nodes = group_across_node_2
    elif rank == 3 or rank == 7:
        group_across_nodes = group_across_node_3

    num_iter = 0
    time_start = time.time()
    while True:
        for train_input, train_label in data_loader:
            num_iter += 1

            train_label = train_label.to(device)
            mask = train_input["attention_mask"].squeeze().to(device)
            input_ids = train_input["input_ids"].squeeze().to(device)

            for param in model.parameters():
                param.grad = None
            output = model(input_ids, mask)

            batch_loss = criterion(output, train_label.long())

            batch_loss.backward()

            async_handlers = []
            for _, param in model.named_parameters():
                ah = torch.distributed.all_reduce(
                    param.grad, op=torch.distributed.ReduceOp.AVG, group=local_group, async_op=True
                )
                async_handlers.append(ah)
            for ah in async_handlers:
                ah.wait()

            async_handlers = []
            for _, param in model.named_parameters():
                ah = torch.distributed.all_reduce(
                    param.grad, op=torch.distributed.ReduceOp.AVG, group=group_across_nodes, async_op=True
                )
                async_handlers.append(ah)
            for ah in async_handlers:
                ah.wait()

            optimizer.step()

            if num_iter % 50 == 0:
                print(f"wrk {rank}: {num_iter} iters done. {1000-num_iter} iters left.")
            if num_iter == 1000:
                break
        if num_iter == 1000:
            break
    time_intv = time.time() - time_start
    print(f"wrk {rank}: training time {time_intv}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--node", type=int, default=0)

    args = parser.parse_args()

    margs = {}
    processes = []
    if args.node == 0:
        for rank in range(4):
            margs["rank"] = rank
            margs["gpu"] = rank % 4
            p = tmp.Process(target=worker, args=(margs,))
            p.start()
            processes.append(p)
        # p = tmp.Process(target=netif_monitor, args=("node0",))
        # p.start()
        # processes.append(p)
    else:
        for rank in range(4, 8):
            margs["rank"] = rank
            margs["gpu"] = rank % 4
            p = tmp.Process(target=worker, args=(margs,))
            p.start()
            processes.append(p)
        # p = tmp.Process(target=netif_monitor, args=("node1",))
        # p.start()
        # processes.append(p)

    for p in processes:
        p.join()
