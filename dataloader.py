import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import data as D

import networkx as nx
import numpy as np
import os

class CustomDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.graphs = []
        self.node_features = []  # 存储节点特征的列表
        self.label = []

        # 遍历所有文件夹
        for label_folder in os.listdir(self.root):
            label_path = os.path.join(self.root, label_folder)
            
            # 文件名是标签
            if os.path.isdir(label_path):
                label = int(label_folder)
            
            if (label == 15 ):
                label = 0
            elif (label == 23):
                label = 1

            # 遍历所有csv文件
            for csv_file in os.listdir(label_path):
                # print('file:' + csv_file)
                csv_path = os.path.join(label_path, csv_file)
                edge_weights = self.read_csv(csv_path)

                # 创建无向图对象
                G = nx.Graph()
                # 矩阵大小
                n = len(edge_weights)
                # 添加节点
                G.add_nodes_from(range(n))
                # 添加边
                for i in range(n):
                    for j in range(i+1, n):
                        if(edge_weights[i][j] != 0):
                            G.add_edge(i, j, weight=edge_weights[i][j])

                # 提取节点特征并存储到列表中
                # node_feature = list(nx.get_node_attributes(G, 'feature').values())
                # self.node_features.append(node_feature)

                self.graphs.append((G, label))
                self.label.append(label)
            # print(self.graphs)
        
        # 将节点特征列表转换为 PyTorch 张量
        self.x = torch.tensor(self.node_features)

        # 获取边的索引和属性
        edge_index_list = []
        edge_attr_list = []
        for G, _ in self.graphs:
            edge_index = torch.tensor(list(G.edges)).t().contiguous()
            edge_attr = torch.tensor([G[u][v]['weight'] for u, v in G.edges])
            edge_index_list.append(edge_index)
            edge_attr_list.append(edge_attr)

        # 将边的索引和属性列表转换为 PyTorch 张量
        self.edge_index = edge_index_list
        self.edge_attr = edge_attr_list
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
    
        return Data(x=self.x[idx], edge_index=self.edge_index[idx], edge_attr=self.edge_attr[idx], y=self.label[idx])
        
    def read_csv(self, csv_path):
        # 从CSV文件中读取边的权重数据
        df = np.loadtxt(csv_path, delimiter=',')
        return df


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.processed_data = []

        # 初始化节点
        data = torch.arange(0, 30)
        self.x = data.view(-1, 1).float()

        # 遍历文件夹
        for label_folder in os.listdir(self.root):
            # print(label_folder)
            label_path = os.path.join(self.root, label_folder)
            if os.path.isdir(label_path):
                label = int(label_folder)

                if (label == 15):
                    label = 0
                elif (label == 23):
                    label = 1

                # 遍历文件夹中的CSV文件
                for csv_file in os.listdir(label_path):
                    # print(csv_file)
                    if csv_file.endswith('.csv'):
                        csv_path = os.path.join(label_path, csv_file)
                        edge_weights = self.read_csv(csv_path)

                        self.processed_data.append((edge_weights, label))
        
        # print('test')

        length = len(self.processed_data)

        # 初始化边和边权重和标签
        self.edge_index = torch.zeros(2, 435)
        self.edge_attr = torch.zeros(length, 435, 1)
        self.label = torch.zeros(length, 1)

        num = 0
        for i in range(1, 30):
            for j in range(i+1, 30):
                self.edge_index[0][num] = i
                self.edge_index[1][num] = j
                num = num + 1
        

        for idx in range(len(self.processed_data)):
            # print(idx)
            edge_weights, label = self.processed_data[idx]
            self.label[idx][0] = label

            n = len(edge_weights)
            num = 0
            for i in range(1, n):
                for j in range(i+1, n):
                    self.edge_attr[idx][num][0] = edge_weights[i][j]
                    num = num + 1

        

    def __len__(self):
        return len(self.processed_data)
   
    def __getitem__(self, idx):
       
        data=D.Data()

        data.x = self.x
        data.label = self.label[idx]
        data.edge_index = self.edge_index
        data.edge_attr = self.edge_attr[idx]

        return data

    # def __getitem__(self, idx):

    #     edge_weights, label = self.processed_data[idx]

    #     # 初始化边和边权重
    #     edge_index = torch.zeros(2, 435)
    #     edge_attr = torch.zeros(435,1)

    #     n = len(edge_weights)
    #     for num in range(435):
    #         for i in range(1, n):
    #             for j in range(i+1, n):
    #                 edge_attr[num] = edge_weights[i][j]
    #                 edge_index[0][num] = i
    #                 edge_index[1][num] = j
            
    #     data=D.Data()

    #     data.x = self.x
    #     data.label = torch.tensor([label])
    #     data.edge_index = edge_index
    #     data.edge_attr = edge_attr

    #     return data


    def read_csv(self, csv_path):
        # 从CSV文件中读取边的权重数据
        df = np.loadtxt(csv_path, delimiter=',')
        return df