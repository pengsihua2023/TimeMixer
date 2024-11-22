```
from torch.optim import lr_scheduler

# 导入数据提供器
from data_provider.data_factory import data_provider
# 导入实验基础类
from exp.exp_basic import Exp_Basic
# 导入工具类：早停、学习率调整、计算准确率
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb

# 忽略警告信息
warnings.filterwarnings('ignore')

# 定义分类实验类，继承自基础实验类 Exp_Basic
class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        # 调用父类的构造函数初始化
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # 构建模型的方法
        
        # 获取训练数据和数据加载器
        train_data, train_loader = self._get_data(flag='TRAIN')
        # 获取测试数据和数据加载器
        test_data, test_loader = self._get_data(flag='TEST')
        
        # 动态调整模型的输入和输出参数
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)  # 设置最大序列长度
        self.args.pred_len = 0  # 预测长度设置为0
        self.args.enc_in = train_data.feature_df.shape[1]  # 输入特征数
        self.args.num_class = len(train_data.class_names)  # 类别数量
        
        # 根据配置初始化模型
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        # 如果使用多GPU训练，则启用数据并行
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model

    def _get_data(self, flag):
        # 获取数据集和数据加载器的方法
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # 选择优化器，使用 RAdam 优化器
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # 选择损失函数，使用交叉熵损失
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        # 验证方法，用于评估模型在验证集上的表现
        total_loss = []  # 存储每批次的损失
        preds = []  # 存储预测值
        trues = []  # 存储真实标签
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 禁用梯度计算以加快推理速度并节省内存
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                # 将数据移动到设备（如GPU）
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # 前向传播获取模型输出
                outputs = self.model(batch_x, padding_mask, None, None)

                # 计算损失
                pred = outputs.detach()
                loss = criterion(pred, label.long().squeeze())
                total_loss.append(loss.item())  # 将损失存储到列表中

                preds.append(outputs.detach())  # 保存预测值
                trues.append(label)  # 保存真实标签

        total_loss = np.average(total_loss)  # 计算平均损失

        preds = torch.cat(preds, 0)  # 拼接所有预测结果
        trues = torch.cat(trues, 0)  # 拼接所有真实标签
        probs = torch.nn.functional.softmax(preds)  # 将预测值转换为概率分布
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # 获取每个样本的预测类别
        trues = trues.flatten().cpu().numpy()  # 将真实标签转换为numpy数组
        accuracy = cal_accuracy(predictions, trues)  # 计算准确率

        # 重新设置模型为训练模式
        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        # 训练模型的方法
        
        # 获取训练、验证和测试数据
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        # 创建用于保存模型检查点的目录
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()  # 记录当前时间

        train_steps = len(train_loader)  # 获取训练数据的批次数
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)  # 初始化早停机制

        model_optim = self._select_optimizer()  # 获取优化器
        criterion = self._select_criterion()  # 获取损失函数

        # 设置学习率调度器
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):  # 训练多个epoch
            iter_count = 0
            train_loss = []  # 记录训练损失
            
            # 设置模型为训练模式
            self.model.train()
            epoch_time = time.time()  # 记录当前epoch的时间

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()  # 梯度清零

                # 数据转移到设备
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # 前向传播
                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))  # 计算损失
                train_loss.append(loss.item())  # 保存当前批次的损失

                # 每隔100步输出一次训练信息
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()  # 反向传播
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)  # 梯度裁剪
                model_optim.step()  # 更新参数

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)  # 计算平均训练损失
            
            # 验证集评估
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            
            # 检查早停条件
            early_stopping(-test_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)

        # 加载最佳模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        # 测试模型的方法
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []  # 预测值列表
        trues = []  # 真实标签列表
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 设置模型为评估模式
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                # 数据转移到设备
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # 获取模型输出
                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())  # 保存预测值
                trues.append(label)  # 保存真实标签

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # 转换为概率分布
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # 获取预测类别
        trues = trues.flatten().cpu().numpy()  # 转换为numpy格式
        accuracy = cal_accuracy(predictions, trues)  # 计算准确率

        # 保存结果
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        file_name = 'result_classification.txt'
        f = open(os.path.join(folder_path, file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return


```
