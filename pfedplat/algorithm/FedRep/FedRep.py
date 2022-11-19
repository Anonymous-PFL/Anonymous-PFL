
import pfedplat as fp
import numpy as np
import torch
import copy
from torch.autograd import Variable
import time


class FedRep(fp.Algorithm):
    def __init__(self,
                 name='FedRep',
                 data_loader=None,
                 model=None,
                 device=None,
                 train_setting=None,
                 client_num=None,
                 client_list=None,
                 online_client_num=None,
                 metric_list=None,
                 max_comm_round=0,
                 max_training_num=0,
                 epochs=1,
                 save_name=None,
                 outFunc=None,
                 write_log=True,
                 params=None,
                 tau=5,  
                 *args,
                 **kwargs):
        if params is not None:
            tau = params['tau']
        if save_name is None:
            save_name = name + ' ' + model.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' tau' + str(tau)
        
        if client_num is None and client_list is not None:
            client_num = len(client_list)
        elif client_num is not None and client_list is None:
            if client_num > data_loader.pool_size:
                client_num = data_loader.pool_size
            client_list = [Client(i, copy.deepcopy(model), device, train_setting, metric_list, tau) for i in range(client_num)]  
            data_loader.allocate(client_list)  
        elif client_num is None and client_list is None:
            raise RuntimeError('Both of client_num and client_list cannot be None or not None.')
        
        super().__init__(name, data_loader, model, device, train_setting, client_num, client_list, online_client_num, metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log)
        self.exist_per_model = True

    def run(self):
        
        batch_num = np.mean(self.send_require_attr('training_batch_num'))
        while not self.terminated(update_count=True):
            com_time_start = time.time()
            
            self.send_train_order(self.epochs)
            
            m_locals, _ = self.send_require_client_per_model()
            com_time_end = time.time()
            cal_time_start = time.time()
            
            self.aggregate(m_locals)
            
            self.current_training_num += self.epochs * batch_num
            
            cal_time_end = time.time()
            self.communication_time += com_time_end - com_time_start
            self.computation_time += cal_time_end - cal_time_start

    def aggregate(self, m_locals):
        w_locals = [m_local.state_dict() for m_local in m_locals]
        averaged_params = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_model_params = w_locals[i]
                if i == 0:
                    averaged_params[k] = local_model_params[k] / self.online_client_num
                else:
                    averaged_params[k] += local_model_params[k] / self.online_client_num
        self.model.load_state_dict(averaged_params)


class Client(fp.Client):
    def __init__(self,
                 id=None,
                 model=None,
                 device=None,
                 train_setting=None,
                 metric_list=None,
                 tau=5,
                 *args,
                 **kwargs):
        super().__init__(id, model, device, train_setting, metric_list, *args, **kwargs)
        self.tau = tau
        
        self.per_model = copy.deepcopy(model)
        self.per_optimizer = train_setting['optimizer'].__class__(filter(lambda p: p.requires_grad, self.per_model.parameters()), lr=self.lr)
        self.per_optimizer.defaults = copy.deepcopy(train_setting['optimizer'].defaults)

    def train(self, epochs):
        
        global_model_params = list(self.model.parameters())
        for idx, p in enumerate(self.per_model.parameters()):
            if idx < len(global_model_params) - 1:
                p.data = global_model_params[idx].data.clone()
        
        self.update_learning_rate(self.per_optimizer, self.lr)
        self.old_per_model = copy.deepcopy(self.per_model)
        
        self.per_model.train()
        
        
        model_params = list(self.per_model.parameters())
        for idx, param in enumerate(model_params):
            param.requires_grad = True if idx == len(model_params) - 1 else False
        for e in range(self.tau):
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)
                
                out = self.per_model(batch_x)
                loss = self.criterion(out, batch_y)
                
                self.per_model.zero_grad()
                loss.backward()
                self.per_optimizer.step()
        
        
        model_params = list(self.per_model.parameters())
        for idx, param in enumerate(model_params):
            param.requires_grad = False if idx == len(model_params) - 1 else True
        for e in range(epochs):
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)
                
                out = self.per_model(batch_x)
                loss = self.criterion(out, batch_y)
                
                self.per_model.zero_grad()
                loss.backward()
                self.per_optimizer.step()
        
        self.model_loss = float(loss)

