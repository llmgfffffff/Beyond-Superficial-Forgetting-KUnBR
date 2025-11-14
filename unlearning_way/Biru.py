from unlearn import *

class DEEP_UNL:
    def __init__(self,base_model:Base_model=None,block_num:int=8,choose_num:int=1,compare_model_path:str="/root/autodl-tmp/Meta-Llama-3-8B-Instruct",compare_train_times:int=1):
        self.base_model = base_model
        self.block_num = block_num
        self.choose_num = choose_num
        self.block_size  = 32//block_num
        self.model = base_model.model
        self.compare_model = base_model.load_llm(compare_model_path)
        self.lr = self.base_model.lr
        self.warmup_steps = self.base_model.warmup_steps
        self.compare_train_times = compare_train_times
        self.args_init()
    def args_init(self):

        self.history_cache = dict()
        for i in range(self.block_num):
            self.history_cache[f"block{i}"] = 0


        self.bidirectional_mapping_table = dict()
        for name,param in self.model.named_parameters():
            number = re.search(r'\d+', name)
            if number is None:
                continue
            else:
                number = int(number.group())
            block_i = number//self.block_size
            if f"block{block_i}" not in self.bidirectional_mapping_table:
                self.bidirectional_mapping_table[f"block{block_i}"] = []
            self.bidirectional_mapping_table[f"block{block_i}"].append(name)
            self.bidirectional_mapping_table[name] = f"block{block_i}"
            
    def exchang_block(self):

        def select_top_k_params(param_dict, k,reverse=False):
            
            if not isinstance(param_dict, dict) or not isinstance(k, int) or k <= 0:
                raise ValueError("参数不合法，param_dict必须是字典，k必须是正整数")

            sorted_params = sorted(param_dict.items(), key=lambda x: x[1],reverse=reverse)

            top_k_params = [param[0] for param in sorted_params[:k]]

            return top_k_params
        def select_half_keys(input_dict):

            if not isinstance(input_dict, dict):
                raise ValueError("输入必须是一个字典")
            
            keys = list(input_dict.keys())
            
            num_keys_to_select = self.choose_num
            
            selected_keys = random.sample(keys, num_keys_to_select)
            
            return selected_keys
        
        def select_keys_by_probability(ratio_dict,num_keys = self.choose_num):
            
            keys = list(ratio_dict.keys())
            probabilities = list(ratio_dict.values())

            selected_keys = random.choices(keys, weights=probabilities, k=num_keys)

            return selected_keys
        
        def filter_zero_values_extended(input_dict):

            outputlst = [key for key, value in input_dict.items() if value<self.compare_train_times]
            if len(outputlst)==0:
                return [key for key, value in input_dict.items() if value]
            return outputlst[0:self.choose_num]

        exchange_block = filter_zero_values_extended(self.history_cache)
        for i in exchange_block:
            self.history_cache[i]+=1
        
        print(f"-----exchange_block--is---{exchange_block}-----------")
        self.choose_exchange_param = []
        choose_result_block = [self.bidirectional_mapping_table[i] for i in exchange_block]
        for i in choose_result_block:
            self.choose_exchange_param+=i
        
        params1 = {name: param for name, param in self.model.named_parameters()}
        params2 = {name: param for name, param in self.compare_model.named_parameters()}
        
        for name in self.choose_exchange_param:
            
            if name in params1 and name in params2:
                param1 = params1[name]
                param2 = params2[name]
                
                if param1.shape != param2.shape:
                    raise ValueError(f"参数 {name} 的形状不匹配，"
                                        f"model1: {param1.shape}, model2: {param2.shape}")
                
                temp = param1.clone()
                param1.copy_(param2)
                param2.copy_(temp)
                
            else:
                raise KeyError(f"一个或两个模型中没有参数 {name}")
        
        self.optimizer = Lion(self.model.parameters(), lr=self.lr, use_triton=True)
        self.compare_optimizer = Lion(self.compare_model.parameters(), lr=self.lr, use_triton=True)

        for name,param in self.compare_model.named_parameters():
            if name in self.choose_exchange_param:
                param.requires_grad = True
                
            else:
                param.requires_grad = False

    def recover_exchange_block(self):
        params1 = {name: param for name, param in self.model.named_parameters()}
        params2 = {name: param for name, param in self.compare_model.named_parameters()}
        print("----------------恢复参数---------------")
        
        for name in self.choose_exchange_param:
            
            if name in params1 and name in params2:
                param1 = params1[name]
                param2 = params2[name]
                
                if param1.shape != param2.shape:
                    raise ValueError(f"参数 {name} 的形状不匹配，"
                                        f"model1: {param1.shape}, model2: {param2.shape}")
                
                temp = param1.clone()
                param1.copy_(param2)
                param2.copy_(temp)
                
            else:
                raise KeyError(f"一个或两个模型中没有参数 {name}")
        
        self.optimizer = Lion(self.model.parameters(), lr=self.lr, use_triton=True)
        self.compare_optimizer = Lion(self.compare_model.parameters(), lr=self.lr, use_triton=True)

    def compare_model_deep_unlearning(self,batches,retain_batches,epoch):

        compare_acc = self.base_model.eval(time = epoch, dataset = self.base_model.val_forget_dataset,
                              task_name = "eval on compare_model",model = self.compare_model)
        print(f"compare_model acc:{compare_acc}")
        for i, batch in enumerate(tqdm(batches, desc=f"deep GD unlearning epoch {epoch}")):

            self.compare_optimizer.zero_grad()

            j = i*random.randint(1,10000) % len(retain_batches)
            forget_loss = -self.base_model.get_loss(batch=batch,model = self.compare_model)
            retain_loss = self.base_model.get_loss(batch=retain_batches[j],model = self.compare_model)

            try:
                loss = forget_loss + self.base_model.retain_coeff * retain_loss
            except Exception as e:

                raise e

            loss.backward()
            
            self.compare_optimizer.step()
            
    def dynamic_deep_unlearning(self,epochs:int=8):

        if epochs is None:
            epochs = self.base_model.epochs
        print(f"num_epochs: {epochs}")
        self.base_model.eval_on_RTV(0,"dynamic_GD_deep_unlearning")

        for epoch in range(epochs):
            if epoch == 0:
                print("in dynamic_deep_choose epochs")

            self.compare_model.train()

            batches = [self.base_model.val_forget_dataset[i : i + self.base_model.batch_size] for i in range(0, len(self.base_model.val_forget_dataset), self.base_model.batch_size)]
            retain_batches = [self.base_model.val_retain_dataset[i : i + self.base_model.batch_size] for i in range(0, len(self.base_model.val_retain_dataset), self.base_model.batch_size)]
            print(f"{len(batches)=}")

            with torch.no_grad():
                self.exchang_block()

            self.compare_model_deep_unlearning(batches,retain_batches,epoch)

            with torch.no_grad():
                self.recover_exchange_block()

            if ((epoch + 1) % self.base_model.eval_every) == 0:
                eval_res = self.base_model.eval_on_RTV(epoch + 1,"dynamic_GD_deep_unlearning")
                