from unlearn import *
from unlearning_way.Biru import DEEP_UNL
from unlearning_way.gd import gd_unlearning
from unlearning_way.Rmu import run_rmu,run_rmu_rtt
from unlearning_way.ria import ria,ria_rtt
from unlearning_way.RIA_mcq import ria_mcq_unlearning
from unlearning_way.ga import ga_mcq_unlearning
from unlearning_way.npo import npo
import unlearning_way.top_layers_compare as topcom

have_knowledge_path = "/root/autodl-tmp/model/Meta-Llama-3-8B-Instruct"
unlearned_path = "/root/autodl-tmp/BIRU/models/Years_gd"
deff_token_path = have_knowledge_path

data_load_name="WMDP"   
now_val_batch_size = 32
now_batch_size = 6

def run_Biru():
    base_model = Base_model(
    data_load_name = data_load_name,
    base_model= "/root/autodl-tmp/BIRU/models/WMDP-biru-ungd",
    diff_tokenizer=have_knowledge_path,
    
    project_name= "exp_data",
    
    lr= 2e-7,
    retain_coeff=1,
    name= f"{data_load_name}_biru-ungd",
    k_shot = 0,
    epochs= 3,
    batch_size = now_batch_size,
    val_batch_size=now_val_batch_size,
    eval_every=4,
    )
    deep_unlearn = DEEP_UNL(base_model,choose_num=1,compare_train_times=3,compare_model_path=have_knowledge_path)
    deep_unlearn.dynamic_deep_unlearning(3*6)

    base_model.save_model(f"models/{data_load_name}-biru-ungd")


if __name__=="__main__":
    
    run_Biru()
