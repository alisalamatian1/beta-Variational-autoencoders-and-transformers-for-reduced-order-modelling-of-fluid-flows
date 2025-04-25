"""
Functions for naming the models in the present study
@yuningw 
"""
from configs.vae import VAE_config
from configs.easyAttn import easyAttn_config


def Name_VAE(cfg):
    
    """
    A function to name the VAE checkpoint 

    Args: 
        cfg: A class of VAE model configuration
    
    Returns:
        name: A string for VAE name
    """

    name =  f'beta{VAE_config.beta}_' + \
            f'dim{VAE_config.latent_dim}_'+\
            f'VAEepochs{VAE_config.epochs}' +\
            f'EASY-Number{easyAttn_config.in_dim}' +\
            f'NumHeads{easyAttn_config.num_head}' +\
            f'Easy-Epoch{easyAttn_config.Epoch}'
    
    return name



def Make_Transformer_Name(cfg):
    """
    A function to name the VAE checkpoint 

    Args: 
        cfg: A class of Transorfmer model configuration
    
    Returns:
        name: A string for Transformer model
    """
    
    name =  f'beta{VAE_config.beta}_' + \
            f'dim{VAE_config.latent_dim}_'+\
            f'VAEepochs{VAE_config.epochs}' +\
            f'EASY-Number{easyAttn_config.in_dim}' +\
            f'NumHeads{easyAttn_config.num_head}' +\
            f'Easy-Epoch{easyAttn_config.Epoch}'
            
    return name



def Make_LSTM_Name(cfg):
    """
    A function to name the LSTM checkpoint 

    Args: 
        cfg: A class of LSTM model configuration
    
    Returns:
        name: A string for LSTM model
    """
    
    case_name = f"LSTM_"+\
                f"{cfg.in_dim}in_"+\
                f"{cfg.d_model}dmodel_"+\
                f"{cfg.next_step}next_"+\
                f"{cfg.nmode}dim_"+\
                f"{cfg.embed}emb_"+\
                f"{cfg.hidden_size}hideen_"+\
                f"{cfg.num_layer}nlayer_"+\
                f"{cfg.out_act}outact_"+\
                f"{cfg.Epoch}Epoch_"+\
                f"{cfg.num_train}N_"+\
                f"{cfg.early_stop}ES_"+\
                f"{cfg.patience}P"
    
    return case_name



