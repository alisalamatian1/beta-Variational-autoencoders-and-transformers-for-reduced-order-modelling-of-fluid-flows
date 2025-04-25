from itertools import product
from configs.vae import VAE_config
from configs.easyAttn import easyAttn_config
from main import main
import sys

# Define search space
'''
Explanation for our choices:
We kept the epoch numbers of the vae as the default 1000 since through our experiments it was shown that it is an upperbound for what is needed
We are changing the latent dim since it could be interesting
epoch list for easy
beta -> orthogonality
easy_params -> determines the number of params as well as the window size
'''
latent_dims = [2,3,5,7]
epochs_list = [50, 100, 500]
betas = [0.001, 0.005, 0.01, 0.05, 0.1]
easy_params = [8, 32, 64, 128]
num_heads = [4, 8, 16]



# Two config test
# latent_dims = [3]
# epochs_list = [50]
# betas = [0.001]
# easy_params = [8]
# num_heads = [4, 8]


# Iterate over all combinations
for latent_dim, epochs, beta, easy_param, num_head in product(
    latent_dims, epochs_list, betas, easy_params, num_heads
):
    
    # check if the num_heads is divisible by the easy params
    if easy_param % num_head != 0:
        continue
    
    # Set class-level attributes
    VAE_config.latent_dim = latent_dim
    easyAttn_config.nmode = latent_dim
    
    easyAttn_config.Epoch = epochs
    
    VAE_config.beta = beta
    
    easyAttn_config.in_dim = easy_param
    easyAttn_config.out_dim = easy_param
    easyAttn_config.d_model = easy_param
    easyAttn_config.time_proj = easy_param
    easyAttn_config.proj_dim = 2 * easy_param
    
    easyAttn_config.num_head = num_head
    
    print(f"\n--- Running with: latent_dim={latent_dim}, easy epochs={epochs}, "
      f"beta={beta}, easy_param={easy_param}, num_head={num_head}")


    #  -m test -t_vae final -t_latent pre -metrics True
    '''
    parser.add_argument('-nn',default="easy", type=str,   help="Choose the model for time-series prediction: easy, self OR lstm")
    parser.add_argument('-re',default=40,     type=int,   help="40 OR 100, Choose corresponding Reynolds number for the case")
    parser.add_argument('-m', default="test", type=str,   help='Switch the mode between train, infer and run')
    parser.add_argument('-t_vae', default="pre",  type=str,    help='The type of saved model: pre/val/final')
    parser.add_argument('-t_latent', default="pre",  type=str,   help='The type of saved model: pre/val/final')
    parser.add_argument('-pod',default=True, type=bool,    help='Compute POD')
    parser.add_argument('-metrics', default=False, type=bool, help="Flag for computing the metrics")
    '''
    # Run training
    # main(m="train", metrics=True)

    # # Run testing
    # main(mode="test", t_vae="final", t_latent="pre", metrics=True)
    try:
        sys.argv = [
            "main.py",
            "-nn", "easy",
            "-re", "40",
            "-m", "run",
            "-t_vae", "pre",
            "-t_latent", "pre",
            "-pod", "False",
            "-metrics", "True"
        ]
        main()

        sys.argv = [
            "main.py",
            "-nn", "easy",
            "-re", "40",
            "-m", "test",
            "-t_vae", "final",
            "-t_latent", "pre",
            "-pod", "False",
            "-metrics", "True"
        ]
        main()
    except Exception as e:
        print(f"Error: {e}")
