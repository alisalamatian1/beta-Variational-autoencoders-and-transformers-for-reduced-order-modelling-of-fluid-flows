"""
Main program

NOTE: The "run" in running mode here means we do both train and infer 

@yuningw
"""

import      torch
import      argparse
from        lib             import init, POD
from        lib.runners     import vaeRunner, latentRunner
from        utils.figs_time import vis_temporal_Prediction
from        utils.figs      import vis_bvae, vis_pod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nn',default="easy", type=str,   help="Choose the model for time-series prediction: easy, self OR lstm")
    parser.add_argument('-re',default=40,     type=int,   help="40 OR 100, Choose corresponding Reynolds number for the case")
    parser.add_argument('-m', default="test", type=str,   help='Switch the mode between train, infer and run')
    parser.add_argument('-t_vae', default="pre",  type=str,    help='The type of saved model: pre/val/final')
    parser.add_argument('-t_latent', default="pre",  type=str,   help='The type of saved model: pre/val/final')
    parser.add_argument('-pod',default=True, type=bool,    help='Compute POD')
    parser.add_argument('-metrics', default=False, type=bool, help="Flag for computing the metrics")
    args  = parser.parse_args()

    device = ('cuda' if torch.cuda.is_available() else "cpu")
## Env INIT
    # datafile = init.init_env(args.re)
    # datafile = "ns_incom_dt1_128.h5"
    # datafile = "data/NSCyl_Re160_dt1_321x129x4490.h5"
    datafile = "ns_incom_dt1_512.h5"

    ## Beta-VAE
    bvae   = vaeRunner(device,datafile, compute_metrics = args.metrics)
    if args.m == 'train':
        bvae.train()
    elif args.m == 'test':
        bvae.infer(args.t_vae)
    elif args.m == 'run':
        bvae.run()


    # POD
    # if args.pod:
    #     POD = POD.POD(datafile, n_test=bvae.config.n_test, re=args.re,
    #                 path='res/', n_modes=10, delta_t=bvae.config.delta_t)
    #     POD.load_data()
    #     POD.get_POD()
    #     POD.eval_POD()

    # # Time-series prediction runner 
    lruner = latentRunner(args.nn,device)
    if args.m == 'train':
        lruner.train()
    elif args.m == 'test':
        lruner.infer(args.t_latent)
    elif args.m == 'run':
        lruner.train()
        lruner.infer(args.t_latent)

    vis_bvae(init.pathsBib.res_path + "modes_" + bvae.filename + ".hdf5",
            init.pathsBib.log_path + bvae.filename)
    # vis_pod(POD)
    vis_temporal_Prediction(model_type=args.nn, predictor=lruner, vae=bvae, compute_metrics=args.metrics)

if __name__ == "__main__":
    main()
