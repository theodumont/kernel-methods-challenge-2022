import time
import argparse
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import models
import kernels
from utils import get_data, accuracy, save_Yte, augment_dataset, array_to_tensor
from hog import hog


# argument parser ========================================================================
parser = argparse.ArgumentParser(description='Kernel Methods for Machine Learning - Kaggle Data Challenge 2022')

parser.add_argument('--data_path',           type=str,   default='./data',      help="")
parser.add_argument('--pred_path',           type=str,   default='./data/preds', help="")
# HOG parameters
parser.add_argument('--hog',                 action='store_true', default=True, help="")
parser.add_argument('--hog_orientations',    type=int,   default=9,        help="")
parser.add_argument('--hog_pix_per_cell',    type=int,   default=8,        help="")
parser.add_argument('--hog_cells_per_block', type=int,   default=3,        help="")
parser.add_argument('--hog_normalization',   type=str,   default='L2-Hys', help="")
# kernel parameters
parser.add_argument('--ker',                 type=str,   default='Log',    help="")
parser.add_argument('--ker_gaussian_sigma',  type=float, default=1.,       help="")
parser.add_argument('--ker_RBF_g',           type=float, default=1.,       help="")
parser.add_argument('--ker_TStudent_p',      type=float, default=2.,       help="")
parser.add_argument('--ker_polynomial_a',    type=float, default=1.,       help="")
parser.add_argument('--ker_polynomial_b',    type=float, default=0.,       help="")
parser.add_argument('--ker_polynomial_p',    type=int,   default=2,        help="")
parser.add_argument('--ker_chi2_gamma',      type=float, default=1.,       help="")
parser.add_argument('--ker_wavelet_a',       type=float, default=1.,       help="")
parser.add_argument('--ker_log_d',           type=float, default=2.,       help="")
parser.add_argument('--ker_GHI_beta',        type=float, default=1.,       help="")
# model parameters
parser.add_argument('--model',               type=str,   default='KRR',    help="")
parser.add_argument('--model_KRR_lambda',    type=float, default=1e-4,     help="")
parser.add_argument('--model_KSVC_C',        type=float, default=1,        help="")
parser.add_argument('--model_KSVC_epsilon',  type=float, default=1e-3,     help="")
# other
parser.add_argument('--augment',             action='store_true', default=True, help="")
parser.add_argument('--augment_repeat',      type=int,   default=2,        help="")


# main script ============================================================================
def main():
    args = parser.parse_args()

    # set parameters ---------------------------------------------------------------------
    # kernel parameters
    if args.ker == 'Gaussian':     kwargs_ker = {'sigma': args.ker_gaussian_sigma}
    elif args.ker == 'RBF':        kwargs_ker = {'g':     args.ker_RBF_g}
    elif args.ker == 'TStudent':   kwargs_ker = {'p':     args.ker_TStudent_p}
    elif args.ker == 'Linear':     kwargs_ker = {}
    elif args.ker == 'Polynomial': kwargs_ker = {'a':     args.ker_polynomial_a, 'b': args.ker_polynomial_b, 'p': args.ker_polynomial_p}
    elif args.ker == 'Chi2':       kwargs_ker = {'gamma': args.ker_chi2_gamma}
    elif args.ker == 'Wavelet':    kwargs_ker = {'a':     args.ker_wavelet_a}
    elif args.ker == 'Log':        kwargs_ker = {'d':     args.ker_log_d}
    elif args.ker == 'GHI':        kwargs_ker = {'beta':  args.ker_GHI_beta}
    else:                          raise NotImplementedError("Kernel not implemented")

    # data augmentation
    if args.augment:
        kwargs_augment = {'repeat': args.augment_repeat}

    # HOG
    if args.hog:
        kwargs_hog = {
            'orientations':    args.hog_orientations,
            'pix_per_cell':    args.hog_pix_per_cell,
            'cells_per_block': args.hog_cells_per_block,
            'normalization':   args.hog_normalization}

    # model
    if args.model == 'KRR':    kwargs_model = {'lambd': args.model_KRR_lambda}
    elif args.model == 'KSVC': kwargs_model = {'C': args.model_KSVC_C, 'epsilon': args.model_KSVC_epsilon}
    else:                      raise NotImplementedError("Model not implemented")



    print("Time: ", time.ctime())
    # data -------------------------------------------------------------------------------
    print(20*'-')
    print('Loading data...')
    Xtr, Xte, Ytr = get_data(data_path=args.data_path)
    if args.augment:
        print("Data augmentation parameters:", kwargs_augment)
        transform = [
            transforms.RandomHorizontalFlip(p=.6),
            transforms.RandomAffine(degrees=(0,10), translate=(0.1,0.2), scale=(0.8,1.2)),
        ]
        Xtr, Ytr = augment_dataset(Xtr, Ytr, transform=transform, **kwargs_augment)

    if args.hog:
        print("HOG parameters:", kwargs_hog)
        Xtr_tensor = array_to_tensor(Xtr)
        Xte_tensor = array_to_tensor(Xte)
        hog_fun = lambda img: hog(img, **kwargs_hog)
        Xtr = np.array([hog_fun(img) for img in tqdm(Xtr_tensor, desc="Computing HOG (train)")])
        Xte = np.array([hog_fun(img) for img in tqdm(Xte_tensor, desc="Computing HOG (test) ")])

    # model ------------------------------------------------------------------------------
    print(20*'-')
    print("Model:", args.model, "with", args.ker, "kernel")
    print("Model parameters: ", kwargs_model)
    print("Kernel parameters:", kwargs_ker)
    model = getattr(models, f"OneVsRest{args.model}")(
        kernel=getattr(kernels, f"{args.ker}Kernel")(**kwargs_ker).kernel,
        **kwargs_model
    )
    model.fit(Xtr, Ytr)
    print("Done!")

    # predictions ------------------------------------------------------------------------
    print(20*'-')
    print(f"- accuracy on training set: {accuracy(Ytr, model.predict(Xtr))}")
    Yte = model.predict(Xte)
    # model_name = f"{args.hog*'HOG_'}{args.augment*'DA_'}{args.ker}_OneVsRest{args.model}_999"
    model_name = ""
    save_Yte(Yte, model_name=model_name)
    print(f"- predictions on test set saved in {args.pred_path}/Yte_pred.csv")



if __name__ == '__main__':
    main()