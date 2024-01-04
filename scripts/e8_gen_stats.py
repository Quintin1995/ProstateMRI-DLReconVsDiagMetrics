import argparse
import matplotlib.pyplot as plt
import glob
from fastMRI_PCa.utils import print_p, read_yaml_to_dict
import pandas as pd


################################  README  ######################################
# NEW -  This contains plot functionality for training losses and metrics. This script
# should be called on a target directory with a .csv file with train results.
# Certain types of metrics and losses are detected and plotted. If you add a new
# metric or loss, then its respective plot functionality should be added to this
# script.


def parse_input_args():
    parser = argparse.ArgumentParser(description='Parse arguments for training a Reconstruction model')

    parser.add_argument('-d',
                        '--dirs',
                        nargs='+',
                        type=str,
                        help='Directories with model training data (Each folder should contain a training_log.csv file). Statistics will be gathered from these folders.')

    args = parser.parse_args()
    print(f"Plotting {args}")

    return args


def plot_train_val_froc(df: pd.DataFrame, outdir: str) -> None:
    print_p("\nfROC column detected in train output. ")

    title = f"Partial AUC fROC"
    
    plt.scatter(df['epoch'], df[f"val_froc"], s=3, label=f"val fROC", color=COLORS[1])

    if DO_EMA:
        df[F"val_fROC_EMA{ALPHA}"] = df.val_froc.ewm(alpha=ALPHA, adjust=False).mean()

        plt.plot(df["epoch"], df[f"val_fROC_EMA{ALPHA}"], color=COLORS_D[1])
        title += f" - EMA{ALPHA}"

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel("pfROC")
    plt.grid()
    plt.legend()
    fname = f"train_output/{outdir}/{outdir}_scatter_fROC.png"
    plt.savefig(fname, dpi=DPI)
    plt.clf()
    plt.close()
    print_p(f"saved figure to {fname}")


def plot_auc(df: pd.DataFrame, outdir: str) -> None:
    print_p("\nAUC column detected in train output. ")

    title = f"AUC - {outdir.replace('_', ' ')}"
    
    plt.scatter(df['epoch'], df[f"AUC"], s=3, label=f"AUC", color=COLORS[0])
    plt.scatter(df['epoch'], df[f"val_AUC"], s=3, label=f"val AUC", color=COLORS[1])

    if DO_EMA:
        df[F"AUC_EMA{ALPHA}"] = df.AUC.ewm(alpha=ALPHA, adjust=False).mean()
        df[F"val_AUC_EMA{ALPHA}"] = df.val_AUC.ewm(alpha=ALPHA, adjust=False).mean()

        plt.plot(df["epoch"], df[f"AUC_EMA{ALPHA}"], color=COLORS_D[0])
        plt.plot(df["epoch"], df[f"val_AUC_EMA{ALPHA}"], color=COLORS_D[1])
        title += f" - EMA{ALPHA}"

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel("AUC")
    plt.grid()
    plt.legend()
    fname = f"train_output/{outdir}/{outdir}_scatter_AUC.png"
    plt.savefig(fname, dpi=DPI)
    plt.clf()
    plt.close()
    print_p(f"saved figure to {fname}")


def plot_loss(df: pd.DataFrame, outdir: str) -> None:
    print_p("\nLoss column detected in train output. ")

    train_params = read_yaml_to_dict(f"train_output/{outdir}/params.yml")
    loss = ""
    try:
        loss = train_params['loss']
    except:
        print(f"\nNO LOSS DETECTED in train parameter dump file {outdir}/params.yml \n")

    title = f"{loss} Loss - {outdir.replace('_', ' ')}"
    
    plt.scatter(df['epoch'], df[f"loss"], s=3, label=f"Loss", color=COLORS[0])
    plt.scatter(df['epoch'], df[f"val_loss"], s=3, label=f"val Loss", color=COLORS[1])

    if DO_EMA:
        df[F"loss_EMA{ALPHA}"] = df.loss.ewm(alpha=ALPHA, adjust=False).mean()
        df[F"val_loss_EMA{ALPHA}"] = df.val_loss.ewm(alpha=ALPHA, adjust=False).mean()

        plt.plot(df["epoch"], df[f"loss_EMA{ALPHA}"], color=COLORS_D[0])
        plt.plot(df["epoch"], df[f"val_loss_EMA{ALPHA}"], color=COLORS_D[1])
        title += f" - EMA{ALPHA}"

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    fname = f"train_output/{outdir}/{outdir}_scatter_loss.png"
    plt.savefig(fname, dpi=DPI)
    plt.clf()
    plt.close()
    print_p(f"saved figure to {fname}")


def plot_psnr(df: pd.DataFrame, outdir: str) -> None:
    print_p("\nPSNR column detected in train output. ")

    title = f"PSNR - {outdir.replace('_', ' ')}"
    
    plt.scatter(df['epoch'], df[f"psnr_metric"], s=3, label=f"PSNR", color=COLORS[0])
    plt.scatter(df['epoch'], df[f"val_psnr_metric"], s=3, label=f"val PSNR", color=COLORS[1])

    if DO_EMA:
        df[F"psnr_metric_EMA{ALPHA}"] = df.psnr_metric.ewm(alpha=ALPHA, adjust=False).mean()
        df[F"val_psnr_metric_EMA{ALPHA}"] = df.val_psnr_metric.ewm(alpha=ALPHA, adjust=False).mean()

        plt.plot(df["epoch"], df[f"psnr_metric_EMA{ALPHA}"], color=COLORS_D[0])
        plt.plot(df["epoch"], df[f"val_psnr_metric_EMA{ALPHA}"], color=COLORS_D[1])
        title += f" - EMA{ALPHA}"

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel("PSNR (dB)")
    plt.grid()
    plt.legend()
    fname = f"train_output/{outdir}/{outdir}_scatter_psnr.png"
    plt.savefig(fname, dpi=DPI)
    plt.clf()
    plt.close()
    print_p(f"saved figure to {fname}")


def plot_ssim(df: pd.DataFrame, outdir: str) -> None:
    print_p("\nSSIM column detected in train output. ")

    title = f"SSIM - {outdir.replace('_', ' ')}"
    
    plt.scatter(df['epoch'], df[f"ssim_metric"], s=3, label=f"SSIM", color=COLORS[0])
    plt.scatter(df['epoch'], df[f"val_ssim_metric"], s=3, label=f"val SSIM", color=COLORS[1])

    if DO_EMA:
        df[F"ssim_metric_EMA{ALPHA}"] = df.ssim_metric.ewm(alpha=ALPHA, adjust=False).mean()
        df[F"val_ssim_metric_EMA{ALPHA}"] = df.val_ssim_metric.ewm(alpha=ALPHA, adjust=False).mean()

        plt.plot(df["epoch"], df[f"ssim_metric_EMA{ALPHA}"], color=COLORS_D[0])
        plt.plot(df["epoch"], df[f"val_ssim_metric_EMA{ALPHA}"], color=COLORS_D[1])
        title += f" - EMA{ALPHA}"

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel("SSIM")
    plt.grid()
    plt.legend()
    fname = f"train_output/{outdir}/{outdir}_scatter_ssim.png"
    plt.savefig(fname, dpi=DPI)
    plt.clf()
    plt.close()
    print_p(f"saved figure to {fname}")


def plot_mse(df: pd.DataFrame, outdir: str) -> None:
    print_p("\nMSE column detected in train output. ")
    
    title = f"MSE - {outdir.replace('_', ' ')}"
    
    plt.scatter(df['epoch'], df[f"mse"], s=3, label=f"MSE", color=COLORS[0])
    plt.scatter(df['epoch'], df[f"val_mse"], s=3, label=f"val MSE", color=COLORS[1])

    if DO_EMA:
        df[F"mse_EMA{ALPHA}"] = df.mse.ewm(alpha=ALPHA, adjust=False).mean()
        df[F"val_mse_EMA{ALPHA}"] = df.val_mse.ewm(alpha=ALPHA, adjust=False).mean()

        plt.plot(df["epoch"], df[f"mse_EMA{ALPHA}"], color=COLORS_D[0])
        plt.plot(df["epoch"], df[f"val_mse_EMA{ALPHA}"], color=COLORS_D[1])
        title += f" - EMA{ALPHA}"

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel("MSE")
    plt.grid()
    plt.legend()
    fname = f"train_output/{outdir}/{outdir}_scatter_mse.png"
    plt.savefig(fname, dpi=DPI)
    plt.clf()
    plt.close()
    print_p(f"saved figure to {fname}")


def plot_iou(df: pd.DataFrame, outdir: str) -> None:
    print_p("\nIoU column detected in train output. ")
    title = f"IoU - {outdir.replace('_', ' ')}"
    
    plt.scatter(df['epoch'], df[f"mean_io_u"], s=3, label=f"IoU", color=COLORS[0])
    plt.scatter(df['epoch'], df[f"val_mean_io_u"], s=3, label=f"val IoU", color=COLORS[1])

    if DO_EMA:
        df[F"mean_io_u_EMA{ALPHA}"] = df.mean_io_u.ewm(alpha=ALPHA, adjust=False).mean()
        df[F"val_mean_io_u_EMA{ALPHA}"] = df.val_mean_io_u.ewm(alpha=ALPHA, adjust=False).mean()

        plt.plot(df["epoch"], df[f"mean_io_u_EMA{ALPHA}"], color=COLORS_D[0])
        plt.plot(df["epoch"], df[f"val_mean_io_u_EMA{ALPHA}"], color=COLORS_D[1])
        title += f" - EMA{ALPHA}"

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel("IoU")
    plt.grid()
    plt.legend()
    fname = f"train_output/{outdir}/{outdir}_scatter_iou.png"
    plt.savefig(fname, dpi=DPI)
    plt.clf()
    plt.close()
    print_p(f"saved figure to {fname}")


################################################################################
WINDOW_SIZE = 25
ALPHA = 0.05

DPI = 250
SKIP_FIRST_2_PERCENT = True
DO_EMA = True
COLORS = ['dodgerblue', 'orange', 'chartreuse']
COLORS_D = ['blue', 'darkgoldenrod', 'yellowgreen']

PERCENT_SKIP = 0.02

if __name__ == '__main__':
        
    args = parse_input_args()

    for dir_ in args.dirs:
        csv = glob.glob(f"train_output/{dir_}/*.csv")[0]
        df = pd.read_csv(csv)

        print(f"indexes skipped = {int(PERCENT_SKIP * len(df.index))}")

        if SKIP_FIRST_2_PERCENT:
            df = df.iloc[int(PERCENT_SKIP * len(df.index)):, :]

        if "loss" in df:
            plot_loss(df, dir_)
        if "psnr_metric" in df:
            plot_psnr(df, dir_)
        if "ssim_metric" in df:
            plot_ssim(df, dir_)
        if "mse" in df:
            plot_mse(df, dir_)
        if "mean_io_u" in df:
            plot_iou(df, dir_)
        if "AUC" in df:
            plot_auc(df, dir_)
        if "val_froc" in df:
            plot_train_val_froc(df, dir_)
        
        print(f"\nDone with: {dir_}")


    print("\n-- DONE --\n")