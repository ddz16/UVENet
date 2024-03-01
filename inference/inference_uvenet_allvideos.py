import argparse
import cv2
import glob
import os
import shutil
import torch
import numpy as np

from basicsr.archs.uvenet_arch import UVENet
from basicsr.metrics.psnr_ssim import calculate_ssim, calculate_psnr
from basicsr.metrics.temporal_metrics import calculate_mabd_mse, calculate_cdc
from basicsr.metrics.uciqe import calculate_uciqe
from basicsr.metrics.uiqm import calculate_uiqm
from basicsr.data.data_util import read_img_seq
from basicsr.utils.img_util import tensor2img


def calculate_NRIQA(imgs):
    # imgs (list): each element has cv2.imread format
    niqe, uiqm, uciqe = 0., 0., 0.
    for img in imgs:
        # niqe += calculate_niqe(img, 0)
        uiqm += calculate_uiqm(img, 0)
        uciqe += calculate_uciqe(img, 0)
    return niqe/len(imgs), uiqm/len(imgs), uciqe/len(imgs)

def calculate_NRIQA1(imgs):
    brisque, pi = 0., 0.
    # for img in imgs:
    #     brisque += brisque_metrics(img).item()
    #     pi += pi_metrics(img).item()
    return brisque/len(imgs), pi/len(imgs)

# def calculate_temporal_metrics(imgs, gt_imgs):
#     mabd_mse, cdc = 0., 0.
#     if gt_imgs is not None:
#         mabd_mse = calculate_mabd_mse(imgs, gt_imgs)
#     cdc = calculate_cdc(imgs)
#     return mabd_mse, cdc

def calculate_temporal_metrics(imgs, gt_imgs):
    # imgs (t h w c): each element has cv2.imread format
    mabd_mse, cdc = 0., 0.
    if gt_imgs is not None:
        gt_imgs = np.stack(gt_imgs, axis=0)
        mabd_mse = calculate_mabd_mse(imgs, gt_imgs)
    cdc = calculate_cdc(imgs)
    return mabd_mse, cdc


def calculate_RIQA(imgs, gt_imgs):
    psnr, ssim, mse = 0., 0., 0.
    if gt_imgs is not None:
        for img, gt_img in zip(imgs, gt_imgs):
            psnr += calculate_psnr(img, gt_img, 0)
            ssim += calculate_ssim(img, gt_img, 0)
            mse += np.mean((img - gt_img)**2)
    return psnr/len(imgs), ssim/len(imgs), mse/len(imgs)


def inference(imgs, imgnames, model, save_path, num_frame=5, gt_path=None):
    total_frame_num = imgs.shape[0]
    output_list = []
    # begin
    for i in range(num_frame//2):
        with torch.no_grad():
            pad = torch.cat([imgs[:1, ...] for _ in range(num_frame//2-i)], dim=0)
            input_pad = torch.cat([pad, imgs[:num_frame//2+i+1, ...]], dim=0)
            # print(input_pad.shape)
            output = model(input_pad.unsqueeze(0)).squeeze()
            output_list.append(output)
    # middle
    for i in range(total_frame_num-num_frame+1):
        with torch.no_grad():
            input = imgs[i:i+num_frame, ...]
            # print(input.shape)
            output = model(input.unsqueeze(0)).squeeze()
            output_list.append(output)
    # end
    for i in range(num_frame//2):
        with torch.no_grad():
            pad = torch.cat([imgs[-1:, ...] for _ in range(i+1)], dim=0)
            input_pad = torch.cat([imgs[total_frame_num-num_frame+i+1:, ...], pad], dim=0)
            # print(input_pad.shape)
            output = model(input_pad.unsqueeze(0)).squeeze()
            output_list.append(output)

    # save imgs
    output_video = []
    output_video_tensor = []
    for output, imgname in zip(output_list, imgnames):
        output_video_tensor.append(output)
        output = tensor2img(output)
        output_video.append(output)
        cv2.imwrite(os.path.join(save_path, f'{imgname}.png'), output)

    assert len(output_video) == total_frame_num

    # calculate metrics
    niqe, uiqm, uciqe = calculate_NRIQA(output_video)
    brisque, pi = calculate_NRIQA1(output_video_tensor)

    if gt_path is not None:
        gt_imgs = read_img_seq(gt_path)
        gt_imgs = list(torch.split(gt_imgs, 1, dim=0))
        gt_imgs = tensor2img(gt_imgs)
    else:
        gt_imgs = None
    mabd_mse, cdc = calculate_temporal_metrics(np.stack(output_video, axis=0), gt_imgs)
    psnr, ssim, mse = calculate_RIQA(output_video, gt_imgs)

    metrics = {
        "psnr": psnr,
        "ssim": ssim,
        "mse": mse,
        "niqe": niqe,
        "brisque": brisque,
        "pi": pi,
        "uiqm": uiqm,
        "uciqe": uciqe,
        "mabd_mse": mabd_mse,
        "cdc": cdc,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='experiments/1_UVENet_SUVE/models/net_g_latest.pth')
    parser.add_argument('--input_path', type=str, default='', help='frames folder or video file')
    parser.add_argument('--save_path', type=str, default='', help='save image path')
    parser.add_argument('--gt_path', type=str, default=None, help='gt video path')
    parser.add_argument('--frames_path', type=str, default='', help='if input_path is video, save the extracted frames')
    parser.add_argument('--interval', type=int, default=5, help='interval size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model
    model = UVENet(
        arch_type="tiny",
        num_frame=5,
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        out_indices=[0,1,2,3]
        )
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(args.save_path, exist_ok=True)

    all_metrics = []
    # extract images from video format files
    print(os.listdir(args.input_path))
    for item in os.listdir(args.input_path):
        item_path = os.path.join(args.input_path, item)
        if not os.path.isdir(item_path):  # video file
            video_name = os.path.splitext(os.path.split(item_path)[-1])[0]
            input_path = os.path.join(args.frames_path, video_name)
            os.makedirs(input_path, exist_ok=True)
            os.system(f'ffmpeg -i {item_path} -vf "fps=5,scale=256:256" -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {input_path}/frame%05d.png')
        else:  # frames folder
            video_name = os.path.basename(item_path)
            input_path = item_path
            gt_path = args.gt_path
            if args.gt_path is not None:
                gt_video_name = "_".join(item.rsplit("_", 1)[:-1])
                gt_path = os.path.join(args.gt_path, gt_video_name)

        # load data and inference
        imgs_list = sorted(glob.glob(os.path.join(input_path, '*.jpg'))+glob.glob(os.path.join(input_path, '*.png')))
        num_imgs = len(imgs_list)

        imgs, imgnames = read_img_seq(imgs_list, return_imgname=True)
        imgs = imgs.to(device)
        save_path = os.path.join(args.save_path, video_name)
        os.makedirs(save_path, exist_ok=True)
        metrics = inference(imgs, imgnames, model, save_path, num_frame=5, gt_path=gt_path)
        print(f"metrics of {video_name}:")
        print(metrics)
        all_metrics.append(metrics)

    average_metrics = {}
    for d in all_metrics:
        for key, value in d.items():
            if key in average_metrics:
                average_metrics[key] += value
            else:
                average_metrics[key] = value

    num_elements = len(all_metrics)
    for key in average_metrics:
        average_metrics[key] /= num_elements
    print("average metrics:")
    print(average_metrics)



if __name__ == '__main__':
    main()
    # python inference/inference_uvenet_allvideos.py --input_path datasets/MVK --save_path results/UVENet/MVK
    # python inference/inference_uvenet_allvideos.py --input_path datasets/SUVE/UW_test --save_path results/UVENet/SUVE --gt_path datasets/SUVE/GT