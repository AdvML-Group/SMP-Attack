import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import numpy as np
import argparse
import csv
import os
from tqdm import tqdm
from models_tf.swin import swin_base_patch4_window7_224

models = ["pit_s_224",'cait_s24_224', 'deit_base_patch16_224', 'swin_small_patch4_window7_224','vit_base_patch16_224','visformer_small','convit_base','twins_pcpvt_base','deit_small_patch16_224','swin-Badv','xcit_small_12_p16_224']  

def create_timm_model(model_name):
    if model_name=='xcit_small_12_p16_224':
        model = timm.create_model(model_name, pretrained=False)
        model_path = './model_tf/xcit-s12-ImageNet-eps-8.pth.tar'
        model.load_state_dict(torch.load(model_path))
        model.cuda()
        model.eval()
    elif model_name=='deit_small_patch16_224':
        model = timm.create_model(model_name, pretrained=False)
        model_path = './model_tf/advdeit_small.pth'
        model.load_state_dict(torch.load(model_path)['model'])
        model.cuda()
        model.eval()
    elif model_name=='swin-Badv':
        model_path = './model_tf/ad-Swin-B.pth'
        model = swin_base_patch4_window7_224(pretrained=False)
        a=torch.load(model_path,map_location='cuda:0')
        new_state_dict={k.replace('module.',''):v for k,v in a['state_dict'].items()}
        model.load_state_dict(new_state_dict)
        model.cuda()
        model.eval()
    else:
        model = timm.create_model(model_name, pretrained=True).eval().cuda()

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return model, transform


def load_lables(label_file):
    labels = []
    with open(label_file) as f:
        ground_truth = f.read().split('\n')[:-1]
    for i in ground_truth:
        labels.append(int(i))
    return np.array(labels)


def verify_images(model, transform, image_path):
    model_pred = [] 
    files = os.listdir(image_path)
    files.sort(key=lambda x: int(x[:-4]))
    with torch.no_grad():
        for filename in tqdm(files):
            image = Image.open(os.path.join(image_path, filename)).convert('RGB')
            tensor = transform(image).unsqueeze(0).cuda() 
            out = model(tensor)
            probabilities = torch.nn.functional.softmax(out[0], dim=0)
            pred_label = torch.argmax(probabilities).detach().cpu()
            model_pred.append(pred_label)
    return np.array(model_pred)


def main(ori_path, adv_path, output_file):
    ori_accuracys = [adv_path]
    adv_accuracys = [adv_path]
    adv_successrates = [adv_path]
    label_file = './labels.txt'
    s = ['model', ]
    s.extend(models)
    ground_truth = load_lables(label_file)-1
    with open(output_file, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(s)
        f.close()
    for model_name in models:
        model, transform = create_timm_model(model_name)
        ori_pre = verify_images(model, transform, ori_path)
        adv_pre = verify_images(model, transform, adv_path)
        ori_accuracy = np.sum(ori_pre == ground_truth) / 1000
        adv_accuracy = np.sum(adv_pre == ground_truth) / 1000
        adv_successrate = np.sum(ori_pre != adv_pre) / 1000
        adv_successrate2 = np.sum(ground_truth != adv_pre) / 1000
        print('ori_acc:{:.1%}/adv_acc:{:.1%}/adv_suc:{:.1%}/adv_suc2:{:.1%}'.format(ori_accuracy, adv_accuracy,
                                                                                    adv_successrate,
                                                                                    adv_successrate2))
        ori_accuracys.append('{:.1%}'.format(ori_accuracy))
        adv_accuracys.append('{:.1%}'.format(adv_accuracy))
        adv_successrates.append('{:.1%}'.format(adv_successrate))

    with open(output_file, 'a+', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow([adv_path])
        writer.writerow(adv_successrates)
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_path', default='./dataset/images')
    parser.add_argument('--adv_path', default='./adv/SMP')
    parser.add_argument('--output_file', default='./log_vit.csv')
    args = parser.parse_args()
    main(args.ori_path, args.adv_path, args.output_file)
