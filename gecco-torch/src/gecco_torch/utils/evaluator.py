import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import pathlib
from piqa import SSIM
import lpips
import torch
from utils.loss_utils import l1_loss # , ssim
from utils.loss_utils import ssim as original_ssim
from utils.image_utils import psnr

class Evaluator():
    def __init__(self,output_path,train_images,val_images,test_images,max_iter,lpips_ssim_iters,log_train_metrics):
        self.log_train_metrics = log_train_metrics
        self.lpips_fn_net = lpips.LPIPS(net = 'alex').cuda()
        self.ssim_fn = SSIM().cuda()
        self.max_iter = max_iter
        if log_train_metrics:
            self.per_train_image_metrics = self._setup_metric_dict(train_images)
        self.per_val_image_metrics = self._setup_metric_dict(val_images)
        self.per_test_image_metrics = self._setup_metric_dict(test_images)

        self.lpips_ssim_iters = lpips_ssim_iters
        if log_train_metrics:
            self._make_dirs("train",train_images,output_path)
        self._make_dirs("val",val_images,output_path)
        self._make_dirs("test",test_images,output_path)

        self.output_path = output_path
        self.latest_iteration = 0

    def verbose_eval(self,iteration,psnr,l1,eval_image_name,eval_image,rendered_eval_image,split:str,depth_map = None,gt_depth_map = None):
        # print(f"Additional Evaluation split {split}...")
        self.latest_iteration = iteration
        eval_image = eval_image.permute(1,2,0)
        rendered_eval_image = rendered_eval_image.permute(1,2,0)
        # print("shape eval_image ",eval_image.shape)
        # print("shape rendered_eval_image ",rendered_eval_image.shape)
        if split == "train":
            per_image_metrics_dict = self.per_train_image_metrics
        if split == "val":
            per_image_metrics_dict = self.per_val_image_metrics
        elif split == "test":
            per_image_metrics_dict = self.per_test_image_metrics

        per_image_metrics_dict[eval_image_name]['iterations'].append(iteration)
        per_image_metrics_dict[eval_image_name]['psnr'].append(psnr.cpu().item())
        per_image_metrics_dict[eval_image_name]['l1'].append(l1.cpu().item())

        if iteration in self.lpips_ssim_iters:
            lpips_value, ssim_value = self._get_lpips_ssim(rendered_eval_image,eval_image)
            per_image_metrics_dict[eval_image_name]['iterations_lpips_ssim'].append(iteration)
            per_image_metrics_dict[eval_image_name]['lpips'].append(lpips_value)
            per_image_metrics_dict[eval_image_name]['ssim'].append(ssim_value)
            if depth_map is not None:
                depth_l1,depth_psnr = self._get_l1_psnr(depth_map,gt_depth_map)
                per_image_metrics_dict[eval_image_name]['depth_l1'].append(depth_l1.cpu().item())
                per_image_metrics_dict[eval_image_name]['depth_psnr'].append(depth_psnr.cpu().item())
                depth_lpips_value, depth_ssim_value = self._get_lpips_ssim(depth_map,gt_depth_map)
                per_image_metrics_dict[eval_image_name]['depth_lpips'].append(depth_lpips_value)
                # print(depth_ssim_value)
                per_image_metrics_dict[eval_image_name]['depth_ssim'].append(depth_ssim_value)

        if depth_map is not None:
            fig, ax = plt.subplots(2, 5, figsize=(35,12)) # only show image
            ax[1,0].imshow(depth_map.detach().detach().cpu().numpy(),cmap='gray')
            ax[1,0].set_title(f'Depth map')
            ax[1,1].imshow(gt_depth_map.detach().detach().cpu().numpy(),cmap='gray')
            ax[1,1].set_title(f'Target Depth map')

            ax[1,2].plot(per_image_metrics_dict[eval_image_name]['iterations'], per_image_metrics_dict[eval_image_name]['depth_l1'], 'r') # red
            ax[1,2].set_title('L1 Loss ({:.4f})'.format(per_image_metrics_dict[eval_image_name]['depth_l1'][-1]))
            ax[1,2].grid(True)

            ax[1,3].plot(per_image_metrics_dict[eval_image_name]['iterations'], per_image_metrics_dict[eval_image_name]['depth_psnr'], 'r') # red
            ax[1,3].set_title('PSNR ({:.4f})'.format(per_image_metrics_dict[eval_image_name]['depth_psnr'][-1]))
            ax[1,3].grid(True)

            ax[1,4].plot(per_image_metrics_dict[eval_image_name]['iterations_lpips_ssim'], per_image_metrics_dict[eval_image_name]['depth_lpips'], 'r') # red
            ax[1,4].plot(per_image_metrics_dict[eval_image_name]['iterations_lpips_ssim'], per_image_metrics_dict[eval_image_name]['depth_ssim'], 'b') # red
            ax[1,4].set_title('LPIPS {:.4f} (red), SSIM {:.4f} (blue)'.format(per_image_metrics_dict[eval_image_name]['depth_lpips'][-1],per_image_metrics_dict[eval_image_name]['depth_ssim'][-1]))
            ax[1,4].grid(True)


        else:
            fig, ax = plt.subplots(1, 5, figsize=(35,7)) # only show image
        ax[0,0].imshow(rendered_eval_image.detach().cpu().numpy())
        ax[0,0].set_title(f'Iteration: {iteration}')

        ax[0,1].imshow(eval_image.detach().detach().cpu().numpy()) # pick one image
        ax[0,1].set_title(f'Target')

        ax[0,2].plot(per_image_metrics_dict[eval_image_name]['iterations'], per_image_metrics_dict[eval_image_name]['l1'], 'r') # red
        ax[0,2].set_title('L1 Loss ({:.4f})'.format(per_image_metrics_dict[eval_image_name]['l1'][-1]))
        ax[0,2].grid(True)

        ax[0,3].plot(per_image_metrics_dict[eval_image_name]['iterations'], per_image_metrics_dict[eval_image_name]['psnr'], 'r') # red
        ax[0,3].set_title('PSNR ({:.4f})'.format(per_image_metrics_dict[eval_image_name]['psnr'][-1]))
        ax[0,3].grid(True)

        ax[0,4].plot(per_image_metrics_dict[eval_image_name]['iterations_lpips_ssim'], per_image_metrics_dict[eval_image_name]['lpips'], 'r') # red
        ax[0,4].plot(per_image_metrics_dict[eval_image_name]['iterations_lpips_ssim'], per_image_metrics_dict[eval_image_name]['ssim'], 'b') # red
        ax[0,4].set_title('LPIPS {:.4f} (red), SSIM {:.4f} (blue)'.format(per_image_metrics_dict[eval_image_name]['lpips'][-1],per_image_metrics_dict[eval_image_name]['ssim'][-1]))
        ax[0,4].grid(True)

        plt.savefig(pathlib.Path(self.output_path,"metrics",split,"img_{}".format(eval_image_name),"iter_{}.png".format(str(iteration).zfill(6))))
        plt.close()

    def _pandas_image_metric_dict(self,image_metric_dict,outpath):
        iterations = image_metric_dict['iterations']
        iterations_lpips_ssim = image_metric_dict['iterations_lpips_ssim']
        pd.DataFrame({
                'iterations':iterations,
                'psnr' : image_metric_dict['psnr'],
                'l1' : image_metric_dict['l1'],
                'lpips' : self._adjust_lengths(iterations,image_metric_dict['lpips'],iterations_lpips_ssim),
                'ssim' :  self._adjust_lengths(iterations,image_metric_dict['ssim'],iterations_lpips_ssim),
            }).to_csv(outpath,index=False)

    def summarize(self):
        # for each single image metrics dataframe and average over all -> graph and dataframe

        # for each single image -> dataframes
        if self.log_train_metrics:
            for im in self.per_train_image_metrics.keys():
                self._pandas_image_metric_dict(self.per_train_image_metrics[im],pathlib.Path(self.output_path,"metrics","train","img_{}".format(im),"metrics_{}.csv".format(self.latest_iteration)))

        for im in self.per_val_image_metrics.keys():
            self._pandas_image_metric_dict(self.per_val_image_metrics[im],pathlib.Path(self.output_path,"metrics","val","img_{}".format(im),"metrics_{}.csv".format(self.latest_iteration)))

        iterations = self.per_val_image_metrics[list(self.per_val_image_metrics.keys())[0]]['iterations']
        iterations_lpips_ssim = self.per_val_image_metrics[list(self.per_val_image_metrics.keys())[0]]['iterations_lpips_ssim']

        if max(iterations) == self.max_iter:
            for im in self.per_test_image_metrics.keys():
                self._pandas_image_metric_dict(self.per_test_image_metrics[im],pathlib.Path(self.output_path,"metrics","test","img_{}".format(im),"metrics_{}.csv".format(self.latest_iteration)))

        if self.log_train_metrics:
            average_loss_per_iteration_train = [self._average_something(self.per_train_image_metrics,'l1',i) for i in \
                                        range(len(self.per_train_image_metrics[list(self.per_train_image_metrics.keys())[0]]['iterations']))]
            average_psnr_per_iteration_train = [self._average_something(self.per_train_image_metrics,'psnr',i) for i in \
                                        range(len(self.per_train_image_metrics[list(self.per_train_image_metrics.keys())[0]]['iterations']))]
            average_lpips_per_iteration_train = [self._average_something(self.per_train_image_metrics,'lpips',i) for i in \
                                        range(len(self.per_train_image_metrics[list(self.per_train_image_metrics.keys())[0]]['iterations_lpips_ssim']))]
            average_ssim_per_iteration_train = [self._average_something(self.per_train_image_metrics,'ssim',i) for i in \
                                        range(len(self.per_train_image_metrics[list(self.per_train_image_metrics.keys())[0]]['iterations_lpips_ssim']))]
        
        average_loss_per_iteration_val = [self._average_something(self.per_val_image_metrics,'l1',i) for i in \
                                      range(len(self.per_val_image_metrics[list(self.per_val_image_metrics.keys())[0]]['iterations']))]
        average_psnr_per_iteration_val = [self._average_something(self.per_val_image_metrics,'psnr',i) for i in \
                                      range(len(self.per_val_image_metrics[list(self.per_val_image_metrics.keys())[0]]['iterations']))]
        average_lpips_per_iteration_val = [self._average_something(self.per_val_image_metrics,'lpips',i) for i in \
                                      range(len(self.per_val_image_metrics[list(self.per_val_image_metrics.keys())[0]]['iterations_lpips_ssim']))]
        average_ssim_per_iteration_val = [self._average_something(self.per_val_image_metrics,'ssim',i) for i in \
                                      range(len(self.per_val_image_metrics[list(self.per_val_image_metrics.keys())[0]]['iterations_lpips_ssim']))]
        
        average_loss_per_iteration_test = [self._average_something(self.per_test_image_metrics,'l1',i) for i in \
                                      range(len(self.per_test_image_metrics[list(self.per_test_image_metrics.keys())[0]]['iterations']))]
        average_psnr_per_iteration_test = [self._average_something(self.per_test_image_metrics,'psnr',i) for i in \
                                      range(len(self.per_test_image_metrics[list(self.per_test_image_metrics.keys())[0]]['iterations']))]
        average_lpips_per_iteration_test = [self._average_something(self.per_test_image_metrics,'lpips',i) for i in \
                                      range(len(self.per_test_image_metrics[list(self.per_test_image_metrics.keys())[0]]['iterations_lpips_ssim']))]
        average_ssim_per_iteration_test = [self._average_something(self.per_test_image_metrics,'ssim',i) for i in \
                                      range(len(self.per_test_image_metrics[list(self.per_test_image_metrics.keys())[0]]['iterations_lpips_ssim']))]
        
        
        if self.log_train_metrics:
            pd.DataFrame({
                'iterations':iterations,
                'train_psnr' : average_psnr_per_iteration_train,
                'train_lpips' : self._adjust_lengths(iterations,average_lpips_per_iteration_train,iterations_lpips_ssim),
                'ssim' : self._adjust_lengths(iterations,average_ssim_per_iteration_train,iterations_lpips_ssim),
            }).to_csv(pathlib.Path(self.output_path,"metrics","train","eval_{}.csv".format(self.latest_iteration)),index=False)
        
        pd.DataFrame({
            'iterations':iterations,
            'val_psnr' : average_psnr_per_iteration_val,
            'val_l1' : average_loss_per_iteration_val,
            'val_lpips' : self._adjust_lengths(iterations,average_lpips_per_iteration_val,iterations_lpips_ssim),
            'val_ssim' : self._adjust_lengths(iterations,average_ssim_per_iteration_val,iterations_lpips_ssim),
        }).to_csv(pathlib.Path(self.output_path,"metrics","val","eval_{}.csv".format(self.latest_iteration)),index=False)

        if max(iterations) == self.max_iter:
            iterations_test = self.per_test_image_metrics[list(self.per_test_image_metrics.keys())[0]]['iterations']
            iterations_test_lpips_ssim = self.per_test_image_metrics[list(self.per_test_image_metrics.keys())[0]]['iterations_lpips_ssim']
            pd.DataFrame({
                'iterations':iterations_test,
                'test_psnr' : average_psnr_per_iteration_test,
                'test_l1' : average_loss_per_iteration_test,
                'test_lpips' : self._adjust_lengths(iterations_test,average_lpips_per_iteration_test,iterations_test_lpips_ssim),
                'test_ssim' : self._adjust_lengths(iterations_test,average_ssim_per_iteration_test,iterations_test_lpips_ssim),
            }).to_csv(pathlib.Path(self.output_path,"metrics","test","eval_{}.csv".format(max(iterations_test))),index=False)


        # individual train plot
        if self.log_train_metrics:
            fig, ax = plt.subplots(1, 3, figsize=(35,4), gridspec_kw={'width_ratios': [1, 1, 1]}) # only show image
            ax[0].plot(iterations, average_loss_per_iteration_train, 'r') # red
            ax[0].set_title('L1 Loss ({:.4f})'.format(average_loss_per_iteration_train[-1]))
            ax[0].grid(True)

            ax[1].plot(iterations, average_psnr_per_iteration_train, 'r') # red
            ax[1].set_title('PSNR ({:.4f})'.format(average_psnr_per_iteration_train[-1]))
            ax[1].grid(True)

            ax[2].plot(iterations_lpips_ssim, average_lpips_per_iteration_train, 'r') # red
            ax[2].plot(iterations_lpips_ssim, average_ssim_per_iteration_train, 'b') # red
            ax[2].set_title('LPIPS (red) {:.4f}, SSIM (blue) {:.4f}'.format(average_lpips_per_iteration_train[-1],average_ssim_per_iteration_train[-1]))
            ax[2].grid(True)

            plt.savefig(pathlib.Path(self.output_path,"metrics","train","metrics_{}.png".format(self.latest_iteration)))
            plt.close()

        # individual val plot

        fig, ax = plt.subplots(1, 3, figsize=(35,4), gridspec_kw={'width_ratios': [1, 1, 1]}) # only show image
        ax[0].plot(iterations, average_loss_per_iteration_val, 'r') # red
        ax[0].set_title('L1 Loss ({:.4f})'.format(average_loss_per_iteration_val[-1]))
        ax[0].grid(True)

        ax[1].plot(iterations, average_psnr_per_iteration_val, 'r') # red
        ax[1].set_title('PSNR ({:.4f})'.format(average_psnr_per_iteration_val[-1]))
        ax[1].grid(True)

        ax[2].plot(iterations_lpips_ssim, average_lpips_per_iteration_val, 'r') # red
        ax[2].plot(iterations_lpips_ssim, average_ssim_per_iteration_val, 'b') # red
        ax[2].set_title('LPIPS (red) ({:.4f}), SSIM (blue) {:.4f}'.format(average_lpips_per_iteration_val[-1],average_ssim_per_iteration_val[-1]))
        ax[2].grid(True)

        plt.savefig(pathlib.Path(self.output_path,"metrics","val","metrics_{}.png".format(self.latest_iteration)))
        plt.close()


        # big train, val, test comparison plot
        fig, ax = plt.subplots(1, 4, figsize=(35,4), gridspec_kw={'width_ratios': [1, 1, 1, 1]}) # only show image
        if self.log_train_metrics:
            ax[0].plot(iterations, average_loss_per_iteration_train, 'r') # red
        ax[0].plot(iterations, average_loss_per_iteration_val, 'b')
        if max(iterations) == self.max_iter:
            average_loss_per_iteration_test_shape_adjusted = np.full_like(np.array(average_loss_per_iteration_val), fill_value=average_loss_per_iteration_test[-1])
            ax[0].plot(iterations, average_loss_per_iteration_test_shape_adjusted, 'g')
            if self.log_train_metrics:
                ax[0].set_title('L1 Loss (red train {:.4f}, blue val {:.4f}, green test {:.4f})'.format(average_loss_per_iteration_train[-1],average_loss_per_iteration_val[-1],average_loss_per_iteration_test_shape_adjusted[-1]))
            else:
                ax[0].set_title('L1 Loss (blue val {:.4f}, green test {:.4f})'.format(average_loss_per_iteration_val[-1],average_loss_per_iteration_test_shape_adjusted[-1]))
        else:
            if self.log_train_metrics:
                ax[0].set_title('L1 Loss (red train {:.4f}, blue val {:.4f})'.format(average_loss_per_iteration_train[-1],average_loss_per_iteration_val[-1]))
            else:
                ax[0].set_title('L1 Loss (blue val {:.4f})'.format(average_loss_per_iteration_val[-1]))
        ax[0].grid(True)

        if self.log_train_metrics:
            ax[1].plot(iterations, average_psnr_per_iteration_train, 'r') # red
        ax[1].plot(iterations, average_psnr_per_iteration_val, 'b')
        if max(iterations) == self.max_iter:
            average_psnr_per_iteration_test_shape_adjusted = np.full_like(np.array(average_psnr_per_iteration_val), fill_value=average_psnr_per_iteration_test[-1])
            ax[1].plot(iterations, average_psnr_per_iteration_test_shape_adjusted, 'g')
            if self.log_train_metrics:
                ax[1].set_title('PSNR (red train {:.4f}, blue val {:.4f}, green test {:.4f})'.format(average_psnr_per_iteration_train[-1],average_psnr_per_iteration_val[-1],average_psnr_per_iteration_test_shape_adjusted[-1]))
            else:
                ax[1].set_title('PSNR (blue val {:.4f}, green test {:.4f})'.format(average_psnr_per_iteration_val[-1],average_psnr_per_iteration_test_shape_adjusted[-1]))
        else:
            if self.log_train_metrics:
                ax[1].set_title('PSNR (red train {:.4f}, blue val {:.4f})'.format(average_psnr_per_iteration_train[-1],average_psnr_per_iteration_val[-1]))
            else:
                ax[1].set_title('PSNR (blue val {:.4f})'.format(average_psnr_per_iteration_val[-1]))
        ax[1].grid(True)

        if self.log_train_metrics:
            ax[2].plot(iterations_lpips_ssim, average_lpips_per_iteration_train, 'r') # red
        ax[2].plot(iterations_lpips_ssim, average_lpips_per_iteration_val, 'b')
        if max(iterations) == self.max_iter:
            average_lpips_per_iteration_test_shape_adjusted = np.full_like(np.array(average_lpips_per_iteration_val), fill_value=average_lpips_per_iteration_test[-1])
            ax[2].plot(iterations_lpips_ssim, average_lpips_per_iteration_test_shape_adjusted, 'g')
            if self.log_train_metrics:
                ax[2].set_title('LPIPS (red train {:.4f}, blue val {:.4f}, green test {:.4f})'.format(average_lpips_per_iteration_train[-1],average_lpips_per_iteration_val[-1],average_lpips_per_iteration_test_shape_adjusted[-1]))
            else:
                ax[2].set_title('LPIPS (blue val {:.4f}, green test {:.4f})'.format(average_lpips_per_iteration_val[-1],average_lpips_per_iteration_test_shape_adjusted[-1]))
        else:
            if self.log_train_metrics:
                ax[2].set_title('LPIPS (red train {:.4f}, blue val {:.4f})'.format(average_lpips_per_iteration_train[-1],average_lpips_per_iteration_val[-1]))
            else:
                ax[2].set_title('LPIPS (blue val {:.4f})'.format(average_lpips_per_iteration_val[-1]))
        ax[2].grid(True)

        if self.log_train_metrics:
            ax[3].plot(iterations_lpips_ssim, average_ssim_per_iteration_train, 'r') # red
        ax[3].plot(iterations_lpips_ssim, average_ssim_per_iteration_val, 'b')
        # print(average_ssim_per_iteration_val)
        if max(iterations) == self.max_iter:
            average_ssim_per_iteration_test_shape_adjusted = np.full_like(np.array(average_ssim_per_iteration_val), fill_value=average_ssim_per_iteration_test[-1])
            ax[3].plot(iterations_lpips_ssim, average_ssim_per_iteration_test_shape_adjusted, 'g')
            if self.log_train_metrics:
                ax[3].set_title('SSIM (red train {:.4f}, blue val {:.4f}, green test {:.4f})'.format(average_ssim_per_iteration_train[-1],average_ssim_per_iteration_val[-1],average_ssim_per_iteration_test_shape_adjusted[-1]))
            else:
                ax[3].set_title('SSIM (blue val {:.4f}, green test {:.4f})'.format(average_ssim_per_iteration_val[-1],average_ssim_per_iteration_test_shape_adjusted[-1]))
        else:
            if self.log_train_metrics:
                ax[3].set_title('SSIM (red train {:.4f}, blue val {:.4f}'.format(average_ssim_per_iteration_train[-1],average_ssim_per_iteration_val[-1]))
            else:
                ax[3].set_title('SSIM (blue val {:.4f}'.format(average_ssim_per_iteration_val[-1]))
        ax[3].grid(True)

        plt.savefig(pathlib.Path(self.output_path,"metrics","metrics_{}.png".format(self.latest_iteration)))
        plt.close()


        lpips_ssim_iters = self.per_val_image_metrics[list(self.per_val_image_metrics.keys())[0]]['iterations_lpips_ssim']
        if max(iterations) == self.max_iter:
            iterations_test_lpips_ssim = self.per_test_image_metrics[list(self.per_test_image_metrics.keys())[0]]['iterations_lpips_ssim']
            average_loss_per_iteration_test_shape_adjusted = np.full_like(np.array(average_loss_per_iteration_val), fill_value=np.nan)
            average_loss_per_iteration_test_shape_adjusted[-1] = average_loss_per_iteration_test[-1]
            average_psnr_per_iteration_test_shape_adjusted = np.full_like(np.array(average_psnr_per_iteration_val), fill_value=np.nan)
            average_psnr_per_iteration_test_shape_adjusted[-1] = average_psnr_per_iteration_test[-1]
            average_lpips_per_iteration_test_shape_adjusted = np.full_like(np.array(average_lpips_per_iteration_val), fill_value=np.nan)
            average_lpips_per_iteration_test_shape_adjusted[-1] = average_lpips_per_iteration_test[-1]
            average_ssim_per_iteration_test_shape_adjusted = np.full_like(np.array(average_ssim_per_iteration_val), fill_value=np.nan)
            average_ssim_per_iteration_test_shape_adjusted[-1] = average_ssim_per_iteration_test[-1]

            if self.log_train_metrics:
                pd.DataFrame({
                    'iterations':iterations,
                    'train_psnr' : average_psnr_per_iteration_train,
                    'val_psnr' : average_psnr_per_iteration_val,
                    'test_psnr' : average_psnr_per_iteration_test_shape_adjusted,
                    'val_l1' : average_loss_per_iteration_val,
                    'test_l1' : average_loss_per_iteration_test_shape_adjusted,
                    'test_lpips' : self._adjust_lengths(iterations,average_lpips_per_iteration_test_shape_adjusted,iterations_test_lpips_ssim),
                    'train_ssim' : self._adjust_lengths(iterations,average_ssim_per_iteration_train,lpips_ssim_iters),
                    'val_ssim' : self._adjust_lengths(iterations,average_ssim_per_iteration_val,lpips_ssim_iters),
                    'test_ssim' : self._adjust_lengths(iterations,average_ssim_per_iteration_test_shape_adjusted,iterations_test_lpips_ssim),
                }).to_csv(pathlib.Path(self.output_path,"metrics","eval_{}.csv".format(self.latest_iteration)),index=False)
            else:
                pd.DataFrame({
                    'iterations':iterations,
                    'val_psnr' : average_psnr_per_iteration_val,
                    'test_psnr' : average_psnr_per_iteration_test_shape_adjusted,
                    'val_l1' : average_loss_per_iteration_val,
                    'test_l1' : average_loss_per_iteration_test_shape_adjusted,
                    'test_lpips' : self._adjust_lengths(iterations,average_lpips_per_iteration_test_shape_adjusted,iterations_test_lpips_ssim),
                    'val_ssim' : self._adjust_lengths(iterations,average_ssim_per_iteration_val,lpips_ssim_iters),
                    'test_ssim' : self._adjust_lengths(iterations,average_ssim_per_iteration_test_shape_adjusted,iterations_test_lpips_ssim),
                }).to_csv(pathlib.Path(self.output_path,"metrics","eval_{}.csv".format(self.latest_iteration)),index=False)

        else:
            if self.log_train_metrics:
                pd.DataFrame({
                    'iterations':iterations,
                    'train_psnr' : average_psnr_per_iteration_train,
                    'val_psnr' : average_psnr_per_iteration_val,
                    'train_l1' : average_loss_per_iteration_train,
                    'val_l1' : average_loss_per_iteration_val,
                    'lpips' : self._adjust_lengths(iterations,average_lpips_per_iteration_train,lpips_ssim_iters),
                    'ssim' : self._adjust_lengths(iterations,average_ssim_per_iteration_val,lpips_ssim_iters),
                }).to_csv(pathlib.Path(self.output_path,"metrics","eval_{}.csv".format(self.latest_iteration)),index=False)
            else:
                pd.DataFrame({
                    'iterations':iterations,
                    'val_psnr' : average_psnr_per_iteration_val,
                    'val_l1' : average_loss_per_iteration_val,
                    'lpips' : self._adjust_lengths(iterations,average_lpips_per_iteration_val,lpips_ssim_iters),
                    'ssim' : self._adjust_lengths(iterations,average_ssim_per_iteration_val,lpips_ssim_iters),
                }).to_csv(pathlib.Path(self.output_path,"metrics","eval_{}.csv".format(self.latest_iteration)),index=False)

    def _adjust_lengths(self,longer_iterations, shorter_list, shorter_iterations):
        for short_it in shorter_iterations:
            assert(short_it in longer_iterations, "Every iteration from the shorter list should be in the longer list")

        shorter_list_longer = []
        for iteration in longer_iterations:
            if iteration in shorter_iterations:
                shorter_list_longer.append(shorter_list[shorter_iterations.index(iteration)])
            else:
                shorter_list_longer.append(np.nan)
        return shorter_list_longer
                
        
        
    def _average_something(self,a_dict,average_target,position):
        sum = 0
        for key in a_dict.keys():
            sum += a_dict[key][average_target][position]
        return sum/len(a_dict.keys())
    
    def _get_l1_psnr(self,rgb_predicted,testimg):
        l1 = l1_loss(rgb_predicted,testimg)
        test_psnr = psnr(rgb_predicted, testimg).mean().double()
        return l1,test_psnr

    def _get_lpips_ssim(self,rgb_predicted,testimg):
        rgb_lpips = rgb_predicted
        tolerance = 1e-5
        try:
            rgb_predicted = rgb_predicted.reshape(-1,3)

            # lpips
            rgb_lpips = rgb_predicted.clone()
            # clip all values to [0,1]
            rgb_lpips = torch.clamp(rgb_lpips, 0, 1-tolerance)
        except Exception as e:
            pass

        rgb_lpips = rgb_lpips.reshape(testimg.shape[0],testimg.shape[1],-1)
        testimg = testimg.reshape(testimg.shape[0],testimg.shape[1],-1)

        target_img_lpips = testimg.clone()
        target_img_lpips = torch.clamp(target_img_lpips, 0, 1-tolerance)

        rgb_lpips = rgb_lpips.permute(2,0,1)
        target_img_lpips = target_img_lpips.permute(2,0,1)
        rgb_lpips = rgb_lpips.unsqueeze(0)
        target_img_lpips = target_img_lpips.unsqueeze(0)

        # lpips_fn_net.cuda()
        # rgb_lpips = rgb_lpips.to(torch.device("cpu"))
        # target_img_lpips = target_img_lpips.to(torch.device("cpu"))

        lpips_loss = self.lpips_fn_net(rgb_lpips,target_img_lpips).item()


        # ssim -> same structure of images as in lpips
        try:
            ssim_loss = self.ssim_fn(rgb_lpips,target_img_lpips).item()
        except AssertionError as e:
            # print("Assertion error in eval ssim_loss")
            # print(e)
            ssim_loss = original_ssim(rgb_lpips,target_img_lpips).item()

        return lpips_loss,ssim_loss



    def _setup_metric_dict(self,images):
        metric_dict = {}
        for img in images:
            # print(img.image_name)
            metric_dict[img.image_name] = {
                "iterations" : [],
                "psnr" : [],
                "l1" : [],
                "iterations_lpips_ssim" : [],
                "lpips" : [],
                "ssim" :[],
                'depth_l1' : [],
                'depth_psnr' : [],
                'depth_lpips' : [],
                'depth_ssim' : [],
            }


        return metric_dict

    def _make_dirs(self,split,images,output_path):
        if not os.path.exists(pathlib.Path(output_path,"metrics")):
            os.makedirs(pathlib.Path(output_path,"metrics")) 
        if not os.path.exists(pathlib.Path(output_path,"metrics",split)):
            os.makedirs(pathlib.Path(output_path,"metrics",split)) 
        for image in images:
            os.makedirs(pathlib.Path(output_path,"metrics",split,"img_{}".format(image.image_name)),exist_ok=True)