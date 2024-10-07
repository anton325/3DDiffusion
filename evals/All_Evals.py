import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



class AllEvals:
    def __init__(self, dir, batch_size) -> None:
        self.ssims = []
        self.lpipss = []
        self.losses = []
        self.likelihoods = []
        self.psnrs = []
        self.chamfers = []
        self.dir = dir
        self.batch_size = batch_size

        self.ssim_circle = []
        self.lpips_circle = []
        self.losses_circle = []
        self.psnrs_circle = []

        self.ssim_srn = []
        self.lpips_srn = []
        self.losses_srn = []
        self.psnrs_srn = []

        self.psnrs_gaussian_gt = []
        self.ssims_gaussian_gt = []
        self.lpips_gaussian_gt = []
        self.losses_gaussian_gt = []

        self.category_psnr = {}
        self.category_ssim = {}
        self.category_lpips = {}
        self.category_loss = {}
    
    def add_srn_eval_category(self, category, psnr, ssim, lpips, loss):
        if category not in self.category_psnr:
            self.category_psnr[category] = []
            self.category_ssim[category] = []
            self.category_lpips[category] = []
            self.category_loss[category] = []
        self.category_psnr[category].append(psnr)
        self.category_ssim[category].append(ssim)
        self.category_lpips[category].append(lpips)
        self.category_loss[category].append(loss)

    def add_srn_eval(self, psnr, ssim, lpips, loss):
        self.ssim_srn.extend(ssim)
        self.lpips_srn.extend(lpips)
        self.losses_srn.extend(loss)
        self.psnrs_srn.extend(psnr)

    def init_ctx_eval(self, n):
        self.psnrs_ctx_angle = [[] for _ in range(n)]
        self.ssims_ctx_angle = [[] for _ in range(n)]
        self.lpipss_ctx_angle = [[] for _ in range(n)]
        self.losses_ctx_angle = [[] for _ in range(n)]

    def add_ctx_eval(self, psnr, ssim, lpips, loss, i):
        self.psnrs_ctx_angle[i].extend(psnr)
        self.ssims_ctx_angle[i].extend(ssim)
        self.lpipss_ctx_angle[i].extend(lpips)
        self.losses_ctx_angle[i].extend(loss)

    def init_ctx_eval_autoregressive(self, n):
        self.psnrs_ctx_angle_autoregressive = [[] for _ in range(n)]
        self.ssims_ctx_angle_autoregressive = [[] for _ in range(n)]
        self.lpipss_ctx_angle_autoregressive = [[] for _ in range(n)]
        self.losses_ctx_angle_autoregressive = [[] for _ in range(n)]

    def add_ctx_eval_autoregressive(self, psnr, ssim, lpips, loss, i):
        self.psnrs_ctx_angle_autoregressive[i].extend(psnr)
        self.ssims_ctx_angle_autoregressive[i].extend(ssim)
        self.lpipss_ctx_angle_autoregressive[i].extend(lpips)
        self.losses_ctx_angle_autoregressive[i].extend(loss)

    def init_circle_eval(self, n):
        self.psnrs_circle_angle = [[] for _ in range(n)]
        self.ssims_circle_angle = [[] for _ in range(n)]
        self.lpipss_circle_angle = [[] for _ in range(n)]
        self.losses_circle_angle = [[] for _ in range(n)]

    def add_circle_eval(self, psnr, ssim, lpips, loss, i):
        self.psnrs_circle_angle[i].extend(psnr)
        self.ssims_circle_angle[i].extend(ssim)
        self.lpipss_circle_angle[i].extend(lpips)
        self.losses_circle_angle[i].extend(loss)

    def append(self, ssim, lpips, loss, psnr):
        self.ssims.append(ssim)
        self.lpipss.append(lpips)
        self.losses.append(loss)
        self.psnrs.append(psnr)
    
    def append_gaussian_gt(self, ssim, lpips, loss, psnr):
        self.ssims_gaussian_gt.append(ssim)
        self.lpips_gaussian_gt.append(lpips)
        self.losses_gaussian_gt.append(loss)
        self.psnrs_gaussian_gt.append(psnr)

    def append_likelihood(self, likelihood):
        self.likelihoods.append(likelihood)
    
    def append_chamfer(self, chamfer):
        self.chamfers.append(chamfer)
    
    def plot_srn(self):
        fig,ax = plt.subplots()
        ax.hist(self.psnrs_srn, label = "PSNR")
        ax.set_title(f"PSNR (mean {round(sum(self.psnrs_srn) / len(self.psnrs_srn),3)}), std: {round(np.std(self.psnrs_srn),2)}")
        ax.set_xlabel("PSNR")
        ax.set_ylabel("Count")
        fig.savefig(self.dir / "psnr_srn_hist.png")
        fig,ax = plt.subplots()
        ax.hist(self.ssim_srn, label = "SSIM")
        ax.set_title(f"SSIM (mean {round(sum(self.ssim_srn) / len(self.ssim_srn),3)}), std: {round(np.std(self.ssim_srn),2)}")
        ax.set_xlabel("SSIM")
        ax.set_ylabel("Count")
        fig.savefig(self.dir / "ssim_srn_hist.png")
        fig,ax = plt.subplots()
        ax.hist(self.lpips_srn, label = "LPIPS")
        ax.set_title(f"LPIPS (mean {round(sum(self.lpips_srn) / len(self.lpips_srn),3)}), std: {round(np.std(self.lpips_srn),2)}")
        ax.set_xlabel("LPIPS")
        ax.set_ylabel("Cound")
        fig.savefig(self.dir / "lpips_srn_hist.png")
        fig,ax = plt.subplots()
        ax.hist(self.losses_srn, label = "l1")
        ax.set_title(f"L1 (mean {round(sum(self.losses_srn) / len(self.losses_srn),3)}), std: {round(np.std(self.losses_srn),2)}")
        ax.set_xlabel("L1")
        ax.set_ylabel("Count")
        fig.savefig(self.dir / "l1_srn_hist.png")


    
    def summarize(self):
        summary_dict = {'iterations' : [10000],
                        'test_psnr' : [sum(self.psnrs) / len(self.psnrs)],
                        'test_l1' : [sum(self.losses) / len(self.losses)],
                        'test_lpsips' : [sum(self.lpipss) / len(self.lpipss)],
                        'test_ssim' : [sum(self.ssims) / len(self.ssims)],
        }
        try:
            summary_dict['circle_psnr'] = [sum([sum(x) for x in self.psnrs_circle_angle]) / sum([len(x) for x in self.psnrs_circle_angle])]
            summary_dict['circle_l1'] = [sum([sum(x) for x in self.losses_circle_angle]) / sum([len(x) for x in self.losses_circle_angle])]
            summary_dict['circle_lpsips'] = [sum([sum(x) for x in self.lpipss_circle_angle]) / sum([len(x) for x in self.lpipss_circle_angle])]
            summary_dict['circle_ssim'] = [sum([sum(x) for x in self.ssims_circle_angle]) / sum([len(x) for x in self.ssims_circle_angle])]
        except:
            print("Error all evals summarize: circle eval werte sind nicht vorhanden")
        
        try:
            summary_dict['circle_psnr_autoregressive'] = [sum([sum(x) for x in self.psnrs_ctx_angle_autoregressive]) / sum([len(x) for x in self.psnrs_ctx_angle_autoregressive])]
            summary_dict['circle_l1_autoregressive'] = [sum([sum(x) for x in self.losses_circle_angle_autoregressive]) / sum([len(x) for x in self.losses_ctx_angle_autoregressive])]
            summary_dict['circle_lpsips_autoregressive'] = [sum([sum(x) for x in self.lpipss_circle_angle_autoregressive]) / sum([len(x) for x in self.lpipss_circle_angle_autoregressive])]
            summary_dict['circle_ssim_autoregressive'] = [sum([sum(x) for x in self.ssims_circle_angle_autoregressive]) / sum([len(x) for x in self.ssims_circle_angle_autoregressive])]
        except:
            print("Error all evals summarize: circle eval autoregressive werte sind nicht vorhanden")
        try:
            summary_dict['srn_psnr'] = [sum(self.psnrs_srn) / len(self.psnrs_srn)]
            summary_dict['srn_l1'] = [sum(self.losses_srn) / len(self.losses_srn)]
            summary_dict['srn_lpsips'] = [sum(self.lpips_srn) / len(self.lpips_srn)]
            summary_dict['srn_ssim'] = [sum(self.ssim_srn) / len(self.ssim_srn)]
        except:
            print("Error all evals summarize srn: manche werte sind nicht vorhanden")
        
        try:
            self.plot_srn()
        except:
            print("Error all evals summarize srn: plot nicht möglich")
        
        try:
            summary_dict['psnr_gaussian_gt'] = [sum(self.psnrs_gaussian_gt) / len(self.psnrs_gaussian_gt)]
            summary_dict['l1_gaussian_gt'] = [sum(self.losses_gaussian_gt) / len(self.losses_gaussian_gt)]
            summary_dict['lpips_gaussian_gt'] = [sum(self.lpips_gaussian_gt) / len(self.lpips_gaussian_gt)]
            summary_dict['ssim_gaussian_gt'] = [sum(self.ssims_gaussian_gt) / len(self.ssims_gaussian_gt)]
        except:
            print("Error all evals summarize gaussian gt: manche werte sind nicht vorhanden")

        df = pd.DataFrame(summary_dict)
        if len(self.likelihoods) == int(len(self.ssims) / (self.batch_size*5)) and len(self.likelihoods) > 0: 
            df['likelihood'] = [sum(self.likelihoods) / len(self.likelihoods)]

        if len(self.chamfers) == int(len(self.ssims) / (self.batch_size*5)) and len(self.chamfers) > 0: 
            df['chamfer_xyz'] = [sum(self.chamfers) / len(self.chamfers)]
        # print(df)
        df.to_csv(self.dir / "eval_10000.csv",
                  index=False)

        df = pd.DataFrame()
        psnrs = []
        ssims = []
        lpipss = []
        losses = []
        categories = []
        for key in self.category_psnr.keys():
            categories.append(key)
            psnrs.append(sum(self.category_psnr[key]) / len(self.category_psnr[key]))
            ssims.append(sum(self.category_ssim[key]) / len(self.category_ssim[key]))
            lpipss.append(sum(self.category_lpips[key]) / len(self.category_lpips[key]))
            losses.append(sum(self.category_loss[key]) / len(self.category_loss[key]))
        df['category'] = categories
        df['psnr'] = psnrs
        df['ssim'] = ssims
        df['lpips'] = lpipss
        df['l1'] = losses
        df.to_csv(self.dir / "category_eval_10000.csv",
                  index=False)

    def plot_ctx_eval(self):
        def sum_up(values):
            return [sum(x)/len(x) for x in values]
        fig,ax = plt.subplots()
        ax.plot(sum_up(self.psnrs_ctx_angle), label = "psnr")
        ax.set_title("PSNR w.r.t. angle")
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("PSNR")
        fig.savefig(self.dir / "psnr_ctx.png")
        fig,ax = plt.subplots()
        ax.plot(sum_up(self.ssims_ctx_angle), label = "ssim")
        ax.set_title("SSIM w.r.t. angle")
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("SSIM")
        fig.savefig(self.dir / "ssim_ctx.png")
        fig,ax = plt.subplots()
        ax.plot(sum_up(self.lpipss_ctx_angle), label = "lpips")
        ax.set_title("LPIPS w.r.t. angle")
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("LPIPS")
        fig.savefig(self.dir / "lpips_ctx.png")
        fig,ax = plt.subplots()
        ax.plot(sum_up(self.losses_ctx_angle), label = "l1")
        ax.set_title("L1 w.r.t. angle")
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("L1")
        fig.savefig(self.dir / "l1_ctx.png")

    def plot_ctx_eval_autoregressive(self):
        def sum_up(values):
            return [sum(x)/len(x) for x in values]
        fig,ax = plt.subplots()
        ax.plot(sum_up(self.psnrs_ctx_angle_autoregressive), label = "psnr")
        ax.set_title("PSNR w.r.t. angle")
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("PSNR")
        fig.savefig(self.dir / "psnr_ctx_autoregressive.png")
        fig,ax = plt.subplots()
        ax.plot(sum_up(self.ssims_ctx_angle_autoregressive), label = "ssim")
        ax.set_title("SSIM w.r.t. angle")
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("SSIM")
        fig.savefig(self.dir / "ssim_ctx_autoregressive.png")
        fig,ax = plt.subplots()
        ax.plot(sum_up(self.lpipss_ctx_angle_autoregressive), label = "lpips")
        ax.set_title("LPIPS w.r.t. angle")
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("LPIPS")
        fig.savefig(self.dir / "lpips_ctx_autoregressive.png")
        fig,ax = plt.subplots()
        ax.plot(sum_up(self.losses_ctx_angle_autoregressive), label = "l1")
        ax.set_title("L1 w.r.t. angle")
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("L1")
        fig.savefig(self.dir / "l1_ctx_autoregressive.png")