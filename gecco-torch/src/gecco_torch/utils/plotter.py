import matplotlib.pyplot as plt
import pathlib
import pickle
import json


def plot_speed_vs_gaussians(num_iterations,gaussians,its_per_sec,path):
    fig, ax1 = plt.subplots(1, 1, figsize=(35,4), gridspec_kw={'width_ratios': [1]}) # only show image
    # ax1.plot(range(num_iterations),gaussians)
    ax1.plot(range(len(gaussians)),gaussians)
    ax1.set_ylabel('Number of gaussians per iteration', color='b')
    ax2 = ax1.twinx()
    ax2.plot(range(len(its_per_sec)),its_per_sec,color='r')
    # ax2.plot(range(num_iterations),its_per_sec)
    ax2.set_ylabel('Number of iterations per second', color='r')
    total_time = sum([1/freq for freq in its_per_sec])
    plt.title(f"Speed and number of gaussians in iteration (total time {total_time:.2f}s) (Final Number of gaussians: {gaussians[-1]})")
    plt.savefig(pathlib.Path(path,"gaussians_speed.png"))
    plt.close()
    with open(pathlib.Path(path,"num_gaussians.json"),"w") as f:
        json.dump(gaussians,f)
    
def plot_num_gaussians(gaussians,path):
    fig, ax1 = plt.subplots(1, 1, figsize=(35,4), gridspec_kw={'width_ratios': [1]}) # only show image
    # ax1.plot(range(num_iterations),gaussians)
    ax1.plot(range(len(gaussians)),gaussians)
    ax1.set_ylabel('Number of gaussians per iteration', color='b')
    plt.title(f"Number of gaussians per iteration (Final Number of gaussians: {gaussians[-1]})")
    plt.savefig(pathlib.Path(path,"gaussians.png"))
    plt.close()
    with open(pathlib.Path(path,"num_gaussians.json"),"w") as f:
        json.dump(gaussians,f)

def plot_time_vs_quality(num_iterations,its_per_sec,eval_iters,val_losses,val_psnrs,path):
    fig, ax1 = plt.subplots(1, 1, figsize=(35,4), gridspec_kw={'width_ratios': [1]}) # only show image
    eval_iters = [0] + eval_iters
    val_losses_filled = [val_losses[i].cpu() for i in range(len(val_losses)) for _ in range(eval_iters[i+1]-eval_iters[i])]
    val_psnrs_filled = [val_psnrs[i].cpu() for i in range(len(val_psnrs)) for _ in range(eval_iters[i+1]-eval_iters[i])]

    time_per_iteration = [1/i for i in its_per_sec]
    
    times = [sum(time_per_iteration[:i]) for i in range(len(val_losses_filled))]

    ax1.plot(times,val_losses_filled,color='b')
    ax1.set_ylabel('Validation loss', color='b')
    ax1.set_xlabel('Time (s)')
    ax2 = ax1.twinx()
    ax2.plot(times,val_psnrs_filled,color='r')
    ax2.set_ylabel('Validation PSNR', color='r')
    total_time = sum(time_per_iteration)
    plt.title(f"Validation loss and PSNR wrt train time (total time {total_time:.2f}s)")
    plt.savefig(pathlib.Path(path,"time_losses.png"))
    plt.close()

def plot_iterations_vs_quality(num_iterations,its_per_sec,eval_iters,val_losses,val_psnrs,path):
    fig, ax1 = plt.subplots(1, 1, figsize=(35,4), gridspec_kw={'width_ratios': [1]}) # only show image
    eval_iters = [0] + eval_iters
    val_losses_filled = [val_losses[i].cpu() for i in range(len(val_losses)) for _ in range(eval_iters[i+1]-eval_iters[i])]
    val_psnrs_filled = [val_psnrs[i].cpu() for i in range(len(val_psnrs)) for _ in range(eval_iters[i+1]-eval_iters[i])]

    time_per_iteration = [1/i for i in its_per_sec]
    
    # ax1.plot(range(num_iterations),val_losses_filled,color='b')
    ax1.plot(range(len(val_losses_filled)),val_losses_filled,color='b')
    ax1.set_ylabel('Validation loss', color='b')
    ax1.set_xlabel('Iteration')
    ax2 = ax1.twinx()
    ax2.plot(range(len(val_psnrs_filled)),val_psnrs_filled,color='r')
    # ax2.plot(range(num_iterations),val_psnrs_filled,color='r')
    ax2.set_ylabel('Validation PSNR', color='r')
    total_time = sum(time_per_iteration)
    plt.title(f"Validation loss and PSNR wrt iteration (total time {total_time:.2f}s)")
    plt.savefig(pathlib.Path(path,"iter_losses.png"))
    plt.close()

def plot_xyz_lrs(xyz_lrs,path):
    fig, ax1 = plt.subplots(1, 1, figsize=(35,4), gridspec_kw={'width_ratios': [1]}) # only show image
    ax1.plot(xyz_lrs,color='b')
    ax1.set_ylabel('lr', color='b')
    ax1.set_xlabel('Iteration')
    plt.title(f"Learning rate")
    plt.savefig(pathlib.Path(path,"xyz_lr.png"))
    plt.close()

def testplot():
    num = 100000
    y = [3] * num
    x = range(num)
    fig, ax1 = plt.subplots(1, 1, figsize=(35,4), gridspec_kw={'width_ratios': [1]}) # only show image
    ax1.plot(x,y)
    plt.show()

if __name__ == "__main__":
    testplot()