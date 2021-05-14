import argparse
import numpy as np
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

if not os.path.exists("saved/figures"): os.makedirs("saved/figures")

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    _, data_loader = config.init_obj('data_loader', module_data).get_loaders()

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns), device=device)

    count = 0
    plt.ion()

    red_patch = mlines.Line2D([], [], color='red', marker='o',
                              markersize=15, label='Target data')
    green_patch = mlines.Line2D([], [], color='green', marker='*',
                          markersize=15, label='Prediction')
    plt.legend(handles=[red_patch, green_patch])
    plt.xlabel('Time step')
    plt.ylabel('Price')
    plt.title("Doge Prediction")
    plt.show()

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

            # Printing out
            y = torch.clone(output).cpu().squeeze().numpy()
            y2 = torch.clone(target).cpu().squeeze().numpy()
            x = np.add(count, np.arange(len(y)))
            count += len(y)

            plt.plot(x, y, c='green', marker="o")
            plt.plot(x, y2, c='red', marker="*")

            plt.savefig('saved/figures/plot_'+str(count)+'.jpg', dpi=300, bbox_inches='tight')
            plt.draw()
            plt.pause(0.2)

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)

    main(config)
    input("Press [enter] to continue.")