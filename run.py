
from convlstm import ConvLSTM
from core.models.predrnn import RNN as predrnn
from core.models.predrnn_v2 import RNN as predrnn2
from e3d_lstm import E3DLSTM
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
from utils import generate_movies
import argparse
from pathlib import Path
import json
# import matplotlib.pyplot as plt

def main(args, device):

    dataset_name = args.dataset_name
    model_name = args.model_name

    ###### DATASET SELECTION ######
    if dataset_name == 'simulated':
        data = np.load('./data/standard_simulated_data_update.npy')
    elif dataset_name == 'real':
        data = np.load('./data/real_data_update.npy')
        # use only the temperature band
        data = data[:, :, [0], :]
    else:
        raise Exception("Dataset name not found, use 'simulated' or 'real'.")
    
    # Normalize
    max = data.max()
    min = data.min()
    data = (data - min) / max
    print(data.shape, data.min(), data.max())

    x, y, c, t = data.shape
    data = torch.from_numpy(data).float()
    data = data.permute(3, 2, 0, 1)
    # t, c, x, y
    # data = data.reshape(t, c, x, y)

    split_time = int(t * 0.8)
    train_data = data[:split_time]
    test_data = data[split_time:]

    # reshape to input into model
    # split long sequence into smaller sequences
    batch_size = 4
    splits = 20
    segments = int(t / splits)

    def preprocess(d):

        ground = []
        shift = []

        for i in range(splits):
            try:
                ground.append(d[i*segments:(i+1)*segments])
                # shift the ground truth by one frame
                shift.append(d[i*segments+1:(i+1)*segments+1])
                # print(d[i*segments+1:(i+1)*segments+1].shape)
            except:
                ground = ground[:-1]
        # print(len(ground), len(shift))
            
        ground = [ground[x:x+batch_size] for x in range(0, len(ground), batch_size)]
        shift = [shift[x:x+batch_size] for x in range(0, len(shift), batch_size)] 
        # print(len(ground), len(ground[0]), len(shift), len(shift[0]))

        return ground, shift

    ground, shift = preprocess(train_data)
    val_ground, val_shift = preprocess(test_data)

    # for x, y in zip(ground, shift):
    # h, g = 0, 0
    # # # # for x, y in zip(ground, None):
    # for x in val_ground:
    #     print(x[0].shape)
    # # #     # for i, j in zip(x, y):
    #     h += x[0].shape[0]
    # # #         # g += j.shape[0]
    # print(h, g)
    # exit()

    # noisy_movies, shifted_movies = generate_movies(n_samples=100)
    # b, t, x, y, c = noisy_movies.shape
    # noisy_movies = torch.from_numpy(noisy_movies.reshape(b, t, c, x, y)).float().cuda()
    # shifted_movies = torch.from_numpy(shifted_movies.reshape(b, t, c, x, y)).float().cuda()
    args.img_width = x
    args.img_height = y
    args.img_channel = c
    ###### MODEL SELECTION ######
    if model_name == 'convlstm':
        # Initialize training
        model = ConvLSTM(input_dim=c,
                        hidden_dim=[32, c],
                        kernel_size=(3, 3),
                        num_layers=2,
                        batch_first=True,
                        bias=True,
                        return_all_layers=False).to(device)
    elif model_name == 'predrnn':
        num_hidden = [int(x) for x in args.num_hidden.split(',')]
        model = predrnn(num_layers=len(num_hidden),
                        num_hidden=num_hidden,
                        configs=args).to(device)
        model.frame_channel = x * y * c
    elif model_name == 'predrnn2':
        num_hidden = [int(x) for x in args.num_hidden.split(',')]
        model = predrnn2(num_layers=len(num_hidden),
                        num_hidden=num_hidden,
                        configs=args).to(device)
        model.frame_channel = x * y * c
    elif model_name == 'e3d':
        model = E3DLSTM(
            input_shape = (c, 4, x, y), # t = 4
            hidden_size = c, # 4
            num_layers = 2,
            kernel_size = (1, 1, 1),
            tau = 2
        ).to(device)
    else:
        raise Exception("Model name not found.")

    ###### TRAINING SETUP ######

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    last_loss = 0.
    rmse_losses = []
    ad_losses = []
    train_losses = []

    epochs = args.num_epochs

    def switcher(s, g):
        # if dataset_name == 'simulated':
        #     if model_name == 'convlstm':
        #         target = s[:, 1:]
        #     else:
        #         target = s[1:]
        # elif dataset_name == 'real':
        #     target = s
        #     if model_name == 'predrnn' or model_name == 'predrnn2':
        #         target = s[1:]
        #     else:
        #         target = s
        #         # target = s[1:]
        #         # g = g[1:]
                # pass
        target = s
        if model_name =='e3d':
            target = target.permute(0, 2, 1, 3, 4)
            g = g.permute(0, 2, 1, 3, 4)[1:]

        return target, g

    pbar = tqdm(range(epochs), desc = 'description')
    for _ in pbar:

        model.train()
        bands_error = [[0, 0],[0, 0],[0, 0],[0, 0]]
        running_loss = 0.
        val_loss = 0.
        ad_loss = 0.

        h, j = 0, 0

        for b_g, b_s in zip(ground, shift):
            # g = torch.from_numpy(np.array(b_g)).float().to(device)
            # g = g.reshape(len(b_g), segments, c, x, y)
            # s = torch.from_numpy(np.array(b_s)).float().to(device)
            # s = s.reshape(len(b_s), segments, c, x, y)
            # print(b_g[2].shape)
            try:
                g = torch.stack(b_g, dim = 0).to(device)
                s = torch.stack(b_s, dim = 0).to(device)
                target, g = switcher(s, g)
        
                # Zero your gradients for every batch!
                optimizer.zero_grad()
                # print(target.shape, g.shape)
                # Make predictions for this batch
                outputs = model(g)#[:,1:]
                # outputs = model(g)
                # print(outputs.shape, s[:, 1:].shape, s.shape)
                if target.size(1) != outputs.size(1):
                    outputs = outputs[:, 1:]
                else:
                    target = target[1:]

                # Compute the loss and its gradients
                loss = loss_fn(outputs, target)

                # loss = loss_fn(outputs[0][0], s)
                loss.backward()

                # Adjust learning weights
                optimizer.step()

                # Gather data and report
                # Divide by number of sequences
                running_loss += loss.item() / len(b_g)
                h += outputs.size(0) * outputs.size(1)
            except: pass   
        
        model.eval()
        preds = []
        gts = []

        for v_g, v_s in zip(val_ground, val_shift):
            # Seems to be an inhomogenous array as some point
            # print(len(v_g), v_g[0].shape, v_g[1].shape)
            try:
            # g = torch.from_numpy(np.array(v_g)).float().to(device)
            # # g = g.reshape(len(v_g), segments, c, x, y)
            # s = torch.from_numpy(np.array(v_s)).float().to(device)
            # s = s.reshape(len(v_s), segments, c, x, y)

                g = torch.stack(v_g, dim = 0).to(device)
                s = torch.stack(v_s, dim = 0).to(device)

                with torch.no_grad():
                    # outputs = model(g)
                    target, g = switcher(s, g)

                    outputs = model(g) #[:, 1:]
                    
                    if target.size(1) != outputs.size(1):
                        outputs = outputs[:, 1:]
                    else:
                        target = target[1:]

                    j += outputs.size(0) * outputs.size(1)

                    inp = outputs * max + min
                    tar = target * max + min
                    val_loss += (loss_fn(inp, tar).item() ** 0.5) / len(v_g)
                    ad_loss += F.l1_loss(inp, tar).item() / len(v_g)

                    # get error for each band
                    if dataset_name == 'simulated':
                        for i, _ in enumerate(bands_error):
                            # print(i, inp.shape, tar.shape)
                            inp_b = inp[:, :, i]
                            tar_b = tar[:, :, i]
                            bands_error[i][0] += (loss_fn(inp_b, tar_b).item() ** 0.5) / len(v_g)
                            bands_error[i][1] += F.l1_loss(inp_b, tar_b).item() / len(v_g)
            
                    if model_name == 'e3d':
                        # convert back to standard
                        inp = inp.permute(0, 2, 1, 3, 4)
                        tar = tar.permute(0, 2, 1, 3, 4)
                    # tar, inp = switcher(tar, inp)

                    preds.append(inp.cpu().numpy())
                    gts.append(tar.cpu().numpy())

            except: pass

        train_losses.append(running_loss)
        rmse_losses.append(val_loss)
        ad_losses.append(ad_loss)
        # print("Epoch", epoch, "Loss", running_loss, "Time", end - start)
        pbar.set_description(f'VRMSE = {val_loss:.5f}, VMAE = {ad_loss:.5f}')
        time.sleep(0.01)

    print(h, j)

    print("sim errors", bands_error)

    save_results(train_loss_arr=train_losses, 
                 test_mae_arr=ad_losses,
                 test_rmse_arr=rmse_losses,
                 predicted_result=preds,
                 gt_result=gts,
                 args=args)

def save_results(train_loss_arr, 
                 test_mae_arr, 
                 test_rmse_arr,
                 predicted_result,
                 gt_result,
                 args):

    # Results
    results = {}
    results['Train_Loss'] = train_loss_arr
    results['Test_MAE'] = test_mae_arr
    results['Test_RMSE'] = test_rmse_arr
    results['Result_Ground'] = gt_result
    results['Result_Pred'] = predicted_result
    results['args'] = vars(args)

    experiment_name = args.experiment_name
    model_name = args.model_name
    # run_name = args.run_name
    dataset_name = args.dataset_name
    lr = args.learning_rate
    epochs = args.num_epochs

    f_name = "./results/{}/{}_{}_lr{}_ep{}.pkl".format(experiment_name, model_name, dataset_name, lr, epochs)

    # Check if directory exists
    Path("./results/{}".format(experiment_name)).mkdir(parents=True, exist_ok=True)

    import pickle

    # Save results
    with open(f_name, 'wb') as f:
        # json.dump(results, f)
        pickle.dump(results, f)

    print("Results saved to: {}".format(f_name))

    return

def load_results(f_name):

    # Load the dictionary results
    with open(f_name, 'rb') as f:
        results = json.load(f)

    return results


if __name__ == "__main__":

    ### ARGUMENTS ###
    parser = argparse.ArgumentParser(
        description="Pipeline"
    )
    parser.add_argument("-n", "--num-epochs", default=50, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.1, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--model-name", default="model_name", type=str)
    parser.add_argument("--experiment-name", default="experiment_name", type=str)
    parser.add_argument("--run-name", default="run_name", type=str)
    parser.add_argument("--dataset-name", default="fire", type=str)

    # predrnn
    parser.add_argument('--input_length', type=int, default=4)
    parser.add_argument('--total_length', type=int, default=4)
    parser.add_argument('--img_width', type=int, default=225)
    parser.add_argument('--img_height', type=int, default=225)
    parser.add_argument('--img_channel', type=int, default=4)
    parser.add_argument('--num_hidden', type=str, default='32')
    parser.add_argument('--filter_size', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--layer_norm', type=int, default=1)

    # GPU Check
    # enable cuda gpu accelaration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.set_printoptions(suppress=True)
    print("GPU CHECK:", device)
    parser.add_argument('--device', type=str, default="cuda:0")

    args = parser.parse_args()
    
    main(args, device)