from config import config
from utils import reader, csv_processor
import torch
import numpy as np
from utils.metrics import metric


def csv_process(file):
    return csv_processor.excel_processor(file)


def predict_common(files, model, dry_run):
    final_file = csv_process(files['source'])
    if dry_run:
        batch_x, batch_x_mark, true_x = reader.convert(final_file, config, dry_run=dry_run, start=1080)
        true_x = torch.tensor(true_x).unsqueeze(0)
        true_x = true_x.float().to(torch.device('cpu'))
    else:
        batch_x, batch_x_mark = reader.convert(final_file, config, dry_run=dry_run, start=0)

    batch_x = torch.tensor(batch_x).unsqueeze(0)
    batch_x_mark = torch.tensor(batch_x_mark).unsqueeze(0)

    with torch.no_grad():
        batch_x = batch_x.float().to(torch.device('cpu'))
        batch_x_mark = batch_x_mark.float().to(torch.device('cpu'))

        # decoder input
        dec_inp = torch.zeros_like(batch_x[:, -config.pred_len:, :]).float()
        # encoder - decoder
        if config.use_amp:
            with torch.cuda.amp.autocast():
                if config.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_x_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_x_mark)
        else:
            if config.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_x_mark)[0]

            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_x_mark)

        f_dim = -1 if config.features == 'MS' else 0
        outputs = outputs[:, -config.pred_len:, :]
        outputs = outputs.detach().cpu().numpy()
        outputs = outputs[:, :, f_dim:]
        pred = reader.flatten_list(outputs.tolist())

        if not dry_run:
            res = {
                'length': len(pred),
                'pred': pred
            }
            return res
        else:
            true_x = true_x.to(torch.device('cpu'))
            true_x = true_x.detach().cpu().numpy()
            true_x = true_x[:, :, f_dim:]
            true = reader.flatten_list(true_x.tolist())

            pred_np = np.array(outputs)
            true_np = np.array(true_x)
            pred_np = pred_np.reshape(-1, pred_np.shape[-2], pred_np.shape[-1])
            true_np = true_np.reshape(-1, true_np.shape[-2], true_np.shape[-1])
            mae, mse, rmse, mape, mspe = metric(pred_np, true_np)

            res = {
                'length': len(pred) if len(pred) == len(true) else -1,
                'pred': pred,
                'true': true,
                'mae': mae.tolist(),
                'mse': mse.tolist(),
                'rmse': rmse.tolist(),
                'mape': mape.tolist(),
                'mspe': mspe.tolist()
            }
            return res
