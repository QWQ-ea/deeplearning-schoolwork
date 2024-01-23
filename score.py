import os
import numpy as np

pred_path = "./pred"
ground_truth_path = "groundtruth.h5"
pred_name = "Prediction_512.h5"

def read_data(data_path):
    import h5py

    data_out = None
    data = h5py.File(data_path, "r")
    for key in data.keys():
        t = float(key)
        data_t = np.array(data[key])
        data_t_X = data_t[:, 0:2]
        data_t_X = np.insert(data_t_X, 2, np.ones(data_t_X.shape[0]) * t, axis=1)
        data_t_y = data_t[:, 2:]
        if data_out is None:
            data_out = np.hstack((data_t_X, data_t_y))
        else:
            data_out = np.vstack((data_out, np.hstack((data_t_X, data_t_y))))

    return data_out

def sqrt_sum_squared_error(data):
    return np.sqrt(np.sum(data ** 2))

def l2_loss(pred, gt):
    return sqrt_sum_squared_error(pred - gt)/sqrt_sum_squared_error(gt)

def score(pred, gt):
    error = 0
    for i in [3, 4, 5]:
        error += l2_loss(pred[:, i], gt[:, i])
    return error/3

def main():
    scores = []
    ground_truth = read_data(ground_truth_path)
    subdirectories = os.listdir(pred_path)

    for subdirectory in subdirectories:
        h5_file = os.path.join(pred_path, subdirectory)
        h5_file = os.path.join(h5_file, pred_name)
        if os.path.isfile(h5_file):
            data = read_data(h5_file)
            score_value = score(data, ground_truth)
            scores.append(score_value)

    import pandas as pd

    df_scores = pd.DataFrame({"Folder": subdirectories, "Score": scores})
    output_file = "scores2.xlsx"
    df_scores.to_excel(output_file, index=False)


main()
