import numpy as np


def threshold_percentage_np(output, target, threshold_val, valid_mask=None, eps=1e-10):
    d1 = output / (target + eps)
    d2 = target / (output + eps)
    max_d1_d2 = np.maximum(d1, d2)
    bit_mat = np.where(max_d1_d2 < threshold_val, 1, 0)

    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = np.sum(valid_mask, axis=(-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]

    count_mat = np.sum(bit_mat, axis=(-1, -2))
    threshold_mat = count_mat / n  # Prevent division by zero
    return np.mean(threshold_mat)


def delta1_acc_np(pred, gt, valid_mask=None):
    return threshold_percentage_np(pred, gt, 1.25, valid_mask)


def delta2_acc_np(pred, gt, valid_mask=None):
    return threshold_percentage_np(pred, gt, 1.25 ** 2, valid_mask)


def delta3_acc_np(pred, gt, valid_mask=None):
    return threshold_percentage_np(pred, gt, 1.25 ** 3, valid_mask)


def abs_relative_difference_np(output, target, valid_mask=None, eps=1e-10):
    abs_relative_diff = np.abs(output - target) / (target + eps)
    # print(f"shape of abs_relative_diff: {abs_relative_diff.shape}")
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = np.sum(valid_mask, axis=(-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]

    abs_relative_diff = np.sum(abs_relative_diff, axis=(-1, -2)) / n  # Prevent division by zero
    # print(f"value of abs_relative_diff: {abs_relative_diff}")
    return np.mean(abs_relative_diff)


def mse_np(output, target, valid_mask=None):
    mse = (output - target) ** 2

    if valid_mask is not None:
        mse[~valid_mask] = 0
        n = np.sum(valid_mask, axis=(-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]

    mse = np.sum(mse, axis=(-1, -2)) / n  # Prevent division by zero
    return np.mean(mse)
