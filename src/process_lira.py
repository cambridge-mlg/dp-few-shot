# Portions of this code are excerpted from:
# https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/advanced_mia_example.py

import pickle
import numpy as np
import os
from lira import compute_score_lira
import argparse

NUM_TARGET_MODELS = 257


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None, help="Path to in_indices, stat, and output score files.")
    args = parser.parse_args()

    for config in ['none', 'film']:
        for shot in ['10', '25', '50', '100']:
            for epsilon in ['1', '2', '4', '8', 'inf']:
                with open(os.path.join(args.data_path, 'in_indices_{}_{}_{}.pkl'.format(config, shot, epsilon)), "rb") as f:
                    in_indices = pickle.load(f)
                with open(os.path.join(args.data_path, 'stat_{}_{}_{}.pkl'.format(config, shot, epsilon)), "rb") as f:
                    stat = pickle.load(f)
                n = len(stat[0])

                # Now we do MIA for each model
                all_scores = []
                all_y_true = []
                for idx in range(NUM_TARGET_MODELS):
                    print(f'Target model is #{idx}')
                    stat_target = stat[idx]  # statistics of target model, shape (n, k)
                    in_indices_target = in_indices[idx]  # ground-truth membership, shape (n,)

                    # `stat_shadow` contains statistics of the shadow models, with shape
                    # (num_shadows, n, k). `in_indices_shadow` contains membership of the shadow
                    # models, with shape (num_shadows, n). We will use them to get a list
                    # `stat_in` and a list `stat_out`, where stat_in[j] (resp. stat_out[j]) is a
                    # (m, k) array, for m being the number of shadow models trained with
                    # (resp. without) the j-th example, and k being the number of augmentations
                    # (1 in our case).
                    stat_shadow = np.array(stat[:idx] + stat[idx + 1:])
                    in_indices_shadow = np.array(in_indices[:idx] + in_indices[idx + 1:])
                    stat_in = [stat_shadow[:, j][in_indices_shadow[:, j]] for j in range(n)]
                    stat_out = [stat_shadow[:, j][~in_indices_shadow[:, j]] for j in range(n)]

                    # Compute the scores and use them for MIA
                    scores = compute_score_lira(stat_target, stat_in, stat_out, fix_variance=True)

                    y_score = np.concatenate((scores[in_indices_target], scores[~in_indices_target]))
                    y_true = np.concatenate((np.zeros(len(scores[in_indices_target])),
                                             np.ones(len(scores[~in_indices_target]))))

                    all_scores.append(y_score)
                    all_y_true.append(y_true)

                all_y_true = np.hstack(all_y_true)
                all_scores  = np.hstack(all_scores)
                result = {
                    'y_true': all_y_true,
                    'scores': all_scores
                }

                if not os.path.exists(args.data_path):
                    os.makedirs(args.data_path)
                with open(os.path.join(args.data_path, 'scores_{}_{}_{}.pkl'.format(config, shot, epsilon)), "wb") as f:
                    pickle.dump(result, f)


if __name__ == '__main__':
    main()
