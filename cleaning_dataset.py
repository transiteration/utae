import os
import json
import random
import numpy as np
from src.dataset import compute_norm_vals

json_file_path = './JAXA/metadata.json'

with open(json_file_path, 'r') as f:
    data = json.load(f)

features = data['features']
n_features = len(features)

folds = list(range(1, 6)) * (n_features // 5 + 1)
folds = folds[:n_features]
random.shuffle(folds)

for i, feature in enumerate(features):
    feature['properties']['Fold'] = folds[i]

features_to_keep = []

with open('errors.txt', 'a') as errors_file:
    for feature in features:
        properties = feature['properties']
        id_patch = properties['ID_PATCH']

        try:
            dates_s2 = properties['dates-S2']

            dates = [dates_s2[str(i)] for i in range(len(dates_s2))]

            date_to_index = {}
            for idx, date in enumerate(dates):
                date_to_index[date] = idx

            indices_to_keep = sorted(date_to_index.values())

            all_indices = set(range(len(dates)))
            indices_to_remove = all_indices - set(indices_to_keep)
            for idx in sorted(indices_to_remove):
                print(f"Duplicate date {dates[idx]} found at index {idx}, removing earlier occurrence(s).")

            new_dates_s2 = {str(i): dates[idx] for i, idx in enumerate(indices_to_keep)}
            properties['dates-S2'] = new_dates_s2

            npy_filename = f'./JAXA/DATA_S2/S2_{id_patch}.npy'

            if os.path.exists(npy_filename):
                array = np.load(npy_filename)

                if array.shape[0] != len(dates):
                    print(f"Warning: Array time dimension {array.shape[0]} does not match number of dates {len(dates)} for patch {id_patch}.")
                    continue

                array = array[indices_to_keep]

                indices_to_keep_final = []
                new_dates = {}

                T, C, H, W = array.shape
                total_pixels = H * W

                for t in range(T):
                    image = array[t]
                    zero_mask = np.all(image == 0, axis=0)
                    zero_pixels = np.sum(zero_mask)
                    zero_ratio = zero_pixels / total_pixels

                    if zero_ratio < 0.10:
                        indices_to_keep_final.append(t)
                        original_index = indices_to_keep[t]
                        date_key = str(original_index)
                        if date_key in dates_s2:
                            new_dates[str(len(new_dates))] = dates_s2[date_key]
                        else:
                            print(f"Date for index {date_key} not found in dates-S2 for patch {id_patch}.")
                    else:
                        print(f"Removing time index {t} from patch {id_patch} due to {zero_ratio*100:.2f}% zero pixels.")

                if not indices_to_keep_final:
                    print(f"No valid images left for patch {id_patch} after filtering.")
                    continue

                array = array[indices_to_keep_final]
                np.save(npy_filename, array)
                print(f"Updated .npy file for patch {id_patch}.")

                properties['dates-S2'] = new_dates

                # Add the new cleaning condition
                T = array.shape[0]
                if len(properties['dates-S2']) != T:
                    error_message = f"Patch {id_patch} removed: length of dates-S2 ({len(properties['dates-S2'])}) does not match T ({T})\n"
                    errors_file.write(error_message)
                    continue
                else:
                    features_to_keep.append(feature)
            else:
                print(f".npy file for patch {id_patch} not found.")
        except Exception as e:
            error_message = f"Error processing patch {id_patch}: {str(e)}\n"
            errors_file.write(error_message)
            continue

# Update the features list with the kept features
data['features'] = features_to_keep

with open(json_file_path, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Updated JSON data saved to {json_file_path}.")

compute_norm_vals(folder="./JAXA", sat="S2")
