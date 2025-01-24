import os
import json
import numpy as np

def process_patch(feature, data_s2_path):
    """Processes a single patch, filtering by zero pixels and duplicate dates."""
    properties = feature['properties']
    id_patch = properties['ID_PATCH']
    npy_filename = os.path.join(data_s2_path, f"S2_{id_patch}.npy")
    print(npy_filename)
    if not os.path.exists(npy_filename):
        print(f".npy file for patch {id_patch} not found.")
        return None

    try:
        array = np.load(npy_filename)
        T, C, H, W = array.shape
        total_pixels = H * W

        indices_to_keep = []
        dates_to_keep = []
        
        for t in range(T):
            image = array[t]
            zero_mask = np.all(image == 0, axis=0)
            zero_ratio = np.sum(zero_mask) / total_pixels
            
            if zero_ratio < 0.10:
                 indices_to_keep.append(t)
                 dates_to_keep.append(properties['dates-S2'][str(t)])
            else:
                print(f"Removing time index {t} from patch {id_patch} due to {zero_ratio*100:.2f}% zero pixels.")
        
        if not indices_to_keep:
            print(f"No valid images left for patch {id_patch} after filtering black pixels.")
            return None

        array = array[indices_to_keep]
        
        # Handle duplicate dates
        dates = [date for date in dates_to_keep]
        date_to_index = {}
        for idx, date in enumerate(dates):
            if date not in date_to_index:
                date_to_index[date] = idx

        indices_to_keep = sorted(date_to_index.values())
        
        # Print and remove duplicates in dates
        all_indices = set(range(len(dates)))
        indices_to_remove = all_indices - set(indices_to_keep)
        for idx in sorted(indices_to_remove):
            print(f"Duplicate date {dates[idx]} found at index {idx}, removing earlier occurrence(s).")
        
        new_dates_s2 = {str(i): dates[idx] for i, idx in enumerate(indices_to_keep)}
        properties['dates-S2'] = new_dates_s2
        array = array[indices_to_keep]
        
        np.save(npy_filename, array)
        print(f"Updated .npy file for patch {id_patch}.")

        #Final Check
        if len(properties['dates-S2']) != array.shape[0]:
             print(f"Patch {id_patch} removed: length of dates-S2 ({len(properties['dates-S2'])}) does not match T ({array.shape[0]})")
             return None
        
        return feature
    
    except Exception as e:
        print(f"Error processing patch {id_patch}: {str(e)}")
        return None

if __name__ == "__main__":
    dataset_dir = "./DATASET"
    data_s2_path = os.path.join(dataset_dir, "DATA_S2")
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    with open(metadata_path, 'r') as json_file:
        metadata = json.load(json_file)
    features = metadata['features']
    updated_features = []
    for feature in features:
        processed_feature = process_patch(feature, data_s2_path)
        if processed_feature:
            updated_features.append(processed_feature)

    metadata['features'] = updated_features
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Updated JSON data saved to {metadata_path}.")