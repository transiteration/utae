import os
import ee
import cv2
import glob
import time
import json
import zipfile
import tempfile
import numpy as np
import requests
import rasterio


# -----------------------
# 1. Cloud Mask with SCL
# -----------------------
def maskS2clouds_scl(image):
    """
    Mask out clouds, cloud shadows, cirrus, and snow using the SCL band.
    The SCL classes (for COPERNICUS/S2_SR_HARMONIZED) are:
        0 = No data
        1 = Saturated / Defective
        2 = Dark area pixels
        3 = Cloud shadow
        4 = Vegetation
        5 = Bare soils
        6 = Water
        7 = Unclassified
        8 = Cloud medium probability
        9 = Cloud high probability
        10 = Thin cirrus
        11 = Snow or ice
    """
    scl = image.select("SCL")
    # Keep only pixels NOT in these classes:
    #   3 (cloud shadow), 7 (unclassified), 8 (cloud medium prob)
    #   9 (cloud high prob), 10 (thin cirrus), 11 (snow/ice)
    mask = scl.neq(3).And(scl.neq(9)).And(scl.neq(10))
    # Apply the mask
    masked = image.updateMask(mask)

    # Finally, select only the spectral bands of interest
    # (You can keep SCL if you want, but here we drop it)
    return masked.select(["B2", "B3", "B4", "B8", "B11", "B12"])


# -----------------------
# 2. Read and Download
# -----------------------
def read_band(band_path):
    """Reads a single band TIF file with rasterio."""
    with rasterio.open(band_path) as src:
        data = src.read(1)
        bbox = src.bounds
        meta = src.meta.copy()
    return data, bbox, meta


def download_file(url, dest_folder):
    """Downloads the Sentinel-2 zip file from a getDownloadURL()."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        zip_path = os.path.join(dest_folder, "image.zip")
        with open(zip_path, "wb") as f:
            f.write(response.content)
        return zip_path
    else:
        raise Exception("Failed to download file.")


def process_image(url, index, patch_path):
    """
    Combines a single downloaded Sentinel-2 zip into a stacked numpy array [C, H, W].
    Only looks for B2, B3, B4, B8, B11, B12. Resizes to 128 x 128 for illustration.
    """
    desired_bands = ["B2", "B3", "B4", "B8", "B11", "B12"]

    with tempfile.TemporaryDirectory() as bands_tmp_dir:
        zip_path = download_file(url, bands_tmp_dir)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(bands_tmp_dir)

        # Create a dictionary of { "B2": path_to_B2, "B3": ... }
        band_paths = {}
        for band in desired_bands:
            # Look for a file with something like "B2" in the filename
            for filename in os.listdir(bands_tmp_dir):
                if band in filename and filename.endswith(".tif"):
                    band_paths[band] = filename
                    break

        # Read and stack the bands
        bands_data = []
        for band in desired_bands:
            if band in band_paths:
                band_path = os.path.join(bands_tmp_dir, band_paths[band])
                band_data, _, _ = read_band(band_path)
                # Resize to 128 x 128 (optional; for demonstration)
                band_data = cv2.resize(
                    band_data, (128, 128), interpolation=cv2.INTER_LINEAR
                )
                bands_data.append(band_data)
            else:
                print(f"Warning: {band} not found in the downloaded files.")
                # If a band is missing, you might skip it or fill with zeros

        stacked_bands = np.stack(bands_data, axis=0)  # shape: [6, 128, 128]
        np.save(f"{patch_path}_{index}.npy", stacked_bands)


# -----------------------
# 3. Download & Stack
# -----------------------
def download_sat_images(bbox, mask_name, patch_dir):
    """
    Downloads Sentinel-2 data within a bounding box, dates,
    cloud masks with SCL, and returns a dict of dates for each image index.
    """
    patch_path = os.path.join(patch_dir, mask_name)
    bbox_coords = ee.Geometry.Rectangle([bbox.left, bbox.bottom, bbox.right, bbox.top])

    # Build your collection, filter dates, apply SCL-based cloud mask
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(bbox_coords)
        .filterDate("2020-01-01", "2020-12-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
        .map(maskS2clouds_scl)
    )

    # Convert to list to loop in Python
    collection_size = collection.size().getInfo()
    image_list = collection.toList(collection_size)
    dates = {}

    for i in range(collection_size):
        image = ee.Image(image_list.get(i))
        date_str = image.date().format("YYYYMMdd").getInfo()
        dates[str(i)] = int(date_str)

        # Download at 10 m scale in EPSG:4326
        url = image.getDownloadURL(
            {"scale": 10, "crs": "EPSG:4326", "region": bbox_coords}
        )

        try:
            process_image(url=url, index=i, patch_path=patch_path)
        except Exception as e:
            print(f"Failed to process {patch_path} at index {i}: {e}")
            dates.pop(str(i))
            continue

    return dates


def stack_sat_images(patch_dir):
    """
    Stacks downloaded .npy files into a time-series array [T, C, H, W].
    Removes the temporary .npy after reading (optional).
    """
    array_list = []
    patch_images = glob.glob(os.path.join(patch_dir, "*.npy"))
    for patch_file in patch_images:
        try:
            arr = np.load(patch_file)
            array_list.append(arr)
            os.remove(patch_file)  # Remove if you don't need it anymore
        except:
            continue
    stacked_array = np.stack(array_list, axis=0)
    return stacked_array  # shape: [T, 6, 128, 128] in this example


# -----------------------
# 4. Main Pipeline
# -----------------------
def get_patch_id(metadata_path):
    """
    Retrieves the last processed patch_id from a metadata JSON,
    or initializes it if none exists.
    """
    metadata = {"type": "FeatureCollection", "features": []}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as json_file:
            metadata = json.load(json_file)
        if metadata["features"]:
            last_feature = metadata["features"][-1]
            patch_id = int(last_feature["properties"]["ID_PATCH"]) + 1
            _id = int(last_feature["id"]) + 1
        else:
            patch_id = 10000
            _id = 0
    else:
        patch_id = 10000
        _id = 0
    return metadata, patch_id, _id


def dataset_processing():
    """
    Main function that:
    1) Reads mask .tif files
    2) Downloads cloud-masked S2 images (B2, B3, B4, B8, B11, B12) using SCL
    3) Stacks them into [T, C, H, W] arrays
    4) Saves the arrays and updates metadata
    """
    masks_dir = "./JAXA/split_1"
    ann_dir = "./JAXA/ANNOTATIONS"
    sat_dir = "./JAXA/DATA_S2"
    metadata_path = "./JAXA/metadata.json"
    processed_files_path = "./JAXA/processed.txt"

    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(sat_dir, exist_ok=True)
    if not os.path.exists(processed_files_path):
        open(processed_files_path, "w").close()

    metadata, patch_id, _id = get_patch_id(metadata_path=metadata_path)

    # Get list of .tif masks
    masks = glob.glob(os.path.join(masks_dir, "*.tif"))

    for mask in masks:
        mask_name = os.path.basename(mask).replace(".tif", "")

        # Check if already processed
        with open(processed_files_path, "r") as txt_file:
            processed_files = [line.strip() for line in txt_file]
        if mask_name in processed_files:
            continue

        start_time = time.time()
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Prepare output paths
            sat_npy_path = os.path.join(sat_dir, f"S2_{patch_id}.npy")
            mask_npy_path = os.path.join(ann_dir, f"TARGET_{patch_id}.npy")

            # Read the mask band (annotation)
            ann, bbox, _ = read_band(mask)

            # Download & process S2 with SCL-based cloud masking
            dates_dict = download_sat_images(
                bbox=bbox, mask_name=mask_name, patch_dir=tmp_dir
            )

            # Stack all .npy files into [T, C, H, W]
            stacked_s2 = stack_sat_images(patch_dir=tmp_dir)

            # Save final arrays
            np.save(sat_npy_path, stacked_s2)
            np.save(mask_npy_path, ann)

            # Update metadata with date info
            # Convert { "0": 20220101, "1": 20220215, ... } if desired
            dates = {str(i): date for i, date in enumerate(dates_dict.values())}
            feature = {
                "id": str(_id),
                "type": "Feature",
                "properties": {
                    "ID_PATCH": patch_id,
                    "dates-S2": dates,
                },
            }
            metadata["features"].append(feature)
            with open(metadata_path, "w") as json_file:
                json.dump(metadata, json_file, indent=4)

            # Mark this mask as processed
            with open(processed_files_path, "a") as txt_file:
                txt_file.write(mask_name + "\n")

            # Increment counters
            patch_id += 1
            _id += 1

            elapsed = time.time() - start_time
            print(
                f"{mask_name} | {elapsed:.2f} sec | {len(dates)} images in time-series."
            )


if __name__ == "__main__":
    ee.Initialize()  # Make sure you're authenticated with earthengine
    dataset_processing()
