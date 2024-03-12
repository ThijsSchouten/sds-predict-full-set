import os
import numpy as np
import tensorflow as tf
import rasterio
from rasterio import windows
from rasterio.merge import merge
from rasterio.plot import reshape_as_image
from rasterio.features import shapes
import geopandas as gpd
from PIL import Image
import glob
from tqdm.auto import tqdm

def normalize(input_image):
    return tf.cast(input_image, tf.float32) / 127.5 - 1

def drop_alpha_channel(image):
    return image[..., :3] 

def chw_to_hwc(arr):
    return np.moveaxis(arr, 0, -1)

def prep_batch_for_pred(batch):
    batch = np.stack([chw_to_hwc(img) for img in batch])
    batch = drop_alpha_channel(batch)  # drop alpha channel after reshaping
    return batch

def predict_tiles(model, tiles):
    prepared_tiles = prep_batch_for_pred(tiles)
    batch = normalize(prepared_tiles)
    return model.predict(batch, verbose=0)

def get_window_coordinates(n_rows, n_cols, tilestride, width, height):
    return [(j * tilestride, i * tilestride, width, height)
            for i in range(n_rows)
            for j in range(n_cols)]

def read_tiles(src, window_coordinates):
    w = window_coordinates
    tiles = []
    new_w = []  # create a new list for the window coordinates
    for idx, (col, row, width, height) in enumerate(w):
        tile = src.read(window=windows.Window(col_off=col, row_off=row, width=width, height=height), boundless=True, fill_value=0)
        
        # Skip if the tile is fully transparent
        if tile.shape[0] == 4 and np.all(tile[3, :, :] == 0):
            continue
        # Skip if the tile has no values
        elif np.all(tile == 0):
            continue
        
        tiles.append(tile)
        new_w.append(w[idx])  # only add the window coordinate if the tile was not skipped
    
    return tiles, new_w  # return the tiles and the filtered window coordinates


def process_predictions(predictions, output_array, window_coordinates, tilestride, overlap_half):
    # Iterate through each prediction and corresponding window coordinate
    for prediction, (col_off, row_off, _, _) in zip(predictions, window_coordinates):
        # Remove singleton dimensions from the prediction and extract prediction band for target class
        prediction = np.squeeze(prediction)
        target_prediction = prediction[:, :, 1] 
        # Normalize the values to 0-255 and convert to 8-bit integer
        target_prediction = (target_prediction * 255).astype(np.uint8) 

        # Discard half of the overlap on each side
        center_prediction = target_prediction[overlap_half:-overlap_half, overlap_half:-overlap_half]

        # Calculate the boundaries for the output array
        row_start = row_off + overlap_half
        row_end = min(row_off + tilesize - overlap_half, output_array.shape[0])
        col_start = col_off + overlap_half
        col_end = min(col_off + tilesize - overlap_half, output_array.shape[1])

        # Calculate the boundaries for the center_prediction
        pred_row_end = row_end - row_start
        pred_col_end = col_end - col_start

        # Write the center_prediction to the correct position in the output array
        output_array[row_start:row_end, col_start:col_end] = center_prediction[:pred_row_end, :pred_col_end]



print("TensorFlow version: {}".format(tf.__version__))
print("Rasterio version: {}".format(rasterio.__version__))

proj = 'app/proj/' # dont change


#################################
# Run settings
ckpt = 'v10-cp-0002' #checkpoint name without extension
species = 'dk' # species, used for the folder where to find the checkpoint, but also as the foldername in the output
img_dir = 'img_kampen' # the dir with images to predict on

# CHANGE THIS 
BATCH_SIZE = 3
tilesize = 512
overlap = 200 #128
overlap_half = overlap // 2
tilestride = tilesize - overlap
#################################



output_dir = f'{proj}/output/{species}/{ckpt}'
os.makedirs(output_dir, exist_ok=True)

ckpt_path = f'{proj}/input/{species}/{ckpt}.ckpt'
print(ckpt_path)
model = tf.keras.models.load_model(ckpt_path)


img_paths = sorted(glob.glob(f'{proj}/input/{img_dir}/*.tif'))
# reverse sort to get the most recent images first
img_paths = img_paths[::-1]

imgs = [os.path.basename(img).split('.')[0] for img in img_paths]


for img, image_path in zip(imgs, img_paths):
    print(f"Predicting image: {img}")
    print(f"info-- image_path: {image_path}")
    try:
        output_mosaic = f'{output_dir}/PRED-{species}-{ckpt}-{img}.tif'

        if os.path.exists(output_mosaic):
            print(f"Output mosaic already exists: {output_mosaic}")
            continue

        crs = None

        print(f"Opening image: {img}")
        with rasterio.open(image_path) as src:
            crs = src.crs
            cols, rows = src.meta['width'], src.meta['height']

            width = height = tilesize
            n_cols = cols // tilestride
            n_rows = rows // tilestride

            output_array = np.zeros((rows, cols), dtype=np.uint8)

            print("Getting window coordinates")
            window_coordinates = get_window_coordinates(n_rows, n_cols, tilestride, width, height)

            print("Predicting bathches")
            for i in tqdm(range(0, len(window_coordinates), BATCH_SIZE), desc='Predicting batches'):
                print(f"Predicting batch {i}", end='\r')
                batch_window_coordinates = window_coordinates[i:i + BATCH_SIZE]

                batch_tiles, new_batch_window_coordinates = read_tiles(src, batch_window_coordinates)

                if len(batch_tiles) > 0:  # check if there ar e any tiles in the batch
                    batch_predictions = predict_tiles(model, batch_tiles)
                    process_predictions(batch_predictions, output_array, new_batch_window_coordinates, tilestride, overlap_half)

        print("Writing output mosaic")
        print(f"info-- CRS: {crs}, transform: {src.transform}, shape: {output_array.shape}")
        with rasterio.open(output_mosaic, 'w', driver='GTiff',
                        height=output_array.shape[0], width=output_array.shape[1],
                        count=1, dtype='uint8',
                        crs=crs, transform=src.transform,
                        compress='deflate') as dest: 
            dest.write(output_array, 1)
    except Exception as e:
        print(f"Failed to predict image: {img}. Error:")
        print(e)
        continue
