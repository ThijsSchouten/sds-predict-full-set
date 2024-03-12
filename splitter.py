import rasterio
from rasterio.windows import Window

import numpy as np

import os

def split_tiff(input_file, tile_size, overlap):
    with rasterio.open(input_file) as src:
        width = src.width
        height = src.height
        print(width, height)

        chunk_number = 0

        for i, left in enumerate(range(0, width, tile_size)):
            for j, top in enumerate(range(0, height, tile_size)):
                # Adjust the boundaries to include overlap
                new_left = max(0, left - overlap) if i else left
                new_top = max(0, top - overlap) if j else top
                new_right = min(left + tile_size + (overlap if (left + tile_size) < width else 0), width)
                new_bottom = min(top + tile_size + (overlap if (top + tile_size) < height else 0), height)

                window = Window(new_left, new_top, new_right - new_left, new_bottom - new_top)
                tile = src.read(window=window)
                chunk_number += 1
                # The tile has 4 bands. Test if the alpha band is fully transparent
                if tile.shape[0] == 4 and np.all(tile[3, :, :] == 0):
                    continue

                output_file = input_file.replace('.tif', f'_{chunk_number}.tif')
                new_transform = src.window_transform(window)

                # Write and use deflate compression
                with rasterio.open(output_file, 'w', driver='GTiff', width=tile.shape[2], height=tile.shape[1],
                                   count=tile.shape[0], dtype=tile.dtype, crs=src.crs, transform=new_transform,
                                   compress="deflate") as dst:
                    dst.write(tile)

                print(f"Saved {output_file}")

input_file = 'app/proj/input/leende/Leende_feb_23_28992.tif'
tile_size = int(512*35)   # Adjust the tile size according to your needs 60
overlap = 512  # Overlap size

split_tiff(input_file, tile_size, overlap)
