import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time

from flow import ThresholdEvent
from objects import GeoGrid


def main():
    # Load in the test data
    print('Loading test data...')
    test_fns = sorted([fn for fn in os.listdir('test_data') if fn[-4:] == '.txt'])
    assert len(test_fns) == 12

    lat = list(np.unique(pd.read_csv('test_data/navigation/lat.txt', header=None).values.flatten()))
    lon = list(np.unique(pd.read_csv('test_data/navigation/lon.txt', header=None).values.flatten()))

    test_images = []
    t_stamps = []
    for fn in test_fns:
        data = pd.read_csv(f'test_data/{fn}', sep='\t', header=None).values
        assert data.shape == (len(lat), len(lon))
        test_images.append(data)
        t_stamps.append(fn[:-4])
    g = GeoGrid(lat, lon)

    # Run ForTraCC
    print('Running ForTraCC...')
    s = time.time()
    tos = ThresholdEvent(
        test_images,
        t_stamps,
        g,
        235
    )
    tos.run_fortracc(
        min_size=150
    )
    e = time.time()
    print(f'Elapsed time: {e - s:.4f}s')

    # Re-create the images using the delineated events
    slate = {timestamp: np.zeros(tos.grid.shape) for timestamp in tos.timestamps}
    for i, series in enumerate(tos.time_series):
        for scene in series:
            event_id, event, _ = scene

            timestamp = event_id.split('.')[0]

            min_row, min_col, max_row, max_col = event.bbox

            mask = event.image
            tmp_image = np.zeros(mask.shape)
            tmp_image[mask] = i + 1

            slate[timestamp][min_row:max_row, min_col:max_col] = tmp_image

    # Plot the events as a series of PDFs
    print('Generating images...')
    for timestamp, image in slate.items():
        image[image == 0] = None

        print(f'\tSaving image at {timestamp} in ./python_outputs/')
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gist_rainbow', norm=None, interpolation='none')
        plt.clim(1, len(tos.time_series))
        plt.savefig(f'python_outputs/{timestamp}.pdf')
        plt.tight_layout()
        plt.close()
    return


if __name__ == '__main__':
    main()
