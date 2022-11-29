from typing import List, Optional

import numpy as np
from skimage import measure


class GeoGrid:
    """
    Object used to store geographical lat/lon coordinates.  For now it's a glorified dictionary.
    """
    def __init__(
            self,
            latitude: List,
            longitude: List
    ):
        """
        :param List latitude: The latitude for each pixel on a grid given as a vector.
        :param List longitude: The longitude for each pixel on a grid given as a vector.
        """
        self.latitue = latitude
        self.longitude = longitude

        self.shape = (len(self.latitue), len(self.longitude))


class Scene:
    """
    Object used for describing a phenomenon defined by a mask.  Stores individual "events"
    as connected components of the provided mask and filters out events below a certain size
    (total number of pixels in connected component).
    """
    def __init__(
            self,
            mask: np.array,
            timestamp: Optional[str] = None,
            connectivity: Optional[int] = 2,
            min_size: Optional[int] = 1
    ):
        """
        :param np.array mask: True/False mask describing a phenomenon.
        :param str timestamp: A string formatted as YYYYMMDDhhmm(e.g. 201501011430) giving the datetime of the
                              provided mask.
        :param int connectivity: Determines how nearest neighbors are chosen when building connected components.
                                 Choosing 1 selects top/bottom, left/right neighbors whereas 2 includes
                                 the diagonals as well.
        :param int min_size: Smallest size connected component(total number of pixels) to include as an "event".
        """
        self.mask = mask
        self.timestamp = timestamp
        self.connectivity = connectivity
        self.min_size = min_size

        self.labels, self.events = self._connected_components()

    def __getitem__(self, item):
        return self.events[item]

    def _connected_components(self):
        """
        Uses scikit-image to calculate the connected components of the provided mask.

        :return np.array labels: An np.array the same size as mask where each pixel is a unique string
                                 that determines which "event"(i.e. connected component) the pixel belongs to.
                                 The string is formatted as {timestamp}.{id} where `timestamp` is the provided
                                 timestamp and `id` is the number between 1 and the total number of events.
                                 If the pixel is a part of the background, the string is 'None'.
        :return dict events: A dictionary that maps an event id to the event itself.  The event id is formatted
                             as {timestamp}.{id} where `timestamp` is the provided timestamp and `id` is the
                             number between 1 and the total number of events.  The event is a RegionProperties
                             object that contains many attributes for the connected component.  See
                             https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
                             for a full list of properties.
        """
        labels, num_events = measure.label(
            self.mask,
            return_num=True,
            connectivity=self.connectivity
        )
        events = measure.regionprops(labels)
        assert len(events) == num_events

        # Assign unique ID to each component
        event_ids = ['None']
        event_ids.extend(
            [
                f'{self.timestamp}.{i + 1}' if events[i].area >= self.min_size else 'None'
                for i in range(num_events)
            ]
        )
        labels = np.array(event_ids)[labels]
        events = {
            c_id: x for c_id, x in zip(event_ids[1:], events)  # Skip the first 'None' event
            if c_id != 'None'
        }

        return labels, events
