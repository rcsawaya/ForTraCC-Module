from collections import OrderedDict
from typing import List, Optional

import numpy as np

from objects import GeoGrid, Scene


class TimeOrderedSequence:
    """
    Object used to convert a sequence of provided masks into a time series of events.  Each event is
    spatially defined as a connected component of a mask and temporally linked to other connected components
    through the ForTraCC algorithm(Vila, Daniel Alejandro, et al. "Forecast and Tracking the Evolution
    of Cloud Clusters (ForTraCC) using satellite infrared imagery: Methodology and validation." Weather
    and Forecasting 23.2 (2008): 233-245.).
    """
    def __init__(
            self,
            masks: List[np.array],
            timestamps: List[str],
            grid: GeoGrid
    ):
        """
        :param List[np.array] masks: A list of masks defining a phenomenon each given as an np.array.
        :param List[str] timestamps: A list of timestamps with the same size as `masks` where each item is a
                                     string formatted as YYYYMMDDhhmm(e.g. 201501011430) giving the datetime of the
                                     corresponding mask.
        :param GeoGrid grid: A GeoGrid object that defines the geographical grid on which the masks live.
        """
        assert len(masks) == len(timestamps), 'The number of images must be equal to the number of timestamps.'

        self.masks = masks
        self.timestamps = timestamps
        self.grid = grid

        self.time_series = []
        self.time_series_map = dict()  # Last added event_id in series -> time_series index

    def _build_scenes(
            self,
            connectivity: Optional[int] = 2,
            min_size: Optional[int] = 1
    ):
        """
        Converts the provided masks into `Scene` objects used to define events.  If need be, this can
        be modified to run in parallel for a large amount of masks.

        :param int connectivity: Determines how nearest neighbors are chosen when building connected components.
                                 Choosing 1 selects top/bottom, left/right neighbors whereas 2 includes
                                 the diagonals as well.
        :param int min_size: Smallest size connected component(total number of pixels) to include as an "event".

        :return List[Scene]: A list of `Scene` objects used to define an event.
        """
        # Can be processed in parallel if need be
        return [
            Scene(
                mask,
                timestamp=self.timestamps[i],
                connectivity=connectivity,
                min_size=min_size
            )
            for i, mask in enumerate(self.masks)
        ]

    @staticmethod
    def _get_overlap_map(
            prev_scene: Scene,
            scene: Scene,
            connectivity: Optional[int] = 2,
            min_olap: Optional[float] = 0.25
    ):
        """
        Calculates the overlap between events in any two consecutive scenes and creates a mapping
        between any event in one scene to the events it overlaps with in the other.

        :param Scene prev_scene: The first `Scene` object in the provided consecutive pair.
        :param Scene scene: The second `Scene` object in the provided consecutive pair.
        :param int connectivity: Determines how nearest neighbors are chosen when building connected components.
                                 Choosing 1 selects top/bottom, left/right neighbors whereas 2 includes
                                 the diagonals as well.
        :param int min_olap: The smallest amount of overlap(number of overlapping pixels relative to the size
                             of the first event) to consider as valid.  Any overlap below this value is ignored.

        :return dict: A dictionary that maps any event in one scene to the events it overlaps with in the other.
        """
        overlap = prev_scene.mask & scene.mask
        olap_scene = Scene(
            overlap,
            connectivity=connectivity
        )

        olap_map = dict()  # event_id -> List[event_ids]
        for olap in olap_scene.events.values():
            min_row, min_col, max_row, max_col = olap.bbox

            prev_event_ids = set(
                prev_scene.labels[min_row:max_row, min_col:max_col][olap.image]
            )
            crnt_event_ids = set(
                scene.labels[min_row:max_row, min_col:max_col][olap.image]
            )
            assert len(prev_event_ids) == len(crnt_event_ids) == 1

            prev_event_id = list(prev_event_ids)[0]
            if prev_event_id == 'None':
                continue

            crnt_event_id = list(crnt_event_ids)[0]
            if crnt_event_id == 'None':
                continue

            frac_olap = olap.area / prev_scene[prev_event_id].area
            if frac_olap < min_olap:
                continue

            # Forward mapping for "split" and "dissipation" events
            if prev_event_id not in olap_map:
                olap_map[prev_event_id] = []
            olap_map[prev_event_id].append(
                {
                    'event_id': crnt_event_id,
                    'event': scene[crnt_event_id],
                    'overlap': frac_olap
                }
            )

            # Backward mapping for "merge" and "initiation" events
            if crnt_event_id not in olap_map:
                olap_map[crnt_event_id] = []
            olap_map[crnt_event_id].append(
                {
                    'event_id': prev_event_id,
                    'event': prev_scene[prev_event_id],
                    'overlap': frac_olap
                }
            )
        return olap_map

    def _add_initiation(self, event_id, event,  tag):
        """
        Adds an "initiation" event to the time series.

        :param str event_id: The event ID of the initiated event.
        :param RegionProperties event: The event to add to the time series.  The event is a RegionProperties
                                       object that contains many attributes for the connected component.  See
                                       https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
                                       for a full list of properties.
        :param str tag: The name to use for describing the event.

        :return set: A set of events to remove from the pool of events which have yet to be labeled.
        """
        assert event_id not in self.time_series_map

        self.time_series.append(
            [(event_id, event, tag)]
        )
        self.time_series_map[event_id] = len(self.time_series) - 1
        return {event_id}

    def _add_continuation(self, prev_event_id, event_id, event, tag):
        """
        Adds a "continuation" event to the time series.

        :param str prev_event_id: The event ID of the continuing event used to identify the time series.
        :param str event_id: The event ID of the new event that is being added as a continuation of the time series.
        :param RegionProperties event: The event to add to the time series.  The event is a RegionProperties
                                       object that contains many attributes for the connected component.  See
                                       https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
                                       for a full list of properties.
        :param str tag: The name to use for describing the event.

        :return set: A set of events to remove from the pool of events which have yet to be labeled.
        """
        assert prev_event_id in self.time_series_map

        time_series_idx = self.time_series_map.pop(prev_event_id)
        self.time_series[time_series_idx].append(
            (event_id, event, tag)
        )
        self.time_series_map[event_id] = time_series_idx

        return {prev_event_id, event_id}

    def _add_dissipation(self, event_id, tag):
        """
        Adds a "dissipation" event to the time series.

        :param str event_id: The event ID to label as "dissipation".
        :param str tag: The name to use for describing the event.

        :return set: A set of events to remove from the pool of events which have yet to be labeled.
        """
        assert event_id in self.time_series_map

        time_series_idx = self.time_series_map[event_id]
        _, event, _ = self.time_series[time_series_idx][-1]
        self.time_series[time_series_idx][-1] = (
            event_id, event, tag
        )

        return {event_id}

    @staticmethod
    def _separate_max_olap(events):
        """
        Separates the event with the largest overlap from the others.

        :param List[dict] events: A list of events each given as a dictionary with keys `event_id`, `event` and
                                  `overlap`.

        :return Tuple[dict, List[dict]]: Returns the event with the largest overlap and a list of the other events.
        """
        max_frac_olap = 0.0
        max_idx = 0
        for i, item in enumerate(events):
            if item['overlap'] > max_frac_olap:
                max_idx = i
                max_frac_olap = item['overlap']
        return events.pop(max_idx), events

    def _add_merge(self, mapped_events, event_id, event):
        """
        Adds a "merge" event to the time series.

        :param List[dict] mapped_events: A list of events each given as a dictionary with keys `event_id`, `event` and
                                         `overlap`.
        :param str event_id: The event ID which indicates the final merged event.
        :param RegionProperties event: The event to add to the time series.  The event is a RegionProperties
                                       object that contains many attributes for the connected component.  See
                                       https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
                                       for a full list of properties.

        :return set: A set of events to remove from the pool of events which have yet to be labeled.
        """
        max_olap_event, events = self._separate_max_olap(mapped_events)
        labeled_events = set()

        # The event with the maximum overlap gets continued
        labeled_events.update(
            self._add_continuation(
                max_olap_event['event_id'],
                event_id,
                event,
                'continuation - merge'
            )
        )

        # The other events dissipate in a merge
        for item in events:
            prev_event_id = item['event_id']
            labeled_events.update(
                self._add_dissipation(prev_event_id, 'dissipation - merge')
            )
        return labeled_events

    def _add_split(self, mapped_events, event_id):
        """
        Adds a "split" event to the time series.

        :param List[dict] mapped_events: A list of events each given as a dictionary with keys `event_id`, `event` and
                                         `overlap`.
        :param str event_id: The event ID which indicates the initial event that later splits into multiple events.

        :return set: A set of events to remove from the pool of events which have yet to be labeled.
        """
        max_olap_event, events = self._separate_max_olap(mapped_events)
        labeled_events = set()

        # The event with the maximum overlap gets continued
        labeled_events.update(
            self._add_continuation(
                event_id,
                max_olap_event['event_id'],
                max_olap_event['event'],
                'continuation - split'
            )
        )

        # The other events initiate with a split
        for item in events:
            next_event_id = item['event_id']
            labeled_events.update(
                self._add_initiation(next_event_id, item['event'], 'initiation - split')
            )
        return labeled_events

    def run_fortracc(
            self,
            connectivity: Optional[int] = 2,
            min_olap: Optional[float] = 0.25,
            min_size: Optional[int] = 150
    ):
        """
        Runs the ForTraCC algorithm.

        :param int connectivity: Determines how nearest neighbors are chosen when building connected components.
                                 Choosing 1 selects top/bottom, left/right neighbors whereas 2 includes
                                 the diagonals as well.
        :param int min_olap: The smallest amount of overlap(number of overlapping pixels relative to the size
                             of the first event) to consider as valid.  Any overlap below this value is ignored.
        :param int min_size: Smallest size connected component(total number of pixels) to include as an "event".

        :return List[Tuple[str, RegionProps, str]]: A list of tuples each of which stores the event ID, the event itself
                                                    and the description of the event as a tuple.
        """
        all_scenes = self._build_scenes(
            connectivity=connectivity,
            min_size=min_size
        )

        prev_scene = Scene(
            np.zeros(self.grid.shape, dtype=bool),
            connectivity=connectivity
        )
        for scene in all_scenes:
            # Create mapping between overlapping events
            olap_map = self._get_overlap_map(
                prev_scene,
                scene,
                connectivity=connectivity,
                min_olap=min_olap
            )

            # Gather all the events between the two images and sort them in descending order by size
            all_events = list(
                prev_scene.events.items()
            )
            all_events.extend(
                list(scene.events.items())
            )
            all_events = OrderedDict(
                sorted(all_events, key=lambda x: x[1].area)
            )  # [smallest, ... , biggest]

            # Go through each event and add to existing time series
            crnt_timestamp = int(scene.timestamp)
            while len(all_events) > 0:
                event_id, event = all_events.popitem()

                event_timestamp = int(event_id.split('.')[0])
                is_new = event_timestamp == crnt_timestamp
                mapped_events = []
                if event_id in olap_map:
                    mapped_events.extend(
                        [
                            x for x in olap_map[event_id] if x['event_id'] in all_events
                        ]
                    )

                # Categorize the event relative to the events it connects to
                labeled_events = set()
                if is_new:
                    if len(mapped_events) == 0:
                        # Initiation of new time series
                        self._add_initiation(
                            event_id,
                            event,
                            'initiation'
                        )
                    elif len(mapped_events) == 1:
                        # Continuation of previous event
                        prev_event_id = mapped_events[0]['event_id']
                        continue_events = self._add_continuation(
                            prev_event_id,
                            event_id,
                            event,
                            'continuation'
                        )
                        labeled_events.update(continue_events)
                    else:
                        # Merging of two or more events
                        merged_events = self._add_merge(
                            mapped_events,
                            event_id,
                            event
                        )
                        labeled_events.update(merged_events)
                else:
                    if len(mapped_events) == 0:
                        # Dissipation of time series
                        self._add_dissipation(event_id, 'dissipation')
                    elif len(mapped_events) == 1:
                        # Continuation of current event
                        next_event_id = mapped_events[0]['event_id']
                        continue_events = self._add_continuation(
                            event_id,
                            next_event_id,
                            mapped_events[0]['event'],
                            'continuation'
                        )
                        labeled_events.update(continue_events)
                    else:
                        # Splitting of two or more events
                        split_events = self._add_split(
                            mapped_events,
                            event_id
                        )
                        labeled_events.update(split_events)

                # Clean up the events that have been labeled this round
                for seen_event in labeled_events:
                    if seen_event in all_events:
                        all_events.pop(seen_event)
            prev_scene = scene
        return self.time_series


class ThresholdEvent(TimeOrderedSequence):
    """
    Specific TimeOrderedSequence object that defines phenomenon based on a threshold.  Any pixel that is
    less than the provided threshold is treated as a part of the phenomenon.
    """
    def __init__(
            self,
            images: List[np.array],
            timestamps: List[str],
            grid: GeoGrid,
            threshold: float
    ):
        """
        :param List[np.array] images: A list of images given as np.arrays.  Each pixel should have the same units
                                      as the provided threshold.
        :param List[str] timestamps: A list of timestamps with the same size as `masks` where each item is a
                                     string formatted as YYYYMMDDhhmm(e.g. 201501011430) giving the datetime of the
                                     corresponding mask.
        :param GeoGrid grid: A GeoGrid object that defines the geographical grid on which the masks live.
        :param float threshold: The threshold below which a phenomenon is defined.  The threshold should have the
                                same units as the values in `images`.
        """
        masks = [image < threshold for image in images]
        super().__init__(masks, timestamps, grid)
