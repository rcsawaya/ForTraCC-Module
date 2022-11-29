# ForTraCC Module
The ForTraCC Module contains tools to implement the 
[ForTraCC algorithm](http://mtc-m16b.sid.inpe.br/col/sid.inpe.br/mtc-m15@80/2008/06.02.17.27/doc/ForTraCC_Published.pdf) 
natively in Python without the need of the original Fortran code.  There are slight differences between the Python and
Fortran implementations, but overall, the Python implementation accomplishes the same task.  Compare the plots in 
`python_outputs` and `fortran_outputs` for more details.

ForTraCC works by first identifying a phenomenon with a series of masks(0 - background, 1 - phenomenon) and then
stitching the temporally disparate phenomena together into a time series based on overlaps between consecutive masks.
An isolated phenomenon is defined by spatially grouping together connected components of a mask
and temporally linking other connected components by their consecutive overlaps.  For example, in its original 
implementation, ForTraCC was used to track mesoscale convective systems that evolve over time where each isolated 
system is a phenomenon.  The original algorithm also included a forecasting component which was meant to predict
future phenomenon, but this feature is not implemented here.

## Quickstart: Simple Thresholding Case
For a full example script(visualization included) of this case see `tutorial.py`.

Suppose we are given a time-ordered series of 2D images stored as a list of 2D numpy arrays.  If we want to represent
a particular phenomenon by thresholding the values of the images, we can apply ForTraCC with

```python
from fortracc_python.flow import ThresholdEvent
from fortracc_python.objects import GeoGrid

grid = GeoGrid(
    latitude,  # The latitude given as a list of numbers(must match number of rows in `images`)
    longitude  # The longitude given as a list of numbers(must match number of columns in `images`)
)
phenomenon = ThresholdEvent(
    images,  # A list of 2D, time-ordered images to use
    timestamps,  # The datetime of each image given as a list of strings
    grid,  # The grid object on which each image is defined
    threshold  # The threshold below which a pixel is marked as a phenomenon
)
time_series = phenomenon.run_fortracc()
```
For defining phenomenon with more than just a single threshold, see the next section below.

## Overview
There are two main modules: `flow.py` and `objects.py`.  The former contains Python classes for running ForTraCC whereas
the latter contains a few useful classes for defining phenomenon.  In this section, we run through the main classes and 
discuss how they may be modified as a part of a larger pipeline. 

### ForTraCC Implementation: `flow.py`
`flow.py` has two classes: A base class `TimeOrderedSequence` 
```python
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
```
and a custom class `ThresholdEvent` which defines a phenomenon with a single threshold
```python
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
```
Looking at the base class, all that is needed to run ForTraCC is a time-ordered list of masks(0 - background, 
1 - phenomenon).  If a particular phenomenon requires more complicated transformations of the native satellite images,
then all that needs to be done is to create a class that generates the phenomenon masks from the original images and pass
these masks to `TimeOrderedSequence`, either by class inheritance or by explicit instantiation.

For example, let's quickly create a class which defines a phenomenon with two threshols instead of just one
```python
class DoubleThresholdEvent(TimeOrderedSequence):
    def __init__(
            self,
            images: List[np.array],
            timestamps: List[str],
            grid: GeoGrid,
            threshold1: float,
            threshold2: float
    ):
        masks = [(threshold1 < image) & (image < threshold2) for image in images]
        super().__init__(masks, timestamps, grid)
```
This also facilitates more complicated definitions of masks such as those based on the anomaly or comparisons with 
higher-order moments calculated from the native images.

### ForTraCC Objects: `objects.py`
`objects.py` contains two useful classes for ForTraCC.  The first, `GeoGrid`, is(for now) a glorified dictionary 
that houses all the geographical information of an image.  
```python
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
```
The second class, `Scene`, is useful for representing the connected components of a single image.  
```python
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
```
Let's focus on `Scene` since `GeoGrid` is extremely simple.  `Scene` uses `skimage.measure` to delineate the connected 
compenents of the provided mask.  Once the connected components are defined, they are passed to 
`skimage.measure.regionprops` which calculates a myriad of properties for each of the connected components(for a full
list of the properties, see https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops).  
These properties may come in handy later down the line, but for now, the only useful one is `area` which gives the total
pixel count of the connected component.  