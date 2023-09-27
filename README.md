# Multi Camera Visitor Tracking Sytem

## Objective

We aim to develop a system for measuring visitor wait times in specific scenarios. This system will help assess engagement levels, identify inefficiencies, and monitor interactions between customers and staff.

## Description

- Creating a visual representation of a designated area in front of the counter or showcase, referred to as a Region of Interest (ROI), and tracking the duration of people's presence within this area. The aim is to enhance and refine this process for optimal efficiency.


#### Test Case Description

- Input: Video (mp4) + ROI coordinates.
- ROI is in the format (xmin,ymin,xmax,ymax)
- Output: Video (mp4)

## Requirements

To install the required Python packages for this project, create a virtual environment (optional but recommended) and run the following command:

```shell
pip install -r requirements.txt
```

## Usage

### Command Line Interface (CLI)

To run the Multi Camera Visitor Tracking System via the command line interface (CLI), follow these steps:

1. **Clone the Repository:** Clone the repo to local machine

``` shell
git clone https://github.com/AtalGupta/visitor-dwell-time.git
```
2. **Run system**: Run the Multi Camera Visitor Tracking System using the following command:

```shell
python main.py
```
## USING state-of-the-art (SOTA) models

1. **Object Detection**: Consider adopting models like EfficientDet or Detectron2's RetinaNet. These models offer advanced architecture designs and have shown superior accuracy and efficiency compared to YOLO in recent benchmarks.

2. **Tracking**: Instead of relying solely on traditional tracking algorithms, incorporate deep learning-based trackers such as FairMOT or Tracktor, which leverage the temporal information for better tracking, especially in crowded scenes.

3. **Feature Fusion**: Explore models that integrate both object detection and tracking in an end-to-end manner. This can potentially improve the system's robustness by allowing better incorporation of temporal information during detection and tracking stages.
