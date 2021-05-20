# FaceMesh EdgeTPU

## Introduction

This repo reproduce the result from mediapipe [FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh.html) on 
Raspberry pi or other platform with Coral EdgeTPU. The model first performs face detection and then align the face. The aligned
faces are fed into mesh generator for face landmark detection.

- [x] add head pose estimation 
- [x] add kalman filter

## Demo
![](assets/demo.gif)

## Performance 
The performance for 1 face on Edge TPU, make sure the Pi has enough power otherwise the performance drops a lot!

|  platform  | Desktop | Raspberry Pi 4 |
| :---------:| :-----: | :------------: |
| TPU |  44FPS  |        14.5FPS  |
| No TPU| 35FPS| 7.7FPS |

## Installation
Install the requirements by:

```pip install -r requirements.txt```

Then run the demo with:

```python main.py```

## Credit to
I borrowed the model from: [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
