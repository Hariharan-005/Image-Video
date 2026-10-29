# RoadVision AI: Intelligent Pothole Detection, Severity Assessment, and Road Condition Analysis Using Image Analytics

RoadVision AI is an image-based road inspection project that uses deep learning to detect and localize potholes, estimate their severity from relative bounding-box area, and generate a preliminary road-condition assessment.

The system combines a trained **YOLO11n** object detection model with rule-based severity classification and a project-defined Road Health Score. It produces annotated road images and structured reports containing pothole counts, severity distribution, road-health estimates, and maintenance priorities.

> **Important:** Severity categories and the Road Health Score are heuristic estimates based on image detections. They are not standardized engineering measurements and do not directly measure pothole depth or volume.

## Features

- Detects and localizes potholes in road images using YOLO11n.
- Uses confidence-filtered bounding-box predictions.
- Estimates relative pothole area.
- Classifies detected potholes as **Low**, **Moderate**, **High**, or **Critical**.
- Calculates a project-defined Road Health Score from 0 to 100.
- Maps the score to an estimated road condition and maintenance priority.
- Draws green bounding boxes on detected potholes.
- Generates structured reports that can be exported to CSV.

## System Workflow

1. **Input:** Provide a road image.
2. **Preprocessing:** Prepare the image according to the model's input requirements.
3. **Pothole detection:** YOLO11n predicts pothole locations and confidence scores.
4. **Filtering:** Retain detections that satisfy the configured confidence threshold.
5. **Severity assessment:** Estimate relative bounding-box area and assign a severity category.
6. **Road assessment:** Calculate the Road Health Score and determine the estimated road condition and maintenance priority.
7. **Output:** Save or display the annotated image and structured assessment report.

## Model and Dataset

### Dataset

The project uses the **BharatPothole** dataset.

Reported dataset organization:

| Split | Number of images |
|---|---:|
| Training | 5,067 |
| Validation | 1,345 |
| Testing | 662 |
| **Total** | **7,074** |

The dataset annotations were standardized for YOLO object detection. Polygon annotations were converted to bounding boxes where required. Valid images without pothole annotations were retained as background examples.

### Model

- **Detector:** YOLO11n
- **Framework:** Ultralytics
- **Input image size:** 640 × 640
- **Training:** 20 total epochs
- **Hardware used:** NVIDIA Tesla T4 GPU
- **Target class:** Pothole

Training settings may need to be adjusted according to available hardware and the configuration in your code.

## Evaluation Results

The trained model was evaluated on the test dataset.

| Metric | Result |
|---|---:|
| Precision | 51.10% |
| Recall | 41.00% |
| mAP@0.50 | 39.96% |
| mAP@0.50:0.95 | 13.52% |

These results are the reported experimental values for this implementation. They indicate that detection performance still has room for improvement, particularly in localization quality and the recovery of potholes that are present in the images.

## Severity and Road Assessment

Each retained pothole detection is assigned a severity category using predefined relative-area thresholds derived from the observed annotation-area distribution.

The system then combines severity weights and area-based factors to calculate the Road Health Score. The score is mapped to an estimated road condition and maintenance priority.

| Road Health Score | Estimated condition | Maintenance priority |
|---|---|---|
| 80–100 | Good | Low |
| 60–<80 | Moderate | Medium |
| 40–<60 | Poor | High |
| Below 40 | Critical | Urgent |

A higher score indicates a better estimated condition under the project's rules. A score of 100 is assigned when no potholes are detected.

**Interpretation note:** A model can miss potholes or generate false detections. Therefore, a score of 100 does not prove that a road is physically free of potholes. The score is a project-defined heuristic and requires validation against expert inspections and real road measurements before operational use.

## Example Output

An example assessment from the project reported:

- Total potholes detected: **15**
- Low: **0**
- Moderate: **2**
- High: **9**
- Critical: **4**
- Road Health Score: **0/100**
- Estimated road condition: **Critical**
- Maintenance priority: **Urgent**

The result illustrates how image detections are summarized into a structured report. It should be interpreted as the output of the current model and assessment rules, not as a standardized engineering rating.

## Requirements

The implementation was developed in Python and Google Colab. The exact dependencies should match the imports and versions used in your notebook.

Typical requirements include:

- Python 3
- `ultralytics`
- `opencv-python`
- `numpy`
- `pandas`
- `matplotlib`
- Google Colab or a local Python environment

Install the main packages with:

```bash
pip install ultralytics opencv-python numpy pandas matplotlib
```

If your notebook uses additional libraries, install those as well.

## Running the Project

The project was developed in Google Colab. To reproduce the workflow:

1. Open the project notebook in Google Colab.
2. Upload or mount Google Drive if the dataset and model checkpoint are stored there.
3. Download the BharatPothole dataset from the source listed above.
4. Extract the dataset and verify the image, label, and dataset configuration paths.
5. Run the dataset preparation and annotation-validation steps.
6. Train YOLO11n or load the trained checkpoint.
7. Run inference on road images.
8. Review the annotated images and generated assessment reports.

The trained checkpoint used in the reported experiment was saved at:

```text
/content/roadvision_runs/pothole_yolo11n/weights/best.pt
```

This is a Google Colab runtime path, not a public download link. To reuse the checkpoint outside that runtime, include it in an appropriate storage location or provide a repository/release link.

### Example inference snippet

The following is a minimal example using Ultralytics. Update the checkpoint and image paths to match your environment.

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
results = model.predict(
    source="path/to/road_image.jpg",
    imgsz=640,
    conf=0.35,
    save=True
)
```

This snippet runs object detection and saves prediction visualizations using Ultralytics. The project's severity classification, Road Health Score, maintenance-priority mapping, and CSV reporting require the corresponding assessment code from the project implementation.

## Suggested Repository Structure

```text
RoadVision-AI/
├── README.md
├── notebooks/
│   └── roadvision_ai.ipynb
├── src/
│   ├── detection.py
│   ├── severity_assessment.py
│   ├── road_health.py
│   └── report_generation.py
├── results/
│   ├── annotated_images/
│   └── reports/
├── requirements.txt
└── .gitignore
```

This is a suggested organization. Keep only the files and folders that actually exist in your repository, and adjust filenames to match your implementation.

## Limitations

- Detection performance is limited by the training data, image quality, and model predictions.
- Small, partially occluded, or visually ambiguous potholes may be missed.
- False detections can affect pothole counts and downstream assessments.
- Relative bounding-box area is not a direct measurement of physical pothole size, depth, or volume.
- The Road Health Score is heuristic and has not been validated as a standardized engineering index.
- The current documented implementation is image-based; video processing should not be considered supported unless implemented and tested.

## Future Improvements

- Expand and diversify the training dataset.
- Improve detection performance through further model tuning and data augmentation.
- Validate severity estimates against expert-labelled assessments.
- Investigate calibrated physical-size or depth estimation using suitable measurements or additional sensors.
- Validate road-health scoring against real inspection data.
- Add video processing if required and evaluate performance on continuous footage.

## References

- **BharatPothole dataset:** https://www.kaggle.com/datasets/surbhisaswatimohanty/bharatpothole

## Disclaimer

RoadVision AI is an academic image-analytics project intended for preliminary pothole detection and road-condition analysis. Its outputs should not replace professional road inspections or engineering assessments.
