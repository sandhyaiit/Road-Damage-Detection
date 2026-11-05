# Transfer learning for road damage detection

This follows Arya, Maeda, Ghosh, Toshniwal, Mraz, Kashiyama, and Sekimoto, [Transfer Learning-based Road Damage Detection for Multiple Countries](https://arxiv.org/abs/2008.13101).

The dataset is RDD2020: 26,620 smartphone photos from Japan, India, and the Czech Republic. Damage boxes are labeled D00 (longitudinal crack), D10 (transverse crack), D20 (alligator crack), and D40 (pothole). Download it from [sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector). The photos are not in this repo.

## What is here

Three detectors were fine-tuned on that set:

- SSD MobileNet, 60,000 steps, in `Training_inference_SSD_mobilnet_60k_steps (2).pdf`
- Faster R-CNN with Inception-ResNet at 640×640, in `Training_infrence_Faster_RCNN_Inception_resnet_640x640.pdf`
- YOLOv3, in `YoloV3_training_inference.ipynb` and `data_prep_With_augmentation_YoloV3.pdf`

SSD MobileNet was the model kept for the pipeline. `Final_pipeline.ipynb` runs it on photos. `Final_pipeline_video.ipynb` runs it on dashcam video. `app.py` is the Streamlit app, and `Deploy_model.ipynb` is the deployment notebook. `Deploy_model.webm` is a short recording of that app.

Data preparation for SSD and Faster R-CNN is in `Data_prep_for_SSD_and_Faster_RCNN_casestudy2.ipynb`. `EDA.ipynb` is the look at the labels before training.

The notebooks were run in Colab, and the saved model path points at Google Drive (`mobile_ner_v2_fpnlite_640x640`). Weights are not included.

## Citation

```text
Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Mraz, A., Kashiyama, T., and Sekimoto, Y.
Transfer Learning-based Road Damage Detection for Multiple Countries.
arXiv:2008.13101, 2020.
```
