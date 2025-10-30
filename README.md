# radar_transformer
Transformer-based deep learning architecture for 3D point matching in sparse radar point clouds

# Cite (BibTeX)
If you use this software please cite:
```
@misc{michalczyk2025learningpointcorrespondencesradar,
      title={Learning Point Correspondences In Radar 3D Point Clouds For Radar-Inertial Odometry}, 
      author={Jan Michalczyk and Stephan Weiss and Jan Steinbrener},
      year={2025},
      eprint={2506.18580},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.18580}, 
}
```

# Remarks
If training on your data then make sure to include correct transformation between IMU and Radar sensors in the `prepare_dataset.py` script.
This is because input to the network are pointclouds in the IMU frame. Also, make sure to adapt the training/inference script to your sensor's
FOV and set the DC filtering offset in the `utils.py` script. DC offset is a constant detection close to (0, 0) caused by antennas cross-talk.
