LAB4 - AI and Machine Learning Training - Tensorflow
Author: Huynh Nguyen Truong Thinh - 1613343
Link fb: fb.com/relife.baka

[=] Collecting dataset:

    Save positive images in raw_datasets/pos
    Save negative images in raw_datasets/neg

[=] Creating h5 file :

    python3 creating_h5.py

    it will create 2 file:
      datasets/test_carvnoncar.h5    # for test set
      datasets/train_carvnoncar.h5   # for train set

[=] Train the model

    python3 main.py

[=] Predict new images:

    Put all images in images directory
    python3 predict_images.py
