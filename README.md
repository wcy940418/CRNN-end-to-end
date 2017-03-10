# CRNN-end-to-end
A CRNN python implement based on TensorFlow</br>

## Dependencies

* TensorFlow r1.0
* lmdb lib
* OpenCV2

Code from Chaoyue Wang</br>

02/25/2017 Update:</br>

1.Tested test.py with 5000 steps trained model. Performance is not good enough. Need more training(2000000 according to author of crnn paper).</br>

02/23/2017 Update:</br>

1.Tested training. Since CPU computation perforamnce is critical low, 1000 steps need ~2 hours.
2.Added test.py to apply model for testing picture.

02/20/2017 Update:</br>

1.Added dataset.py for handle lmdb database.</br>
2.Adjusted the architecture of directory.</br>

02/15/2017 Update:</br>

1.Added batch normalization to model.py
2.Added training.py for training dataset.</br>

02/09/2017 Update:</br>

1.Added model.py file.</br>
