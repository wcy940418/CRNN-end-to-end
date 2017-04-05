# CRNN-end-to-end
A CRNN python implement based on TensorFlow</br>

## Dependencies

* TensorFlow r1.0
* lmdb lib
* OpenCV2
* Baidu's WarpCTC

Code from Chaoyue Wang</br>

03/24/2017 Update:</br>

1.Added utility.py to handle check points and string conversion.</br>
2.Added check point system to keep model.</br>

02/25/2017 Update:</br>

1.Tested test.py with 3000 steps trained model. Performance is not good enough. Need more training(2000000 according to author of crnn paper).</br>
2.Tested training musicscore dataset, some warnings happened: No valid path.</br>

02/23/2017 Update:</br>

1.Tested training. Since CPU computation perforamnce is critical low, 1000 steps need ~2 hours.</br>
2.Added test.py to apply model for testing picture.</br>

02/20/2017 Update:</br>

1.Added dataset.py for handle lmdb database.</br>
2.Adjusted the architecture of directory.</br>

02/17/2017 Update:</br>

1.Added batch normalization to model.py</br>
2.Added training.py for training dataset.</br>

02/15/2017 Update:</br>

1.Added model.py file.</br>
