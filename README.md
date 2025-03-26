# PointNetLK: Point Cloud Registration using PointNet for C3VD

### Requires:
* PyTorch and torchvision
* NumPy
* SciPy
* MatPlotLib
* plyfile

### Main files for experiments:
* train_classifier.py: train PointNet classifier for c3vd
* train_pointlk.py: train PointNet-LK for c3vd
* ./ptlk/data/datasets.py: add C3VD class
* trian_clsassifier_ref.py: original one of PointNetLK
* trian_pointlk_ref.py: original one of PointNetLK


### Bash shell scripts for cluster qsub:
* train_c3vd.sh: train PointNet classifier and transfer to PointNet-LK for C3VD
* ex1_train.sh: original one of PointNetLK


### Citation

```
@InProceedings{yaoki2019pointnetlk,
       author = {Aoki, Yasuhiro and Goforth, Hunter and Arun Srivatsan, Rangaprasad and Lucey, Simon},
       title = {PointNetLK: Robust & Efficient Point Cloud Registration Using PointNet},
       booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
       month = {June},
       year = {2019}
}
```
