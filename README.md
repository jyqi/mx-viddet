
# Object-Detection on ImageNet-VID
The project is implemented to compare the performance of [**R-FCN**](https://arxiv.org/abs/1605.06409) and [**FPN**](https://arxiv.org/abs/1612.03144) for object detection on [**ImageNet VID dataset**](http://bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php). Firstly, I will introduce the requirements and steps to run the code, and then I will compare the performance of these two networks.
This project is mainly based on [**Deformable ConvNets**](https://github.com/msracver/Deformable-ConvNets) with major contributors Yuwen Xiong, Haozhi Qi, Guodong Zhang, Yi Li, Jifeng Dai, Bin Xiao, Han Hu and Yichen Wei, thanks for the contribution.
## Run the code
### Requirements
1.	MXNet from the official repository. I tested my code on MXNet-cu101.
2.	Python 2.7. the code does not support Python 3. If you want to use Python 3, you need to modify the code to make it work.
3.	The following Python packages are required (also included in requirement.txt):
```
mxnet-cu101
Cython
EasyDict
opencv-python
mxboard
pyyaml
```
### Installation
1.	Clone the repository:
```
git clone https://github.com/jyqi/mx-viddet.git
cd mx-viddet
```
2.	Run `sh ./init.sh`. The scripts will build cython module automatically and create some necessary folders.
3.	Install MXNet and all dependencies by
```
pip install -r requirements.txt
```
4.	Make sure the correct cuda is on your `LD_LIBRARY_PATH`
### Preparation for Training & Testing
1.	Download the ImageNet VID dataset, and symlink the dataset.
```
# symlink the ImageNet-VID dataset
ln -s /path_to_dataset/ILSVRC/Annotations/ ./data/ILSVRC/Annotations/
ln -s /path_to_dataset/ILSVRC/Data/ ./data/ILSVRC/Data/
ln -s /path_to_dataset/ILSVRC/ImageSets/VID/ ./data/ILSVRC/ImageSets/VID/
```
2.	Download ImageNet-pretrained ResNet-v1-101 backbone model, and put it under the following folder:
```
./model/pretrained_model/resnet_v1_101-0000.params
```
### Usage
1.	All of the experiment settings(GPU #, dataset, etc.) are kept in yaml config files at following folder:
```
./experiments/rfcn/cfgs/
./experiments/fpn/cfgs/
```
2.	Four config files have been provided so far, namely, R-FCN/R-FCN-DCN for ImageNet VID, FPN/FPN-DCN for ImageNet VID, respectively.
3.	To perform experiments, run the python scripts with the corresponding config file as input. 
For example, to train and test R-FCN-DCN on ImageNet VID with ResNet-v1-101, use the following command:
```
python experiments/rfcn/rfcn_end2end_train_test.py –cfg experiemnts/rfcn/cfgs/resnet_v1_101_imagenet_vid_rfcn_dcn_end2end_ohem.yaml
```
to train and test FPN-DCN on ImageNet VID with ResNet-v1-101, use the following command:
```
python experiments/fpn/fpn_end2end_train_test.py –cfg experiemnts/fpn/cfgs/resnet_v1_101_imagenet_vid_fpn_dcn_end2end_ohem.yaml
```
A cache folder would be created automatically to save the model and the log under ./output/rfcn_dcn/imagenet_VID/.
