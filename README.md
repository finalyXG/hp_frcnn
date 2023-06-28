This file is the extention of the file "README.md"

0. **How to create bounding box files based on the images and the segmentation maps**
#### Example to create bouding box files for a given LIP dataset samples
export PYTHONPATH=./:./utils:./utils/pytorch_vision; python ./mains/main_create_bbox.py -c ./configs/lip_gen_bbox.json

After running this command line, we should be able to see a new directory "lip_bbox_sample" in "./datasets/LIP/". This is our new generated dataset files containing bounding box in a numpy file format.

1. **How to view bounding box on an example image**
#### Example to run the main script to view an example of LIP bbox dataset 
export PYTHONPATH=./:./utils:./utils/pytorch_vision; python ./mains/main_view_bbox.py -c ./configs/lip_view_bbox.json --i old_lip.png --dopt lip_origin --dir ./can_del_saved


where (1) "--i" is the argument to specify the output image name of an sample image with bouding box drawn; (2) "--dopt" is "dataset option", which is used to specify which dataset config option we are going to apply to the dataset, e.g., "lip_origin" means we do not apply relabelling trick while "lip_new_mapping" uses new mappings.
In this way, you can replace "--i old_lip.png --dopt lip_origin" to "--i new_lip.png --dopt lip_new_mapping", then you can go to the directory "./tmp" to check the different annotaion by different options; (3) "--dir" specifies the logging directory.


2. **How to run a training process**

#### Example to run the main script of a few samples of LIP dataset:
export PYTHONPATH=./:./utils:./utils/pytorch_vision; python ./mains/main_sample_training.py -c ./configs/lip_original_sample.json


#### Example to run the main script of a few samples of LIP dataset with relabel
export PYTHONPATH=./:./utils:./utils/pytorch_vision; python ./mains/main_sample_training.py -c ./configs/lip_new_mapping_sample.json



3. **Open Tensorboard server to monitor the training process** 

Type `tensorboard --logdir=saved/log/ --samples_per_plugin images=20` at the project root, then server will open at a particular link, e.g., `http://localhost:6006`