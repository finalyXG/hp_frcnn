This file is the extention of the file "README.md"

1. **How to run**  

# Example to run the main script of a few samples of LIP dataset
export PYTHONPATH=/home/cafi/laurence/lip_fast_rcnn:/home/cafi/laurence/lip_fast_rcnn/utils:/home/cafi/laurence/lip_fast_rcnn/utils/pytorch_vision:/home/cafi/laurence/lip_fast_rcnn; python ./mains/main_sample_training.py -c ./configs/lip_original_sample.json

# Example to run the main script of a few samples of LIP dataset with relabel
export PYTHONPATH=/home/cafi/laurence/lip_fast_rcnn:/home/cafi/laurence/lip_fast_rcnn/utils:/home/cafi/laurence/lip_fast_rcnn/utils/pytorch_vision:/home/cafi/laurence/lip_fast_rcnn; python ./mains/main_sample_training.py -c ./configs/lip_new_mapping_sample.json


2. **Open Tensorboard server** 
Type `tensorboard --logdir=saved/log/ --samples_per_plugin images=20` at the project root, then server will open at a particular link, e.g., `http://localhost:6006`