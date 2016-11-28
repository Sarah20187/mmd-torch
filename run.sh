#th main.lua -data ../../data/exp21 -nClasses 7 -cache ../../data/cache22 -epochSize 100 -netType alexnetowt

th main.lua -caffe 1 \
-pretrainedPrototxtPath /home/huayizeng/mmd-torch/models/VGG_CNN_S_deploy.prototxt \
-pretrainedModelPath /home/huayizeng/mmd-torch/models/VGG_CNN_S.caffemodel

# todo-hy: we can load the pretrained caffe model totally identical to the one used in MMD-caffe
# two diff: 1. learning rate decay | 2. mean using the imagenet?

th main.lua \
-nDonkeys 1 \
-isDA 0 \
-caffe 1 \
-pretrainedPrototxtPath /home/huayizeng/mmd-torch/models/VGG_CNN_S_deploy.prototxt \
-pretrainedModelPath /home/huayizeng/mmd-torch/models/VGG_CNN_S.caffemodel \
-data ../data/office \
-nClasses 31 -cache ../data/cache-mmd \
-epochSize 100 \
-batchSize 64


# from scratch
th main.lua \
-data ../data/office \
-nClasses 31 -cache ../data/cache-mmd \
-epochSize 100 \
-batchSize 64

#mmd with inv
th main.lua \
-nDonkeys 1 \
-policy inv \
-isDA 1 \
-caffe 1 \
-pretrainedPrototxtPath /home/huayizeng/mmd-torch/models/VGG_CNN_S_deploy.prototxt \
-pretrainedModelPath /home/huayizeng/mmd-torch/models/VGG_CNN_S.caffemodel \
-data ../data/office \
-nClasses 31 -cache ../data/cache-mmd \
-epochSize 100 \
-batchSize 64


th main.lua \
-nDonkeys 1 \
-policy inv \
-isDA 1 \
-caffe 1 \
-pretrainedPrototxtPath /home/huayizeng/mmd-torch/models/VGG_CNN_S_deploy.prototxt \
-pretrainedModelPath /home/huayizeng/mmd-torch/models/VGG_CNN_S.caffemodel \
-data ../data/office \
-nClasses 31 -cache ../data/cache-mmd \
-epochSize 10 \
-batchSize 64