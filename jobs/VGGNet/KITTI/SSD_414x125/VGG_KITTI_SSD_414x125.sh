/home/chenqi-didi/Documents/work/caffe/build/tools/caffe train \
--solver="models/VGGNet/KITTI/SSD_414x125/solver.prototxt" \
--weights="models/VGGNet/VGG16.v2.caffemodel" \
--gpu 0,1 2>&1 | tee jobs/VGGNet/KITTI/SSD_414x125/VGG_KITTI_SSD_414x125.log
