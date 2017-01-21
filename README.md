### SSD Train Own Data Tutorial

> This tutorial written by Jin Tian, Master in Tsinghua University, if you have any question about this post, contact me via wechat: `jintianiloveu`. Repost is also welcomed, but please remain this copyright info, enjoy :)

### Part A Data Orginization
Before we get started, I have say that SSD original source code data orginization is really a mess. If you want train your own data you don't know where to go. But now, I am going change it, reshape it to a simple and clear way. **you just clone source code and make it, the rest thing is all about my code, using my code you can sperate caffe-ssd source code from your dataset folder in a more clear way**

### Part B Get Your Images And Labels
First of all, get your images and labels, I assume that you have 7000 images and same count labels in txt format, orginize them in 2 folder, called **Images** which contains all images, and **Labels** which contains all labels. And, most important at all is that, every single image **must have same name mapped label txt file**, means if you have a image `0001.jpg` you must labeled it in `0001.txt`. And all txt files must in this format:
```
class_index xmin ymin xmax ymax
```
It's simple enough! In my way, I place all images inside `~/data/MyDataset/Images` and all my labels in `~/data/MyDataset/Labels`, hopefully please do not change `Images` and `Labels` folder name, we gonna use it.

### Part C Create Your Work Folder
Seperate from caffe-ssd source code directory, you can create a invidual folder named `MyDataset`, our all work will compelet in this folder. OK, clone my `kitti-ssd` into your's anywhere you like. you can change this folder name as you like(example. face-ssd). Here inside we got this things:
```
data
models
train_ssd_kitti.py
```
ok, next we are going work with `data` first, because we have to generate lmdb file first.

### Part D Create lmdb Database
OK, in this step, we are going put all data into lmdb database., this will generate a lmdb folder inside `~/data/MyDataset` folder which contains KITTI_trainval and KITTI_test data.
```
cd data
bash create_list.sh
bash create_data.sh
```
Done! now you get `trainval.txt` `test.txt` and `test_name_size.txt`
But you have to get your labelmap_dataset.prototxt, here is suggestion:
> if you have 5 classed named the 6th class in name `background`

```
item {
  name: "none_of_the_above"
  label: 6
  display_name: "background"
}
```
And later in `train_ssd.py` you gonna change two value : `num_classes`  and `background_index_id`.

### Part E Get VGG Pretrain Model and Start Train SSD
Download VGG pretrain model and place into `models/VGGNet` , the everything was done! Just a little change in `train_ssd_kitti.py` you can train ssd ready! Here is something you have to change:
```
sys.path.insert(0, "/home/chenqi-didi/Documents/work/caffe/python")
data_root_dir = "/home/chenqi-didi/data/"
caffe_root = "/home/chenqi-didi/Documents/work/caffe"
train_data = data_root_dir + "KITTI/lmdb/KITTI_trainval_lmdb"
# The database file for testing data. Created by data/KITTI/create_data.sh
test_data = data_root_dir + "KITTI/lmdb/KITTI_test_lmdb"
model_name = "VGG_KITTI_{}".format(job_name)

# Directory which stores the model .prototxt file.
save_dir = "models/VGGNet/KITTI/{}".format(job_name)
# Directory which stores the snapshot of models.
snapshot_dir = "models/VGGNet/KITTI/{}".format(job_name)
# Directory which stores the job script and log file.
job_dir = "jobs/VGGNet/KITTI/{}".format(job_name)
# Directory which stores the detection results.
output_result_dir = "{}/data/KITTI/results/{}/Main".format(os.environ['HOME'], job_name)
label_map_file = "{}/data/labelmap_kitti.prototxt".format(current_dir)

# Defining which GPUs to use.
gpus = "0,1"
```

Find above code change all KITTI into your dataset name, save it and you are ready to go!
```
python train_ssd_kitti.py
```

### Finally, For Test Result
got your image path in data, for example `data/test2.jpg` and then change `ssd_detection.py` file path, then run:
```
python ssd_detection.py
```
Here is the result:
![image](http://ojek5ksya.bkt.clouddn.com/np56AsBKU38WLcZH.png)