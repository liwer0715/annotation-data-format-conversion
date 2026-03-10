
# 检测box标注数据转换命令

- check_format参数默认为True，有需要可以自己设置为False

## voc2coco
- python3 convert_detection.py --format voc2coco --input_voc_xmls_dir xml文件夹路径  --output_coco_json_file xxx.json
##
## voc2yolo
#### --categories参数不指定就会自动生成标签文件，顺序不受控制
- python3 .\convert_detection.py --format voc2yolo --input_voc_xmls_dir xml文件夹路径  --output_dir xxx --categories class1 class2 class3...
## 
## coco2voc
- python3 .\convert_detection.py --format coco2voc --input_coco_json_file xxx.json  --output_dir xxx
##
## coco2yolo 
- python3 .\convert_detection.py --format coco2yolo --input_coco_json_file xxx.json  --output_dir xxx
##
## yolo2voc 
#### yolo的classes.txt文件记录的标签名按行存储
- python3 .\convert_detection.py --format yolo2voc --input_yolo_txts_dir txt文件夹路径 --input_yolo_imgs_dir xxx --output_dir xxx --input_yolo_classes_file classes.txt
##
## yolo2coco
- python3 .\convert_detection.py --format yolo2voc --input_yolo_txts_dir txt文件夹路径 --input_yolo_imgs_dir xxx --output_coco_json_file xxx.json --input_yolo_classes_file classes.txt


&nbsp;
****
&nbsp;


# 分类标注数据转换命令

## voc转imagenet的txt格式

- python3 convert_classify.py --format voc2imagenet --input_voc_imgs_dir imgs文件夹路径 --input_voc_xmls_dir xml文件夹路径 --output_dir 输出目录
