---
name: annotation-converter
description: 在不同的深度学习标注数据格式（COCO、VOC、YOLO）之间进行转换，支持目标检测和分类任务。当用户需要在不同格式之间转换标注数据时调用。
---

# 标注数据格式转换器

此技能帮助您在不同的深度学习标注数据格式之间进行转换，支持目标检测和分类任务。它支持以下转换：

## 支持的转换

### 目标检测格式
- COCO 转 VOC
- COCO 转 YOLO
- VOC 转 COCO
- VOC 转 YOLO
- YOLO 转 COCO
- YOLO 转 VOC

### 分类格式
- VOC 转 ImageNet（txt 格式）

## 使用说明

当您需要转换标注数据时，只需描述您的转换任务，包括：
1. 源格式
2. 目标格式
3. 输入文件/目录的路径
4. 输出目录/文件的路径
5. 任何额外参数（例如，VOC 转 YOLO 转换时的类别）

## 示例

### 示例 1：将 VOC 转换为 COCO
**用户请求：** 将我在 `/data/voc/xmls` 中的 VOC 标注转换为 COCO 格式，并将输出保存到 `/data/coco/annotations.json`

**操作：** 技能将运行：
```bash
python3 convert_detection.py --format voc2coco --input_voc_xmls_dir /data/voc/xmls --output_coco_json_file /data/coco/annotations.json
```

### 示例 2：将 COCO 转换为 YOLO
**用户请求：** 将 `/data/coco/annotations.json` 中的 COCO 标注转换为 YOLO 格式，并保存到 `/data/yolo`

**操作：** 技能将运行：
```bash
python3 convert_detection.py --format coco2yolo --input_coco_json_file /data/coco/annotations.json --output_dir /data/yolo
```

### 示例 3：使用特定类别将 VOC 转换为 YOLO
**用户请求：** 将 `/data/voc/xmls` 中的 VOC 标注转换为 YOLO 格式，使用类别 [person, car, dog]，并保存到 `/data/yolo`

**操作：** 技能将运行：
```bash
python3 convert_detection.py --format voc2yolo --input_voc_xmls_dir /data/voc/xmls --output_dir /data/yolo --categories person car dog
```

### 示例 4：将 YOLO 转换为 VOC
**用户请求：** 将 `/data/yolo/txts` 中的 YOLO 标注（图片在 `/data/yolo/images`，类别文件在 `/data/yolo/classes.txt`）转换为 VOC 格式，保存到 `/data/voc`

**操作：** 技能将运行：
```bash
python3 convert_detection.py --format yolo2voc --input_yolo_txts_dir /data/yolo/txts --input_yolo_imgs_dir /data/yolo/images --output_dir /data/voc --input_yolo_classes_file /data/yolo/classes.txt
```

### 示例 5：将 VOC 分类转换为 ImageNet
**用户请求：** 将图片在 `/data/voc/images`、XML 在 `/data/voc/xmls` 的 VOC 分类标注转换为 ImageNet 格式，保存到 `/data/imagenet`

**操作：** 技能将运行：
```bash
python3 convert_classify.py --format voc2imagenet --input_voc_imgs_dir /data/voc/images --input_voc_xmls_dir /data/voc/xmls --output_dir /data/imagenet
```

## 注意事项
- `check_format` 参数默认启用，用于验证输入文件的格式
- 对于 YOLO 转 VOC/COCO 转换，您需要提供 classes.txt 文件
- 对于 VOC 转 YOLO 转换，您可以指定类别以控制类别顺序
- 确保输入路径正确且可访问
- 输出目录将在不存在时自动创建
- 如果缺少必要的输入，需要提示用户

## 工作原理

此技能解析您的请求以确定转换类型和参数，然后使用正确的参数调用相应的转换脚本。它无缝处理检测和分类标注的转换。