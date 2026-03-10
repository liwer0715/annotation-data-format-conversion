import os
import json
from tqdm import tqdm
from PIL import Image
from utils import checking_format


def yolo_2_coco(yolo_txt_dir,
                img_dir,
                coco_output_path,
                classes_file,
                check_format=True):
    """
    将YOLO格式的标注转换为COCO格式的JSON文件。

    参数:
    - yolo_label_dir: str, YOLO TXT标注文件的目录
    - img_dir: str, 图像文件的目录
    - coco_output_path: str, 输出COCO JSON文件的路径
    - classes_file: str, yolo中记录的标签文件
    - check_format: bool, 是否检查标注格式
    """
    images = []
    annotations = []
    categories = []
    img_id = 0

    if check_format:
        checking_format.check_yolo_txt_file(yolo_txt_dir)

    # 构建COCO格式的类别列表
    with open(classes_file, "r") as f:
        for line_number, line in enumerate(f, start=1):  # 使用enumerate跟踪行号,id起始是1
            class_name = line.strip()  # 去除行尾的换行符
            categories.append({
                "id": line_number,
                "name": f"{class_name}",
                "supercategory": ""
            })

    # 读取YOLO的TXT文件
    for img_file in tqdm(sorted(os.listdir(img_dir))):
        img_id += 1
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        if img_file.endswith(tuple(image_extensions)):
            img_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(img_dir, img_file)

            img = Image.open(img_path)
            img_width, img_height = img.size  # 添加一个函数来获取图像的尺寸
            img.close()
            # 创建图像条目
            images.append({
                "file_name": img_file,
                "height": img_height,
                "width": img_width,
                "id": img_id,
            })

            txt_file = os.path.join(yolo_txt_dir, img_name + ".txt")
            if not os.path.exists(txt_file):
                print(f"{txt_file} not found")
                continue
            # 读取对应图像的YOLO标注
            with open(txt_file) as f:
                for line in f:
                    parts = line.strip().split()
                    class_id, x_center, y_center, w, h = (
                        int(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    )
                    # coco类别id从1开始,yolo的是0开始
                    class_id += 1

                    # 确定类别ID
                    category_id = next(
                        (cat["id"] for cat in categories if cat["id"] == class_id),
                        None)

                    # 创建COCO格式的标注
                    annotations.append({
                        "id":
                        len(annotations) + 1,
                        "image_id":
                        img_id,
                        "category_id":
                        category_id,
                        "bbox": [
                            int((x_center - w/2) * img_width),
                            int((y_center - h/2) * img_height),
                            int(w * img_width),
                            int(h * img_height),
                        ],
                        "area":
                        int(w * h * img_width * img_height),
                        "iscrowd":
                        0,
                    })

    # 构建COCO格式的数据结构
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    # 写入JSON文件
    with open(coco_output_path, "w") as f:
        json.dump(coco_format, f, indent=2)

    if check_format:
        checking_format.check_coco_json(coco_output_path)


if __name__ == "__main__":
    # yolo的txt文件目录
    yolo_txt_dir = "data/labels/txts"
    # yolo标注数据中记录的类别文件
    classes = "data/labels/classes.txt"
    # 与txt文件匹配的图像文件目录
    img_dir = "data/imgs"
    # 输出COCO格式的JSON文件
    coco_output_path = "data/yolo2coco.json"

    yolo_2_coco(yolo_txt_dir, img_dir, coco_output_path, classes)
