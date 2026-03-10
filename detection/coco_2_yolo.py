import os
import json
from tqdm import tqdm
import shutil
from utils import checking_format


def coco2yolo_convert(size, box):
    """
    将COCO格式的边界框坐标转换为YOLO格式的坐标。

    参数:
    - size: 一个包含图像宽度和高度的元组 (w, h)。
    - box: 一个表示COCO格式边界框的元组 (x, y, w, h)，其中x, y是边界框中心的坐标，w, h是边界框的宽度和高度。

    返回值:
    - 一个表示YOLO格式边界框的元组 (x, y, w, h)，其中x, y是边界框中心的坐标，w, h是边界框的宽度和高度，已经根据图像尺寸进行了归一化。
    """
    dw = 1.0 / (size[0])
    dh = 1.0 / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def coco_2_yolo(coco_json_file, ana_txt_dir, check_format=True):
    """
    将COCO格式的标注文件转换为YOLO格式的标注文本，不处理图片。

    参数:
    - coco_json_file: str, COCO格式标注文件的路径。
    - ana_txt_dir: str, 生成的YOLO格式标注文本的输出目录路径。
    - check_format: bool, 是否检查标注文件格式。

    """
    if check_format:
        checking_format.check_coco_json(coco_json_file)

    data = json.load(open(coco_json_file, "r"))

    # 创建输出目录及其子目录
    if os.path.exists(ana_txt_dir):
        shutil.rmtree(ana_txt_dir)
    os.mkdir(ana_txt_dir)
    os.mkdir(os.path.join(ana_txt_dir, "txts"))

    # 创建类别映射文件，并初始化类别映射字典
    id_map = {}  # coco数据集的id不连续！是从1开始的，重新映射一下再输出！
    with open(os.path.join(ana_txt_dir, "classes.txt"), "w") as f:
        # 写入classes.txt
        for i, category in enumerate(data["categories"]):
            f.write(f"{category['name']}\n")
            id_map[category["id"]] = i

    # 组织标注信息，按图像ID分组
    anns = {}
    for ann in data["annotations"]:
        imgid = ann["image_id"]
        anns.setdefault(imgid, []).append(ann)

    # 遍历图像信息，生成对应的YOLO格式标注文本
    for img in tqdm(data["images"]):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
        with open(os.path.join(os.path.join(ana_txt_dir, "txts"), ana_txt_name),
                  "w") as f_txt:

            ann_img = anns.get(img_id, [])
            for ann in ann_img:
                box = coco2yolo_convert((img_width, img_height), ann["bbox"])
                f_txt.write(
                    "%s %s %s %s %s\n" %
                    (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))

    if check_format:
        checking_format.check_yolo_txt_file(os.path.join(ana_txt_dir, "txts"),
                                            len(data["categories"]))


if __name__ == "__main__":

    # coco的json文件路径
    coco_json_file = "./coco_train.json"
    # yolo格式的数据集保存路径
    ana_txt_dir = "./labels"
    coco_2_yolo(coco_json_file, ana_txt_dir)
