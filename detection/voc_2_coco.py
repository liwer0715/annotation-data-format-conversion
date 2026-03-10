import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil
from utils import checking_format


def voc_2_coco(xmls_dir, json_output_path, check_format=True, need_imgs=False, input_voc_imgs_dir=None, copy_imgs=True):
    """
    将PASCAL VOC格式的数据集转换为COCO格式，支持处理图片

    参数:
    - xmls_dir: str, 包含VOC XML标注文件的目录路径。
    - json_output_path: str, 生成的COCO格式标注信息的输出JSON文件路径。
    - check_format: bool, 是否检查XML标注文件格式，默认为True。
    - need_imgs: bool, 是否需要处理图片。
    - input_voc_imgs_dir: str, VOC格式图片文件夹的路径。
    - copy_imgs: bool, 是否拷贝图片，否则移动图片。

    """

    if need_imgs and not os.path.exists(input_voc_imgs_dir):
        raise Exception("VOC格式图片文件夹不存在")
    if not os.path.exists(xmls_dir):
        raise Exception("VOC格式XML文件文件夹不存在")

    images = []
    annotations = []
    categories = []
    category_set = set()
    img_id = 0

    if check_format:
        checking_format.check_xml_format(xmls_dir)

    # 创建输出目录
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    if need_imgs:
        imgs_output_dir = os.path.join(os.path.dirname(json_output_path), "imgs")
        os.makedirs(imgs_output_dir, exist_ok=True)

    # 遍历VOC数据集中的XML文件
    for xml_file in tqdm(sorted(os.listdir(xmls_dir))):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xmls_dir, xml_file)
            img_path = xml_file.replace(".xml", ".jpg")  # 假设图像文件名为xml文件去掉.xml后缀
            img_id += 1

            # 解析XML文件
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 获取图像信息
            img = {
                "file_name": img_path,
                "height": int(root.find("size").find("height").text),
                "width": int(root.find("size").find("width").text),
                "id": img_id,
            }
            images.append(img)

            # 遍历XML中的每个对象，提取类别和边界框信息
            for obj in root.findall("object"):
                category = obj.find("name").text
                # 处理类别信息，确保每个类别有唯一的ID
                if category not in category_set:
                    category_id = len(categories) + 1
                    categories.append({
                        "id": category_id,
                        "name": category,
                        "supercategory": ""
                    })
                    category_set.add(category)
                else:
                    category_id = next(
                        (cat["id"] for cat in categories if cat["name"] == category),
                        None,
                    )
                # 提取并计算边界框坐标
                bbox = obj.find("bndbox")
                x, y, w, h = (
                    int(bbox.find("xmin").text),
                    int(bbox.find("ymin").text),
                    int(bbox.find("xmax").text) - int(bbox.find("xmin").text),
                    int(bbox.find("ymax").text) - int(bbox.find("ymin").text),
                )

                # 构建标注信息
                ann = {
                    "id": len(annotations) + 1,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
                annotations.append(ann)

            # 拷贝图片
            if need_imgs:
                source_img_path = os.path.join(input_voc_imgs_dir, img_path)
                target_img_path = os.path.join(imgs_output_dir, img_path)
                if os.path.exists(source_img_path):
                    if copy_imgs:
                        shutil.copy(source_img_path, target_img_path)
                    else:
                        shutil.move(source_img_path, target_img_path)
                else:
                    print(f"图片文件不存在: {source_img_path}")

    # 构建COCO格式的数据结构
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    # 写入JSON文件
    with open(json_output_path, "w") as f:
        json.dump(coco_format, f, indent=2)

    if check_format:
        checking_format.check_coco_json(json_output_path)


if __name__ == "__main__":

    # xml数据集路径
    xmls_dir = "./xmls"
    # json输出路径
    json_path = "./voc2coco.json"
    # 调用转换函数
    voc_2_coco(xmls_dir, json_path)
