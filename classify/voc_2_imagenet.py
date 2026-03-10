import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from utils import checking_format
from PIL import Image
import shutil


def voc_2_imagenet(imgs_dir, xmls_dir, output_dir, check_format=True):
    """
    参数：
    - imgs_dir: string, VOC数据集中图像文件所在的目录。
    - xmls_dir: string, VOC数据集中XML文件所在的目录。
    - output_dir: string, 输出目录，会在该目录下保存裁剪图片、txt文件、和标签文件。
    - check_format: bool, 是否检查数据集格式。
    """
    if check_format:
        checking_format.check_voc_img_xml(imgs_dir, xmls_dir)
        checking_format.check_xml_format(xmls_dir)

    # 清空输出目录，并重新创建
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    crop_imgs_dir=os.path.join(output_dir, "crop_images")
    os.mkdir(crop_imgs_dir)
    txt_output = os.path.join(output_dir, "imagenet.txt")
    label_output = os.path.join(output_dir, "labels.txt")

    crop_img_index = 0
    imgs = os.listdir(imgs_dir)

    category_set = []
    # 创建或清空输出文本文件
    with open(txt_output, "w") as f:
        for xml_file in tqdm(os.listdir(xmls_dir)):
            img_path = None
            img_name = xml_file[:-4]
            for img_ext in ["jpg", "png", "bmp"]:
                if f"{img_name}.{img_ext}" in imgs:
                    img_path = os.path.join(imgs_dir, f"{img_name}.{img_ext}")
                    break

            if xml_file.endswith(".xml"):

                img = Image.open(img_path)

                width, height = img.size
                # 解析XML文件
                xml_path = os.path.join(xmls_dir, xml_file)
                tree = ET.parse(xml_path)
                root = tree.getroot()
                if int(root.find("size").find("height").text) != height or int(
                        root.find("size").find("width").text) != width:
                    print(f"{xml_file} 记录的图片尺度与真实图片尺度不匹配.")
                    continue
                for obj in root.findall("object"):
                    category = obj.find("name").text
                    xmin = int(obj.find("bndbox").find("xmin").text)
                    xmax = int(obj.find("bndbox").find("xmax").text)
                    ymin = int(obj.find("bndbox").find("ymin").text)
                    ymax = int(obj.find("bndbox").find("ymax").text)

                    if xmin >= xmax or ymin >= ymax:
                        print(f"{xml_file} 中box坐标标注有误.")
                        continue
                    # 处理类别信息，确保每个类别有唯一的ID
                    if category not in category_set:
                        category_id = len(category_set)
                        category_set.append(category)
                    else:
                        category_id = category_set.index(category)

                    crop_img = img.crop((xmin, ymin, xmax, ymax))

                    # 写入TXT文件、存储图片
                    try:
                        crop_img_path = os.path.join(os.path.abspath(crop_imgs_dir),
                                                     f"{crop_img_index}.png")
                        f.write(f"{crop_img_path},{category_id}\n")
                        crop_img.save(crop_img_path)
                        crop_img_index += 1
                    except Exception as e:
                        print(e)
                        print(f"{xml_file} 处理失败.")
                        continue

            else:
                print(f"{xml_file} is not a valid XML file.")
    with open(label_output, "w") as f:
        for category in category_set:
            f.write(f"{category}\n")

if __name__ == "__main__":
    # 调用函数，指定XML目录和输出TXT文件路径
    xml_dir = "../data/xmls"
    imgs_dir = "../data/imgs"
    txt_output = "./imagenet.txt"
    voc_2_imagenet(imgs_dir, xml_dir, txt_output)
