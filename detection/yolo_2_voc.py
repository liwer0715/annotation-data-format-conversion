import os
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring
import shutil
from PIL import Image
from tqdm import tqdm
from utils import checking_format


def yolo_2_voc(yolo_txt_dir, img_dir, xmls_dir, classes_file, check_format=True, need_imgs=True, copy_imgs=True):
    """
    将YOLO格式的TXT标注转换为VOC格式的XML标注，支持处理图片。

    参数：
    - yolo_label_dir: str, YOLO TXT标注文件的目录
    - xmls_dir: str, 输出VOC XML标注文件的目录
    - img_dir: str, 图像文件的目录
    - classes_file: str, 类别文件
    - check_format: bool, 是否检查标注格式
    - need_imgs: bool, 是否需要处理图片
    - copy_imgs: bool, 是否拷贝图片，否则移动图片
    """
    if not os.path.exists(yolo_txt_dir):
        raise Exception("YOLO格式标注文件目录不存在")
    if not os.path.exists(img_dir):
        raise Exception("图片文件目录不存在")
    if not os.path.exists(classes_file):
        raise Exception("类别文件不存在")

    # 构建存储目录
    if os.path.exists(xmls_dir):
        shutil.rmtree(xmls_dir)
    os.mkdir(xmls_dir)
    xmls_output_dir = os.path.join(xmls_dir, "xmls")
    os.mkdir(xmls_output_dir)
    if need_imgs:
        imgs_output_dir = os.path.join(xmls_dir, "imgs")
        os.mkdir(imgs_output_dir)

    # 构建标签映射列表
    categories = {}
    with open(classes_file, "r") as f:
        for line_number, line in enumerate(f):  # 使用enumerate跟踪行号,id起始是1
            class_name = line.strip()  # 去除行尾的换行符
            categories[line_number] = class_name

    if check_format:
        checking_format.check_yolo_txt_file(yolo_txt_dir)

    # 遍历图像文件，转为VOC格式
    for img_file in tqdm(sorted(os.listdir(img_dir))):
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        if img_file.endswith(tuple(image_extensions)):
            img_path = os.path.join(img_dir, img_file)
            img_name = os.path.splitext(img_file)[0]
            xml_file = os.path.join(xmls_output_dir, img_name + ".xml")

            img = Image.open(img_path)
            img_width, img_height = img.size
            img.close()

            # 创建XML文件的根元素及基本结构
            root = Element("annotation")
            folder = SubElement(root, "folder")
            folder.text = "JPEGImages"
            filename = SubElement(root, "filename")
            filename.text = img_file
            path = SubElement(root, "path")
            path.text = img_path
            source = SubElement(root, "source")
            database = SubElement(source, "database")
            database.text = "Unknown"
            size = SubElement(root, "size")
            width = SubElement(size, "width")
            width.text = str(img_width)
            height = SubElement(size, "height")
            height.text = str(img_height)
            depth = SubElement(size, "depth")
            depth.text = "3"
            segmented = SubElement(root, "segmented")
            segmented.text = "0"

            # 检查是否存在对应的TXT文件
            txt_file = os.path.join(yolo_txt_dir, img_name + ".txt")
            if not os.path.exists(txt_file):
                print(f"{txt_file} not found")
                continue

            # 读取TXT文件，逐行转换为xml要求的box存储格式
            with open(txt_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    class_id, x, y, w, h = (
                        int(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    )
                    obj = SubElement(root, "object")
                    name = SubElement(obj, "name")
                    name.text = categories[class_id]
                    pose = SubElement(obj, "pose")
                    pose.text = "Unspecified"
                    truncated = SubElement(obj, "truncated")
                    truncated.text = "0"
                    difficult = SubElement(obj, "difficult")
                    difficult.text = "0"
                    bndbox = SubElement(obj, "bndbox")
                    xmin = SubElement(bndbox, "xmin")
                    xmin.text = str(int((x - w/2) * img_width))
                    ymin = SubElement(bndbox, "ymin")
                    ymin.text = str(int((y - h/2) * img_height))
                    xmax = SubElement(bndbox, "xmax")
                    xmax.text = str(int((x + w/2) * img_width))
                    ymax = SubElement(bndbox, "ymax")
                    ymax.text = str(int((y + h/2) * img_height))
            # 保存XML文件
            xml = tostring(root, pretty_print=True)
            with open(xml_file, "wb") as f:
                f.write(xml)

            # 拷贝图片
            if need_imgs:
                source_img_path = os.path.join(img_dir, img_file)
                target_img_path = os.path.join(imgs_output_dir, img_file)
                if os.path.exists(source_img_path):
                    if copy_imgs:
                        shutil.copy(source_img_path, target_img_path)
                    else:
                        shutil.move(source_img_path, target_img_path)
                else:
                    print(f"图片文件不存在: {source_img_path}")

    if check_format:
        checking_format.check_xml_format(xmls_output_dir)


if __name__ == "__main__":
    # yolo数据的txt文件存储路径
    yolo_txt_dir = "data/labels/txts"
    # 指定输出的voc格式的xml文件存储路径
    xmls_dir = "data/out2"
    # 与yolo的txt文件对应的图像文件存储路径
    img_dir = "data/imgs"
    # yolo中记录的类别文件
    classes_file = "data/labels/classes.txt"
    yolo_2_voc(yolo_txt_dir, xmls_dir, img_dir, classes_file)
