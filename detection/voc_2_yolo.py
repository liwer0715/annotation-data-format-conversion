import os
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm
from utils import checking_format


def voc2yolo_convert(size, box):
    """
    将VOC格式的box标注转换为YOLO格式的box标注

    参数:
    - size: 图片尺寸，形如(w, h)
    - box: 框的坐标，形如(xmin, ymin, xmax, ymax)

    返回值:
    - 返回一个四元组(x_center, y_center, width_bbox, height_bbox)，分别代表框的中心点坐标、
      宽度和高度在图片尺寸归一化后的值
    """

    x_center = (box[0] + box[2]) / 2.0 / size[0]
    y_center = (box[1] + box[3]) / 2.0 / size[1]
    width_bbox = (box[2] - box[0]) / size[0]
    height_bbox = (box[3] - box[1]) / size[1]
    return x_center, y_center, width_bbox, height_bbox


def voc_2_yolo(xmls_dir, yolo_label_dir, categories=[], check_format=True, need_imgs=False, input_voc_imgs_dir=None, copy_imgs=True):
    """
    将VOC的XML标注转换为YOLO格式的TXT标注，支持处理图片

    - xmls_dir: str, VOC XML标注文件的目录
    - yolo_label_dir: str, 输出YOLO TXT标注文件的目录
    - categories: list, 类别列表，如果不指定，就会按照加载顺序动态生成，确保与YOLO模型训练时使用的类别顺序一致
    - check_format: bool, 是否检查标注文件格式，默认为True
    - need_imgs: bool, 是否需要处理图片
    - input_voc_imgs_dir: str, VOC格式图片文件夹的路径
    - copy_imgs: bool, 是否拷贝图片，否则移动图片
    """

    if need_imgs and not os.path.exists(input_voc_imgs_dir):
        raise Exception("VOC格式图片文件夹不存在")
    if not os.path.exists(xmls_dir):
        raise Exception("VOC格式XML文件文件夹不存在")

    if check_format:
        checking_format.check_xml_format(xmls_dir)

    # 构建输出目录
    if os.path.exists(yolo_label_dir):
        shutil.rmtree(yolo_label_dir)
    os.mkdir(yolo_label_dir)
    txt_output_dir = os.path.join(yolo_label_dir, "txts")
    os.mkdir(txt_output_dir)
    if need_imgs:
        os.mkdir(os.path.join(yolo_label_dir, "imgs"))

    # 遍历XML文件，转换为txt文件
    for voc_file in tqdm(sorted(os.listdir(xmls_dir))):
        if voc_file.endswith(".xml"):
            xml_file = os.path.join(xmls_dir, voc_file)
            tree = ET.parse(xml_file)
            root = tree.getroot()

            filename = root.find("filename").text
            width = int(root.find("size").find("width").text)
            height = int(root.find("size").find("height").text)

            txt_file = os.path.join(txt_output_dir,
                                    os.path.splitext(filename)[0] + ".txt")
            with open(txt_file, "w") as f:
                for obj in root.findall("object"):
                    class_name = obj.find("name").text
                    if class_name not in categories:
                        categories.append(class_name)

                    bbox = obj.find("bndbox")
                    xmin = float(bbox.find("xmin").text)
                    ymin = float(bbox.find("ymin").text)
                    xmax = float(bbox.find("xmax").text)
                    ymax = float(bbox.find("ymax").text)

                    # 转换为YOLO格式的归一化坐标
                    x_center, y_center, width_bbox, height_bbox = voc2yolo_convert(
                        (width, height), [xmin, ymin, xmax, ymax])

                    f.write(
                        f"{categories.index(class_name)} {x_center} {y_center} {width_bbox} {height_bbox}\n"
                    )

            # 拷贝图片
            if need_imgs:
                source_img_path = os.path.join(input_voc_imgs_dir, filename)
                target_img_path = os.path.join(yolo_label_dir, "imgs", filename)
                if os.path.exists(source_img_path):
                    if copy_imgs:
                        shutil.copy(source_img_path, target_img_path)
                    else:
                        shutil.move(source_img_path, target_img_path)
                else:
                    print(f"图片文件不存在: {source_img_path}")

    # 生成标签文件
    with open(os.path.join(yolo_label_dir, "classes.txt"), "w") as f:
        for category in categories:
            f.write(f"{category}\n")

    if check_format:
        checking_format.check_yolo_txt_file(txt_output_dir)


if __name__ == "__main__":

    # xml的路径
    voc_annotation_dir = "data/xmls"
    # yolo格式的标注数据存储路径
    yolo_label_dir = "data/labels"
    voc_2_yolo(voc_annotation_dir, yolo_label_dir)
