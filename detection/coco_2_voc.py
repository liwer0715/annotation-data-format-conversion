import os
from lxml.etree import Element, SubElement, tostring
import xml.etree.ElementTree as ET
from pycocotools.coco import COCO
import shutil
from tqdm import tqdm
from utils import checking_format


def make_voc_dir(output_dir):
    # labels 目录若不存在，创建labels目录。若存在，则清空目录
    if not os.path.exists(os.path.join(output_dir, "./VOC2007/Annotations")):
        os.makedirs(os.path.join(output_dir, "./VOC2007/Annotations"))
    if not os.path.exists(os.path.join(output_dir, "./VOC2007/ImageSets")):
        os.makedirs(os.path.join(output_dir, "./VOC2007/ImageSets"))
        os.makedirs(os.path.join(output_dir, "./VOC2007/ImageSets/Main"))
    if not os.path.exists(os.path.join(output_dir, "./VOC2007/JPEGImages")):
        os.makedirs(os.path.join(output_dir, "./VOC2007/JPEGImages"))
    return os.path.join(output_dir, "./VOC2007/Annotations"), os.path.join(
        output_dir, "./VOC2007/JPEGImages")


def coco_2_voc(
    coco_json_path=None,
    output_dir="output",
    check_format=True,
    use_default_format=False,
    need_imgs=False,
    coco_imgs_path=None,
    copy_imgs=True,
):
    """
    将COCO格式的标注数据转换为PASCAL VOC XML格式，每个xml名和图片名相同。

    参数:
    - coco_json_path: str, COCO格式标注文件的完整路径，必须指定。
    - output_dir: str, 转换后的xml文件和图片输出保存的目录，必须指定。
    - check_format: bool, 默认为True，是否检查格式。
    - use_default_format: bool, 是否使用VOC官方格式存储。
    - need_imgs: bool, 默认为False，是否需要移动或者拷贝图片。
    - coco_imgs_path: str, COCO格式图片文件夹的完整路径。如果need_imgs为True，则必须指定。
    - copy_imgs: bool, 默认True，将图片文件复制到输出目录，否则采用移动的方式。
    """
    if need_imgs and not os.path.exists(coco_imgs_path):
        raise Exception("COCO格式图片文件夹不存在")
    if not os.path.exists(coco_json_path):
        raise Exception("COCO格式json文件不存在")

    if check_format:
        checking_format.check_coco_json(coco_json_path)

    imgs_dir = ""
    xmls_dir = ""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    if use_default_format:
        # 生成默认格式
        xmls_dir, imgs_dir = make_voc_dir(output_dir)
    else:
        os.mkdir(os.path.join(output_dir, "xmls"))
        os.mkdir(os.path.join(output_dir, "imgs"))
        xmls_dir = os.path.join(output_dir, "xmls")
        imgs_dir = os.path.join(output_dir, "imgs")

    # 加载COCO JSON数据
    coco = COCO(coco_json_path)
    images = coco.dataset["images"]
    annotations = coco.dataset["annotations"]

    # 创建XML文件
    for img in tqdm(images):
        img_id = img["id"]
        img_path = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]

        # 找到对应图像的所有注解
        img_anns = [ann for ann in annotations if ann["image_id"] == img_id]

        # 创建XML元素树
        root = Element("annotation")
        folder = SubElement(root, "folder")
        folder.text = "VOC2007"
        filename = SubElement(root, "filename")
        filename.text = img_path
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

        for ann in img_anns:
            category = coco.loadCats(ann["category_id"])[0]
            category_name = category["name"]
            iscrowd = ann.get("iscrowd", 0)

            obj = SubElement(root, "object")
            name = SubElement(obj, "name")
            name.text = category_name
            pose = SubElement(obj, "pose")
            pose.text = "Unspecified"
            truncated = SubElement(obj, "truncated")
            truncated.text = "0"
            difficult = SubElement(obj, "difficult")
            difficult.text = str(iscrowd)
            bndbox = SubElement(obj, "bndbox")
            x = SubElement(bndbox, "xmin")
            x.text = str(int(ann["bbox"][0]))
            y = SubElement(bndbox, "ymin")
            y.text = str(int(ann["bbox"][1]))
            xmax = SubElement(bndbox, "xmax")
            xmax.text = str(int(ann["bbox"][0] + ann["bbox"][2]))
            ymax = SubElement(bndbox, "ymax")
            ymax.text = str(int(ann["bbox"][1] + ann["bbox"][3]))

        # 保存XML文件
        xml_path = os.path.join(xmls_dir, f"{img_path[:-4]}.xml")
        if need_imgs:
            source_img_path = os.path.join(coco_imgs_path, img_path[:-4])
            target_img_path = os.path.join(imgs_dir, img_path)
            if not os.path.exists(source_img_path):
                print(f"{source_img_path} not exists")
                continue
            if copy_imgs:
                shutil.copy(source_img_path, target_img_path)
            else:
                shutil.move(source_img_path, target_img_path)

        xml = tostring(root, pretty_print=True)  # 格式化显示，该换行的换行
        with open(xml_path, "wb") as f:
            f.write(xml)

    if check_format:
        checking_format.check_xml_format(xmls_dir)


if __name__ == "__main__":
    help(coco_2_voc)
    # coco数据集的json文件路径
    json_path = "data/voc2coco.json"
    # coco数据集的图片文件夹路径
    img_path = ".data/imgs"
    # 转换VOC后的保存路径
    output_path = "data/out1"

    # 调用转换函数
    coco_2_voc(json_path, img_path, output_path)
