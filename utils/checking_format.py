from pycocotools.coco import COCO
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm


def check_coco_json(file_path):
    """
    检查COCO JSON文件格式是否正确

    参数：
    - file_path：str - COCO JSON文件的路径
    """
    # 尝试加载COCO JSON文件
    try:
        # 尝试加载COCO JSON文件
        coco = COCO(file_path)

        # 如果没有抛出异常，说明文件格式大致正确
        print("COCO JSON 文件格式看起来是正确的。")

        # 可以进一步检查数据的完整性，例如检查是否有必需的字段
        # 例如，检查'images'和'annotations'是否存在
        if "images" not in coco.dataset:
            print("警告：JSON文件缺少'images'字段。")
        if "annotations" not in coco.dataset:
            print("警告：JSON文件缺少annotations'字段。")
        if "categories" not in coco.dataset:
            print("警告：JSON文件缺少'categories'字段。")
    except Exception as e:
        # 如果加载过程中出现任何错误，打印错误信息
        print(f"检查COCO JSON文件时发生错误：{e}")
    else:
        print("JSON文件格式检查完成。")


def check_xml_format(xmls_dir):
    """
    检查XML文件格式是否正确

    参数：
    - xml_file：str - XML文件的路径
    - root：Element - XML文件的根元素，ET对象
    """
    for i in tqdm(os.listdir(xmls_dir)):
        one_xml_path = os.path.join(xmls_dir, i)
        tree = ET.parse(one_xml_path)
        root = tree.getroot()
        try:
            # 检查根元素是否正确
            if root.tag != "annotation":
                raise ValueError('Root element should be "annotation"')

            # 检查必要的子元素存在
            for elem_name in ["filename", "size"]:
                if root.find(elem_name) is None:
                    raise ValueError(f'Missing "{elem_name}" element')

            height = int(root.find("size").find("height").text)
            width = int(root.find("size").find("width").text)
            if height <= 0 or width <= 0:
                raise ValueError("宽高不能小于等于0")

            # 对于每个对象元素，检查必要的属性
            objects = root.findall("object")
            for obj in objects:
                for attr in ["name", "bndbox"]:
                    if obj.find(attr) is None:
                        raise ValueError(f'Missing "{attr}" in object')

                # 检查边界框坐标（x, y, width, height）
                bndbox = obj.find("bndbox")
                for coord in ["xmin", "ymin", "xmax", "ymax"]:
                    if bndbox.find(coord) is None:
                        raise ValueError(f'Missing "{coord}" in bounding box')

        except (ET.ParseError, ValueError) as e:
            print(f'XML文件 "{one_xml_path}" 格式错误: {e}')
    print("xml文件检查完成。")


def check_yolo_txt_file(txts_dir, num_classes=80):
    """
    检查YOLO格式的txt文件是否正确

    参数：
    - txts_dir：str - txt文件夹的路径
    - num_classes：int - 类别数量，默认为80
    """

    for i in tqdm(os.listdir(txts_dir)):
        one_txt_path = os.path.join(txts_dir, i)
        try:
            # 打开txt文件并逐行读取
            with open(one_txt_path, "r") as file:
                lines = file.readlines()
            errors = []

            for line_number, line in enumerate(lines, start=1):
                parts = line.strip().split()

                # 检查每一行是否分为5个部分
                if len(parts) != 5:
                    errors.append(f"文件 {one_txt_path} 的第 {line_number} 行错误: 不是5个值。")
                    continue

                try:
                    class_id, x_center, y_center, width, height = map(float, parts)
                except ValueError as e:
                    errors.append(f"文件 {one_txt_path} 的第 {line_number} 行错误: 无法转换为浮点数。")

                # 检查class_id是否在合理范围内
                if not (0 <= class_id < num_classes):
                    errors.append(
                        f"文件 {one_txt_path} 的第 {line_number} 行错误: class_id ({class_id}) 不在0到{num_classes-1}之间。"
                    )

                # 检查比例值是否在[0, 1]之间
                for value, name in zip(
                    [x_center, y_center, width, height],
                    ["x_center", "y_center", "width", "height"],
                ):
                    if not (0 <= value <= 1):
                        errors.append(
                            f"文件 {one_txt_path} 的第 {line_number} 行错误: {name} ({value}) 不在0到1之间。"
                        )
        except Exception as e:
            print(f'TXT文件 "{one_txt_path}" 存在格式错误: {e}')

        if errors:
            print(f"文件 {one_txt_path} 格式错误:")
            for error in errors:
                print(error)
            print("\n")
    print("txt文件检查完成。")


def check_voc_img_xml(imgs_dir, xmls_dir):
    """
    检查voc的图片和xml文件是否一一对应
    参数：
    - imgs_dir：str - 图片文件夹的路径
    - xmls_dir：str - xml文件夹的路径
    """
    imgs = os.listdir(imgs_dir)
    xmls = os.listdir(xmls_dir)
    print(f"图片数量是:{len(imgs)}")
    print(f"xml数量是:{len(xmls)}")

    for i in tqdm(xmls):
        img_path = None
        for img_ext in ["jpg", "png", "bmp"]:
            img_name = i.split(".")[0]
            if f"{img_name}.{img_ext}" in imgs:
                img_path = os.path.join(imgs_dir, f"{img_name}.{img_ext}")
                break
        if img_path is None:
            print(f"{i}对应的图片不存在")
            continue


if __name__ == "__main__":

    # 使用函数检查XML文件
    xmls_dir = r"../../data\xmls"

    check_xml_format(xmls_dir)

    # 使用函数检查指定的TXT文件
    txts_dir = "../../data/txts/txts"
    check_yolo_txt_file(txts_dir, 2)
