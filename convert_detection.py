from detection import coco_2_voc
from detection import coco_2_yolo
from detection import voc_2_coco
from detection import voc_2_yolo
from detection import yolo_2_coco
from detection import yolo_2_voc
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("--format",
                   type=str,
                   default="coco2voc",
                   help="coco2voc、coco2yolo、voc2coco、voc2yolo、yolo2coco、yolo2voc")
parse.add_argument("--output_dir",
                   type=str,
                   default="./output",
                   help="转换为voc、yolo数据后的保存路径,是文件夹")
parse.add_argument("--input_coco_json_file",
                   type=str,
                   default="",
                   help="coco转voc、yolo时需要指定json文件路径")
parse.add_argument("--input_voc_xmls_dir",
                   type=str,
                   default="",
                   help="voc转coco、yolo时需要指定xml文件夹路径")
parse.add_argument("--output_coco_json_file",
                   type=str,
                   default="",
                   help="voc转coco时，指定要生成的json文件路径")
parse.add_argument("--input_yolo_txts_dir",
                   type=str,
                   default="",
                   help="yolo转voc、coco时需要指定txt文件夹路径")
parse.add_argument("--input_yolo_classes_file",
                   type=str,
                   default="",
                   help="yolo中存储的标签文件路径")
parse.add_argument("--categories",
                   nargs="+",
                   default=[],
                   help="voc转yolo时，指定的类别列表，如果为空，会自动生成的顺序不固定的类别列表")
parse.add_argument("--input_yolo_imgs_dir",
                   type=str,
                   default="",
                   help="yolo转voc、coco时，指定yolo标注格式的图片存放路径")
parse.add_argument("--input_coco_imgs_dir",
                   type=str,
                   default="",
                   help="coco转voc、yolo时，指定coco标注格式的图片存放路径")
parse.add_argument("--input_voc_imgs_dir",
                   type=str,
                   default="",
                   help="voc转coco、yolo时，指定voc标注格式的图片存放路径")
parse.add_argument("--check_format",
                   type=bool,
                   default=True,
                   help="检查json、xml、txt格式是否正确")
parse.add_argument("--copy_imgs", type=bool, default=True, help="是否拷贝图片")
parse.add_argument("--use_default_format",
                   type=bool,
                   default=False,
                   help="use default format")
parse.add_argument("--need_imgs", type=bool, default=True, help="是否需要处理图片")

args = parse.parse_args()
methods = ["coco2voc", "coco2yolo", "voc2coco", "voc2yolo", "yolo2coco", "yolo2voc"]


def convert(args):
    assert args.format in methods, "format must be in %s" % methods

    print("start converting ...")
    if args.format == "coco2voc":
        coco_2_voc.coco_2_voc(args.input_coco_json_file, args.output_dir,
                              args.check_format,
                              args.use_default_format,
                              args.need_imgs,
                              args.input_coco_imgs_dir,
                              args.copy_imgs)
    elif args.format == "coco2yolo":
        coco_2_yolo.coco_2_yolo(args.input_coco_json_file, args.output_dir,
                                args.check_format,
                                args.need_imgs,
                                args.input_coco_imgs_dir,
                                args.copy_imgs)
    elif args.format == "voc2coco":
        voc_2_coco.voc_2_coco(args.input_voc_xmls_dir, args.output_coco_json_file,
                              args.check_format,
                              args.need_imgs,
                              args.input_voc_imgs_dir,
                              args.copy_imgs)
    elif args.format == "voc2yolo":
        voc_2_yolo.voc_2_yolo(args.input_voc_xmls_dir,
                              args.output_dir,
                              categories=args.categories,
                              check_format=args.check_format,
                              need_imgs=args.need_imgs,
                              input_voc_imgs_dir=args.input_voc_imgs_dir,
                              copy_imgs=args.copy_imgs)
    elif args.format == "yolo2coco":
        yolo_2_coco.yolo_2_coco(args.input_yolo_txts_dir, args.input_yolo_imgs_dir,
                                args.output_coco_json_file,
                                args.input_yolo_classes_file, args.check_format,
                                args.need_imgs,
                                args.copy_imgs)
    elif args.format == "yolo2voc":
        yolo_2_voc.yolo_2_voc(args.input_yolo_txts_dir, args.input_yolo_imgs_dir,
                              args.output_dir, args.input_yolo_classes_file,
                              args.check_format,
                              args.need_imgs,
                              args.copy_imgs)

    print("Conversion complete ...")


if __name__ == "__main__":

    convert(args)
