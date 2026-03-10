from classify import voc_2_imagenet
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("--format",
                   type=str,
                   default="voc2imagenet",
                   help="目前只实现了voc2imagenet")
parse.add_argument("--input_voc_imgs_dir",
                   type=str,
                   default="",
                   help="coco转voc、yolo时需要指定json文件路径")
parse.add_argument("--input_voc_xmls_dir",
                   type=str,
                   default="",
                   help="voc转coco、yolo时需要指定xml文件夹路径")
parse.add_argument("--output_dir",
                   type=str,
                   default="output",
                   help="输出路径,包含裁剪图片、txt文件、标签文件")
parse.add_argument("--check_format",
                   type=bool,
                   default=True,
                   help="对xml文件格式进行检查，默认为True")

args = parse.parse_args()
methods = ["voc2imagenet"]


def convert(args):
    assert args.format in methods, "format must be in %s" % methods

    print("start converting ...")
    if args.format == "voc2imagenet":
        voc_2_imagenet.voc_2_imagenet(args.input_voc_imgs_dir, args.input_voc_xmls_dir,
                                      args.output_dir, args.check_format)

    print("Conversion complete ...")


if __name__ == "__main__":

    convert(args)
