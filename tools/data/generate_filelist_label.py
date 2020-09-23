import os
import argparse
import xml.etree.ElementTree as ET

def main():
    parser = argparse.ArgumentParser(
        description='This script support generating file list and labels of voc data.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='path to annotation files directory.')
    parser.add_argument('--output_filelist', type=str, default='trainval.txt',
                        help='path to output file list.')
    parser.add_argument('--ext', action='store_true',
                        help="file list include extension or not")
    parser.add_argument('--label', action='store_true',
                        help="generate labels file or not, only support xml files")
    parser.add_argument('--output_labels', type=str, default='labels.txt',
                        help='path to labels file.')
    args = parser.parse_args()

    total_xml = os.listdir(args.data_dir)
    
    with open(args.output_filelist, 'w') as filelist:
        for i in total_xml:
            # file list
            if args.ext:
                file_name = i + '\n'
            else:
                file_name = os.path.splitext(i)[0] + '\n'
            filelist.write(file_name)

    if args.label:
        with open(args.output_labels, 'w') as labels:
            labellist = set()
            for i in total_xml:
                if i[-3:] != 'xml':
                    raise Exception('generating labels file only support xml files')

                with open(os.path.join(args.data_dir, i), 'r') as f:
                    tree = ET.parse(f)
                for obj in tree.findall("object"):
                    cls = obj.find("name").text
                    labellist.add(cls)
        
            for i in list(labellist):
                label_name = i + '\n'
                labels.write(label_name)

if __name__ == '__main__':
    main()