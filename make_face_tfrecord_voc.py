# encoding: utf-8

import os
import tensorflow as tf
from lxml import etree

def parse_xml(xml_path):
    with tf.gfile.GFile(xml_path, 'r') as f:
        xml_str = f.read()
        xml = etree.fromstring(xml_str)

    # with open(xml_path, 'r') as xf:
    #     data = xf.read()
    #     # str = data.decode(encoding='utf-8')#, errors='ignore')
    #     # print(str)
    #     # str = data.decode('')
    #     parser = etree.XMLParser()
    #     e = etree.fromstring(data, parser=parser)
    #
    #     for a in e.findall('annotation'):
    #         for s in a.findall('size'):
    #             for h in s.findall('height'):
    #                 print(h, type(h))


# def parse_xml(xml_path):
#     with open(xml_path, 'rb') as xf:
#         data = xf.read()
#         str = data.decode(encoding='utf-8')#, errors='ignore')
#         # print(str)
#         # str = data.decode('')
#         parser = etree.XMLParser()
#         e = etree.fromstring(str, parser=parser)
#
#         for a in e.findall('annotation'):
#             for s in a.findall('size'):
#                 for h in s.findall('height'):
#                     print(h, type(h))

# from xml.etree import ElementTree
#
# def parse_xml(xml_path):
#     e = ElementTree.parse(xml_path, parser=ElementTree.XMLParser(encoding='us-ascii')).getroot()
#     for a in e.findall('annotation'):
#         for s in a.findall('size'):
#             for h in s.findall('height'):
#                 print(h, type(h))


if __name__=='__main__':
    IMAGE_PATH = '/Users/gglee/Data/images/37--Soccer/37_Soccer_Soccer_37_211.jpg'
    XML_PATH = os.path.splitext(IMAGE_PATH)[0] + '.xml'
    print(XML_PATH)

    if os.path.exists(XML_PATH):
        parse_xml(IMAGE_PATH)

