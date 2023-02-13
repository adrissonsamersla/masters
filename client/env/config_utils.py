import os, sys

import xml.etree.ElementTree as ET

def parse_xml(path, i):
    orig_name = os.path.join(path, "grsim.xml")
    dest_name = os.path.join(path, f".grsim_{i}.xml")

    tree = ET.parse(orig_name)
    for var in tree.getroot().iter('Var'):
        if var.attrib['name'] == "Vision multicast port":
            port = int(var.text)
            port += i
            var.text = str(port)
        elif var.attrib['name'] == "Command listen port":
            port = int(var.text)
            port += i
            var.text = str(port)
        elif var.attrib['name'] == "Blue Team status send port":
            port = int(var.text)
            port += i * 100
            var.text = str(port)
        elif var.attrib['name'] == "Yellow Team status send port":
            port = int(var.text)
            port += i * 100
            var.text = str(port)

    tree.write(dest_name)


if __name__ == '__main__':
    num = int(sys.argv[-1])
    print(f"Generating {num} itaSim config files")

    for i in range(num):
        parse_xml("./config/", i)
