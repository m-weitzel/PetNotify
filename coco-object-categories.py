import os

import sys, getopt
import json


cat_2017 = os.path.normpath('D:/Programmierung/Python/annotations_trainval2017/COCO/annotations/instances_val2017.json')


def main(argv):
	json_file = None
	try:
		opts, args = getopt.getopt(argv, "hy:")
	except getopt.GetoptError:
		print
		'coco_categories.py -y <year>'
		sys.exit(2)

	json_file = cat_2017

	if json_file is not None:
		with open(json_file, 'r') as COCO:
			js = json.loads(COCO.read())
			print(json.dumps(js['categories']))

	with open('data.txt', 'w') as outfile:
		json.dump(json.dumps(js['categories']), outfile)

if __name__ == "__main__":
	main(sys.argv[1:])