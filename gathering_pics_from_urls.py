import urllib.request
import cv2
import numpy as np
import os
import sys
import progressbar
from pprint import pprint

__version__ = '1.0.1'
__update_date__ = 'Dec-29-2018'

helper = '''
[HELP]

Example:
If you want to gather pics from urls from 'https://gs.com/s' (actually from ImageNet urls) and store the images with the syntax likes this 'a-1.jpg' ,'a-2.jpg' and so on. All the images are stored in out_dir directory. If it does not exist, it will be automatically created. The amount of pictures to gather is set to max of the number of urls from that url file. It just gathers , does not operate any other stuffs like resize or convert to grayscale

\''' python3 gathering_pics_from_urls.py url=[]https://gs.com/s prefix=a out_dir=datasets/neg \'''


If you want to specify all the gathered images, you need to set 'keepOrg=False' and resize to fixed size with 'size=100,100' (width and heigh respectively)  and convert to grayscale 'cvtGray=True'

\''' python3 gathering_pics_from_urls.py url=[]https://gs.com/s prefix=a out_dir=datasets/neg num_gathering=1000 keepOrg=False size=(100,100) cvtGray=True \'''
'''

def gathering_pics_from_urls(url, prefix, out_dir, num_gathering=1000,keepOrg=False,size=(100,100),cvtGray=True):
	'''
	Args:
		url : url of the file that contains urls of images
		prefix : format name for images for ex: prefix = 'a' , image format is 'a-1.jpg', 'a-2.jpg',...
		out_dir : path of directory that contains the downloaded images, it will automatically create the directory if it does not exist
		num : the amount of images that will be gathered
		keepOrg : if True, all the images are reserved the actual size
			if False, 'size', 'cvtGray' are used
		size : if keepOrg is False, all the images are scaled to the fixed size
		cvtGray : if keepOrg is False, all the images are converted to grayscale

	Returns: None
	'''

	neg_image_urls = urllib.request.urlopen(url).read().decode().split('\n')

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	if type(num_gathering) == str: # 'max'
		num_gathering = len(neg_image_urls)

	if len(neg_image_urls) < num_gathering:
		num_gathering = len(neg_image_urls)

	bar = progressbar.ProgressBar(maxval=num_gathering, widgets=[progressbar.Bar('=','[',']'),' ', progressbar.Percentage()])
	bar.start()

	print('[INFO] Gathering ', num_gathering,'/',len(neg_image_urls), 'images')

	num_gathered = 0
	for pic_num, url in enumerate(neg_image_urls[:num_gathering]):
		try:
			img_path = out_dir + '/' + prefix+'-'+str(pic_num) +'.jpg'
			urllib.request.urlretrieve(url, img_path)
			if not keepOrg:
				if cvtGray:
					img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
				resized_image = cv2.resize(img, size)
				cv2.imwrite(img_path,resized_image)
			num_gathered += 1
		except Exception as e:
			# print('[ERROR] 1 error occurs!')
			pass
		bar.update(pic_num+1)

	bar.finish()
	print('[INFO] Gathering Done ! Total ', num_gathered , ' pics ')

def main(dict_args):
	url = dict_args['url']
	prefix = dict_args['prefix']
	out_dir = dict_args['out_dir']
	num_gathering = dict_args['num_gathering']
	keepOrg = dict_args['keepOrg']
	size = dict_args['size']
	cvtGray = dict_args['cvtGray']
	print('[INFO] Initial Settings: ')
	pprint(dict_args)
	gathering_pics_from_urls(url,prefix,out_dir,num_gathering=num_gathering,keepOrg=keepOrg,size=size,cvtGray=cvtGray)

def print_e(s=' '):
	print('[ERROR] Extracting args failed ! ', s)
	print(helper)

def extract_args(lst_args):
	'''
	Args:
		lst_args : list of args ['datasets/neg', 'num_gathering=1000']

	Default Values:
		num_gathering='max'
		keepOrg=False
		size=(100,100)
		cvtGray=True

	Returns:
		ret : if True, extracting successfully, otherwise failed somehow
		dict_args: dictionary of args {'out_dir':'datasets/neg', 'num_gathering':1000}
	'''

	def val(dt):
		return dt.split('=')[1]
	def key(dt):
		return dt.split('=')[0]

	dict_args = dict()
	for arg in lst_args[1:]:
		if key(arg) == 'url':
			try:
				#print('[DEBUG]',arg,arg.split('\''))
				dict_args[key(arg)] = arg.split('=[]')[1]
			except Exception as e:
				print_e(s='Please put url with syntax url=[]https://something' + str(e))
				return False, dict_args
		else:
			dict_args[key(arg)] = val(arg)

	try:
		dict_args['url']
	except:
		print_e(s='Missing url argument')
		return False, dict_args

	try:
		dict_args['out_dir']
	except:
		print_e(s='Missing out_dir argument')
		return False, dict_args

	try:
		dict_args['prefix']
	except:
		print_e(s='Missing prefix argument')
		return False, dict_args

	try:
		v = dict_args['num_gathering']
		if v != 'max':
			dict_args['num_gathering'] = int(v)
	except:
		dict_args['num_gathering']='max'

	try:
		dict_args['keepOrg'] = bool(dict_args['keepOrg'])
	except:
		dict_args['keepOrg'] = False

	try:
		dict_args['size'] = tuple(map(int,dict_args['size'].split(',')))
	except:
		dict_args['size'] = (100,100)

	try:
		dict_args['cvtGray'] = bool(dict_args['cvtGray'])
	except:
		dict_args['cvtGray'] = True

	return True, dict_args

if __name__ == '__main__':

	ret, dict_args = extract_args(sys.argv)

	if ret:
		main(dict_args)
