import cv2
import numpy as np
import os

def get_data(input_path):
    found_bg = False
    all_imgs = {}
    
    classes_count = {}
    class_mapping = {}

    visualise = True
    DIR = '/home/rod/Desktop/Research/RodNet/Datasets/WiderFace/WIDER_train/RS_images'
    with open(input_path,'r') as f:

        #print('Parsing annotation files')
        count = 0
        for line in f:
            line_split = line.strip().split(',')
            (filename,x1,y1,x2,y2,class_name) = line_split
            imgPath = os.path.join(DIR,filename)
            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}
                img = cv2.imread(imgPath)
                #print(filename)

                (rows,cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = imgPath
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                all_imgs[filename]['imageset'] = 'trainval'
                 
                
            
#            printProgressBar(count, 159424, prefix='Parsing annotation file')
            count += 1
            all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})


        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])
		
		# make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch
        print("Done w/")
		
        return all_data, classes_count, class_mapping
"""
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):   
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

"""
