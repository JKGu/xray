#########################################
# How To run file 
#
# python json_to_csv.py --input_directory=images\train --output_directory=data --csv_name=train.csv
#
# Created By:- Amenity Technology
#
##########################################

import os
import numpy as np
import pandas as pd
import json
import argparse
import glob

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input_directory", required=True,help="path of json file")
ap.add_argument("-o", "--output_directory", required=True,help="path where csv will be saved")
ap.add_argument("-c", "--csv_name", required=True,help="path where csv will be saved")
args = vars(ap.parse_args())
def json_to_csv(input_path,output_path,csv_name):
    '''
    input_path[directory]: must be path of json file
    output_path[directory]: path where you want to save .csv file
    '''
    xmin,xmax,ymin,ymax,height,width,name,imageName = [],[],[],[],[],[],[],[]
    print(len(input_path))
    total_length = len(input_path) + 1
    json_list = glob.glob(input_path+'\\*.json')

    for json_name in json_list:
        print(json_name)
        data = json.load(open(json_name))
        detected = False
        for x in data['shapes']:
            try:
                x1 = data['shapes'][0]['points'][0][0]
                x2 = data['shapes'][0]['points'][1][0]
                y1 = data['shapes'][0]['points'][0][1]
                y2 = data['shapes'][0]['points'][1][1]
                xmin.append(min(x1,x2))
                xmax.append(max(x1,x2))
                ymin.append(min(y1,y2))
                ymax.append(max(y1,y2))
            # print("{}:{}".format(xmax,xmin))
                height.append(abs(y1-y2))
                width.append(abs(x1-x2))
                name.append(data['shapes'][0]['label'])
                base_jsonname = os.path.basename(json_name)
                # base_imagename = os.path.basename(image_dir)
                # image_type = base_imagename.split('.')[1]
                imageName.append(base_jsonname.split('.')[0]  + '.png')
                detected = True
            except:
                pass
        if not detected:
            xmin.append(0)
            xmax.append(data["imageWidth"])
            ymin.append(0)
            ymax.append(data["imageHeight"])
        # print("{}:{}".format(xmax,xmin))
            height.append(data["imageHeight"])
            width.append(data["imageWidth"])
            name.append("notDetected")
            base_jsonname = os.path.basename(json_name)
            # base_imagename = os.path.basename(image_dir)
            # image_type = base_imagename.split('.')[1]
            imageName.append(base_jsonname.split('.')[0]  + '.png')
    
    final_df = {"class":name,"filename":imageName,"height":height,"width":width,"xmax":xmax,"xmin":xmin,"ymax":ymax,"ymin":ymin}
    df = pd.DataFrame(final_df)
    csv_path = output_path+'\\{}'.format(csv_name)
    print("csv_path: ",csv_path)
    try:
        if os.path.exists(csv_path):
            print ("File exist")
            print("Remove the file to Store New File")
            os.remove(csv_path)
        df.to_csv(csv_path)
    except:
        pass
                

def main():
    print("JSON TO CSV CODE>>>>")
if __name__=='__main__':
    json_to_csv(args['input_directory'], args['output_directory'],args['csv_name'])