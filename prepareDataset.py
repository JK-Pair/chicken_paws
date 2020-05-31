import numpy as np 
import json
import os
import cv2
import sys
from tqdm import tqdm

class jsonFile():

    def __init__(self, path_Img, path_json):
        self.path_Img = path_Img
        self.path_json = path_json
        self.listt = []
        self.width = 640 # y
        self.hight = 480 # x

    def subcolor(self, files):
        self.colorlist = []
        self.colordict = dict()

        img = cv2.imread(self.path_Img + "\\" + files)   
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        for i in range(self.hight):
            for j in range(self.width):
                if np.array_equal(img[i][j], [0, 0, 0]) != True and str(tuple(img[i][j])) not in self.colorlist:
                    self.colorlist.append(str(tuple(img[i][j])))
                else:
                    pass

        for colors in self.colorlist:
            
            if colors == '(255, 0, 255)' or colors == '(0, 255, 0)' or colors == '(0, 0, 255)' or colors == '(100, 0, 0)':
                self.colordict[colors] = {"category": "Paw", "super_category" : "Chicken"}
            elif colors == '(255, 0, 0)' or colors == '(5, 255, 255)'  or colors == '(0, 100, 0)'or colors == '(0, 0, 100)' or colors == '(0, 255, 255)' or colors == '(155, 75, 35)' or colors == '(90, 0, 100)' or colors == '(0, 180, 70)':
                self.colordict[colors] = {"category": "Cyst", "super_category" : "Cyst"}
            else:
                pass
        
        return self.colordict

    def subimage(self, files):
        self.subimg = dict()
        self.subimg["mask"] = "masks/{}".format(files)
        self.subimg["color_categories"] = self.subcolor(files)

        return self.subimg

    def submasks(self):
        self._mask = dict()
        allFile = os.listdir(self.path_Img)

        for files in tqdm(allFile):   
            print(files)
            self._mask["images/{}".format(files)] = self.subimage(files)

        return self._mask

    def supercate(self):
        self._super = {"Cyst" : ["Cyst"],
                        "Chicken":["Paw"]
                        }

        return self._super
    def main(self):

        ownJsonFormat = dict()
        ownJsonFormat["masks"] = self.submasks() 
        ownJsonFormat["super_categories"] = self.supercate()


        # output_path = "C:\\Users\\wilat\\OneDrive\\Desktop\\Minbury_Dataset\\train\\prepareDataset.json"
        output_path = self.path_json
        with open(output_path, 'w+') as output_file:
            json.dump(ownJsonFormat, output_file)


if __name__ == "__main__":

    # jso = jsonFile(path_Img, path_json)
    # ROOT_path = "C:\\Users\\wilat\\OneDrive\\Desktop\\Augment_image\\train"
    ROOT_path = r"C:\Users\wilat\OneDrive\Desktop\Real_val\\"

    jso = jsonFile(ROOT_path + "masks", ROOT_path + "mask_definitions.json")
    jso.main()


    