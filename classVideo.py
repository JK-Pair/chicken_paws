"""
MIT License
Copyright (c) 2019 Adam Kelly
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import cv2
import skimage
import random
import colorsys
from tqdm import tqdm
import numpy as np
from processing import *
from database import*
from pyimagesearch.centroidtracker import CentroidTracker
import json
SettingPath = os.path.join(os.path.dirname(__file__), 'SettingModelPath.json')
with open(SettingPath) as json_file:
    data = json.load(json_file)
database_path = data['database']
ROOT_DIR = 'C:\\Users\\wilat\\Downloads\\Mask_RCNN-master\\Mask_RCNN-master'
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append(ROOT_DIR) 
Project_DIR = os.path.join(ROOT_DIR, "chicken_paws_project")


ct = CentroidTracker()

class videoGenarator():

    def __init__(self, model, dataset_train, video_file, video_save_dir, vid_name, v_format, fourcc):
        self.model = model
        self.dataset_train = dataset_train
        self.video_file = video_file
        self.video_save_dir = video_save_dir
        self.vid_name = vid_name
        self.v_format = v_format
        self.fourcc = fourcc

    def checkPawForPredict(self, numOfClass, class_ids):
        status = True
        _class = class_ids.tolist()
        if numOfClass not in _class:
            status = False
        return status

    def sortt(self, val):
        return val[1]

    def sortBbox(self, listBbox): 
        listBbox.sort(key = self.sortt, reverse = True)
        return listBbox

    def roi_detect_paw(self, roi, ids):
        roi_class_scoreII = []
        for count in range(len(roi)):
            tempI = self.dataset_train.class_names[ids[count]]
            if(tempI == 'Paw'):
                tempII = roi[count]
                roi_class_scoreII.append(tempII)
                
        _sorted = self.sortBbox(roi_class_scoreII)
        return _sorted

    def identity_paw(self, image, rects):
        objects = ct.update(rects)
        for (objectID, centroid) in objects.items():
            text = "ID {}".format(objectID)
            image = cv2.putText(image, text, (centroid[0], centroid[1]),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)
            
        return objects, image

    def random_colors(self, N, bright=True):
        """ Generate random colors. 
            To get visually distinct colors, generate them in HSV space then
            convert to RGB.
        Args:
            N: the number of colors to generate
            bright: whether or not to use bright colors
        Returns:
            a list of RGB colors, e.g [(0.0, 1.0, 0.0), (1.0, 0.0, 0.5), ...]
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def apply_mask(self, image, mask, color, alpha=0.5):
        """ Apply the given mask to the image.
        Args:
            image: a cv2 image
            mask: a mask of which pixels to color
            color: the color to use
            alpha: how visible the mask should be (0 to 1)
        Returns:
            a cv2 image with mask applied
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                    image[:, :, c] *
                                    (1 - alpha) + alpha * color[c] * 255,
                                    image[:, :, c])
        return image

    def display_instances(self, image, boxes, masks, ids, names, scores, colors):
        """ Take the image and results and apply the mask, box, and label
        Args:
            image: a cv2 image
            boxes: a list of bounding boxes to display
            masks: a list of masks to display
            ids: a list of class ids
            names: a list of class names corresponding to the ids
            scores: a list of scores of each instance detected
            colors: a list of colors to use
        Returns:
            a cv2 image with instances displayed   
        """
        n_instances = boxes.shape[0]

        if not n_instances:
            return image # no instances
        else:
            assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

        for i, color in enumerate(colors):
            # Check if any boxes to show
            if not np.any(boxes[i]):
                continue
            
            # Check if any scores to show
            if scores is not None:
                score = scores[i] 
            else:
                score = None

            # Add the mask
            image = self.apply_mask(image, masks[:, :, i], color)
            
            # Add the bounding box
            y1, x1, y2, x2 = boxes[i]
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            
            # Add the label
            label = names[ids[i]]
            if score:
                label = f'{label}'#{score:.2f}'
                
            label_pos = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2) # center of bounding box
            image = cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        
            if names[ids[i]] == "Paw":
                objTracking, image = self.identity_paw(image, self.roi_detect_paw(boxes, ids))

                
        return objTracking, image

    def displayInstances(self, image, boxes, masks, ids, names, scores, colors):
        """ Take the image and results and apply the mask, box, and label
        Args:
            image: a cv2 image
            boxes: a list of bounding boxes to display
            masks: a list of masks to display
            ids: a list of class ids
            names: a list of class names corresponding to the ids
            scores: a list of scores of each instance detected
            colors: a list of colors to use
        Returns:
            a cv2 image with instances displayed   
        """
        n_instances = boxes.shape[0]

        if not n_instances:
            return image # no instances
        else:
            assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

        for i, color in enumerate(colors):
            # Check if any boxes to show
            if not np.any(boxes[i]):
                continue
            
            # Check if any scores to show
            if scores is not None:
                score = scores[i] 
            else:
                score = None

            # Add the mask
            image = self.apply_mask(image, masks[:, :, i], color)
            
            # Add the bounding box
            y1, x1, y2, x2 = boxes[i]
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            
            # Add the label
            label = names[ids[i]]
            if score:
                label = f'{label}'#{score:.2f}'
                
            label_pos = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2) # center of bounding box
            image = cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
                
        return image
        

    def main(self, saveImg = False, pathImg = r"C:\Users\wilat\OneDrive\Desktop\AugmentV\test\ResNet198images"):

        ############################################################
        #  Random colors
        ############################################################
        colors = self.random_colors(25)
        # colors = [(1.0, 1.0, 0.0)] * 30

        # Change color representation from RGB to BGR before displaying instances
        colors = [(color[2], color[1], color[0]) for color in colors]


        ############################################################
        #  Genarate Video
        ############################################################
        input_video = cv2.VideoCapture(str(self.video_file))
        frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(input_video.get(cv2.CAP_PROP_FPS))
        output_video = None
        vid_size = None
        current_frame = 0

        print("frame_count: ", frame_count)
        print("fps: ", fps)
        writeFrame = 0
        resultByID = dict()
        for i in tqdm(range(frame_count)):
            # Read the current frame
            ret, frame = input_video.read()
            if not ret:
                break
                
            current_frame += 1
            
            # Change color representation from BGR to RGB before running model.detect()
            detect_frame = frame[:, :, ::-1]    
            
            # start = time.time()
 
            # Run inference on the color-adjusted frame
            results = self.model.detect([detect_frame], verbose=0)
            r = results[0]
            n_instances = r['rois'].shape[0]
            # print("Time : ", time.time() - start)
            # Make sure we have enough colors
            if len(colors) < n_instances:
                # not enough colors, generate more
                more_colors = self.random_colors(n_instances - len(colors))
                
                # Change color representation from RGB to BGR before displaying instances
                more_colors = [(color[2], color[1], color[0]) for color in more_colors]
                colors += more_colors
                

            class_Paw = 2
            status = self.checkPawForPredict(class_Paw, r['class_ids'])  #IF that frame doesn't have an paw.
            if status == True or current_frame == 1:
                objTracking, display_frame = self.display_instances(frame, r['rois'], r['masks'], r['class_ids'], 
                                                self.dataset_train.class_names, r['scores'], colors[0:n_instances])

                # process = processing(r['masks'], r['rois'], r['class_ids'], objTracking)
                if current_frame == 1 or current_frame % 10 == 0:
                    writeFrame += 1
                    process = processing(r['masks'], r['rois'], r['class_ids'], objTracking)
                    assert os.path.exists(Project_DIR), 'Project_DIR for save each frames does not exist.)'
                    image_DIR = os.path.join(Project_DIR, "images")
                    cv2.imwrite(r"{}\0.png".format(image_DIR), display_frame)
                    # print("\nBefore compare reults: ", resultByID)
                    print("obj: ", objTracking)
                    resultByID, final_result = process.main(resultByID, image_DIR, "0.png")
                    # print("After compare reults: ", resultByID)
                    print("final_result: ", final_result)

                    ####################TEMP##################################################
                    for key, value in objTracking.items():
                        value = value.tolist()
                        if str(key) in final_result['grade']:
                            grade_ID = final_result['grade'][str(key)]
                            print("grade_ID", grade_ID)
                            display_frame = cv2.putText(display_frame, str(grade_ID), (value[0], value[1]), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2)
                    cv2.imwrite(r"{}\{}.png".format(self.video_save_dir, writeFrame), display_frame)



                    # imgShow = cv2.imread(r"{}\0.png".format(image_DIR))
                    # cv2.imshow("{}".format(writeFrame), imgShow)
                    # if cv2.waitKey(0) == 27:
                        # cv2.destroyAllWindows()
                    # database_DIR = os.path.join(Project_DIR, 'FPD_Database_.db')
                    
                    dbFPD = databaseFPD(database_path)
                    dbFPD.main(resultByID, final_result)
            
            else:
                display_frame = self.displayInstances(frame, r['rois'], r['masks'], r['class_ids'], 
                                                self.dataset_train.class_names, r['scores'], colors[0:n_instances])
            # Make sure we got displayed instances
            if display_frame is not None:
                frame = display_frame

            # Create the output_video if it doesn't yet exist
            if output_video is None:
                if vid_size is None:
                    vid_size = frame.shape[1], frame.shape[0]
                output_video = cv2.VideoWriter(str(self.vid_name), self.fourcc, float(fps), vid_size, True)
                
            # Resize frame if necessary
            if vid_size[0] != frame.shape[1] and vid_size[1] != frame.shape[0]:
                frame = cv2.resize(frame, vid_size)
            
            # Write the frame to the output_video
            output_video.write(frame)
            # if current_frame == 1 or current_frame % 10 == 0:
            #     writeFrame += 1
            #     assert os.path.exists(Project_DIR), 'Project_DIR for save each frames does not exist.)'
            #     image_DIR = os.path.join(Project_DIR, "images")
            #     cv2.imwrite(r"{}\0.png".format(image_DIR), display_frame)
        
            #     resultByID, final_result = process.main(resultByID, image_DIR, "0.png")
            #     # database_DIR = os.path.join(Project_DIR, 'FPD_Database_.db')
            #     dbFPD = databaseFPD(database_path)
            #     dbFPD.main(resultByID, final_result)

            # if saveImg == True:
                # cv2.imwrite(r"{}\{}.png".format(self.video_save_dir, writeFrame), display_frame)
      

        output_path = r"C:\Users\wilat\OneDrive\Desktop\images_II\ratioByID.json"

        with open(output_path, 'w+') as output_file:
            # json.dump(resultByID, output_file)
            json.dump(final_result, output_file)
            

        input_video.release()
        output_video.release()

if __name__ == "__main__":
    print("DIR: ", Project_DIR)