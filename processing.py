import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw
from pathlib import Path
import skimage

class processing():
    
    def __init__(self, masks, rois, class_ids, objTraking):
        self.idenCyst = dict()
        self.idenPaw  = dict()
        self.allResults = dict()
        self.masks = masks
        self.rois = rois
        self.class_ids = class_ids
        self.objTrack = objTraking
        
    def centroid(self, bbox):
        startX, startY, endX, endY = bbox
        cX = int((startX + endX) / 2.0)
        cY = int((startY + endY) / 2.0)
        return [cY, cX]  #np.asarray([cY, cX])
    
    def compareTwoDict(self, original, newValue):
        
        if original != {}:
            for key, value in newValue.items():
                for key1, value1 in newValue[key].items():
                    if original[key].get(key1) == None or newValue[key][key1] > original[key][key1]:
                        original[key][key1] = newValue[key][key1]
                    else:
                        pass
                else:
                    pass
            self.allResults = original
        else:
            self.allResults = newValue
            
        return self.allResults
        

    def compute_overlaps_masks(self, masks1, masks2):
        """Computes IoU overlaps between two sets of masks.
        masks1, masks2: [Height, Width, instances]
        """

        # If either set of masks is empty return empty result
        if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
            return np.zeros((masks1.shape[-1], masks2.shape[-1]))
        # flatten masks and compute their areas
        masks1 = np.reshape(masks1, (-1, masks1.shape[-1])).astype(np.float32)
        masks2 = np.reshape(masks2, (-1, masks2.shape[-1])).astype(np.float32)
        area1 = np.sum(masks1, axis=0)
        area2 = np.sum(masks2, axis=0)

        # intersections and union
        intersections = np.dot(masks1.T, masks2)
        union = area1[:, None] + area2[None, :] - intersections
        overlaps = intersections / union

        return overlaps.any()
    
    def separateVar(self, real_test_dir, nameFile):
        areaMask, roi_Bbox, blackMask, _class = [], [], [], []
        for i in range(self.masks.shape[2]):
            temp = skimage.io.imread(os.path.join(real_test_dir, nameFile))
            for j in range(temp.shape[2]):
                temp[:,:,j] = temp[:,:,j] * self.masks[:,:,i]
                area = np.reshape(self.masks[:,:,i], (-1, self.masks[:,:,i].shape[-1])).astype(np.float32).sum()
                if area not in areaMask:
                    areaMask.append(area)
                    roi_Bbox.append(self.centroid(self.rois[i,:]))
                    _class.append(self.class_ids[i])
            blackMask.append(temp)
        return areaMask, roi_Bbox, blackMask, _class
    
    def indentifyVar(self, bbox, cclass):
        identityCate, identityP = dict(), dict()
        indexPaw, indexCyst = [], []
        for key, value in self.objTrack.items(): #key = ID, value = Centroid
            value = value.tolist()
            for item in bbox:
                idexBbox = bbox.index(item)
                if(value == item):
                    identityP[str(key)] = idexBbox
                    indexPaw.append(idexBbox)

        for item in range(len(bbox)):
            if item not in indexPaw:
                indexCyst.append(item)

        identityCate[str(cclass[indexPaw[0]])] = identityP
        return identityCate, indexCyst
    
    def checkOverlap(self, blackMask, identity, areaList, locateC, locateP):
        overlap = False
        cystList = []
        categories = dict()
        self.idenPaw[identity] = int(areaList[locateP])
        self.idenCyst[identity] = 0
        for c in locateC:
            overlap = self.compute_overlaps_masks(blackMask[c], blackMask[locateP])
            if overlap == True:
                cystList.append(c)
        if len(cystList) == 1:
            self.idenCyst[identity] = int(areaList[cystList[0]])

        if len(cystList) > 1 and len(cystList) <= 2:
            overlap = self.compute_overlaps_masks(blackMask[0], blackMask[1])
            if overlap == True:
                cyst1, cyst2 = cystList[0], cystList[1]
                union = np.reshape((self.masks[:,:,cyst1]+self.masks[:,:,cyst2]), (-1, (self.masks[:,:,cyst1]+self.masks[:,:,cyst2]).shape[-1])).astype(np.float32).sum()
                self.idenCyst[identity] = int(union)

            else:
                self.idenCyst[identity] = int(areaList[cystList[0]] + areaList[cystList[1]])
       
        else:
            pass

        categories['2'] = self.idenPaw
        categories['1'] = self.idenCyst
        return categories

    def separateGrade(self, grade):
        _grade = 0
        if grade == 0.0:
            grade = 1
        elif grade > 20.0:
            grade = 4
        elif grade > 10.0 and grade <= 20.0:
            grade = 3
        elif grade > 0.0 and grade <= 10.0:
            grade = 2

        return grade


    def rationResult(self, allResults):
        ratioDict = dict()
        gradeDict = dict()
        for ikey, ivalue in allResults.items():
            for jkey, jvalue in ivalue.items():
                cyst = allResults["1"][jkey]
                paw = allResults["2"][jkey]
                ratioDict[jkey] = (cyst/paw) * 100
                gradeDict[jkey] = self.separateGrade(int(ratioDict[jkey]))

        return {'ratio' : ratioDict, 'grade' : gradeDict}

               
    def main(self, resultByID, real_test_dir, nameFile):

        area, bbox, bMask, cclass = self.separateVar(real_test_dir, nameFile)
        _idCate, indexC = self.indentifyVar(bbox, cclass)
        for key, value in _idCate['2'].items():
            resultPerFrame = self.checkOverlap(bMask, key, area, indexC, value)
            
        self.compareTwoDict(resultByID, resultPerFrame)
        final_result = self.rationResult(self.allResults)
        return self.allResults, final_result


if __name__ == "__main__":
    pass
    # real_test_dir = "C:\\Users\\wilat\\OneDrive\\Desktop\\Real_val\\images"
    # nameFile = "40.png"
    # process = processing(r['masks'], r['rois'], r['class_ids'], objTracking)
    # resultByID = process.main(resultByID, real_test_dir, nameFile[-1])