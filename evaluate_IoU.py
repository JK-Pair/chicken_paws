import json

class evaluation():

    def __init__(self, coco_instances):
        self.coco_instances = r"{}\coco_instances.json".format(coco_instances)
        self.saveFileIOU = r"{}\coco_answer.json".format(coco_instances)
        self.saveconFusion = r"{}\confusion.json".format(coco_instances)
        self.bbox_image = dict()
        self.result_IoU = dict()
        self.result_confusion = dict()

    def sortList(self, bbox):
        temp = [bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]] 
        return temp

    def IOU(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
        h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
        if w_intersection <= 0 or h_intersection <= 0: # No overlap
            return 0
        I = w_intersection * h_intersection
        U = w1 * h1 + w2 * h2 - I # Union = Total Area - I
        return I / U

    def sortMaximum(self, lst, sequence):
        return sorted( [x for x in lst], reverse=True )[:sequence]
        
    def calculate_IoU(self, filename, pred, answer):
        ans_name = answer[filename]
        pred_name = pred[filename]
        cate_dict = dict()
        for key in ans_name.keys():
            temp = []
            checkPred = []
            for item in range(len(ans_name[key])):

                for jtem in range(len(pred_name[key])):
                    percen = self.IOU(ans_name[key][item], pred_name[key][jtem])
                    print("\nIOU ans: {}, pred: {}".format(ans_name[key][item], pred_name[key][jtem]))
                    print("item: {}, jtem: {}, percen {}".format(item, jtem, percen))

                    if percen >= 0.60:

                        if jtem in checkPred:
                            if percen > temp[jtem]:
                                temp[jtem] = percen
                            
                            else:
                                temp.append(percen)
                        elif jtem not in checkPred:
                            checkPred.append(jtem)
                            temp.append(percen)
                        
            print("checkPred:", checkPred)          
            print("temp: ", temp) 
            cate_dict[key] = self.sortMaximum(temp, len(ans_name[key])) 
            print("sorted: ", cate_dict[key])
            # cate_dict[key] = temp  #edit

        return cate_dict  

    def resultFormat(self, nameFile, roi, class_id):
        cate_dict = dict()
        bbox1 = []
        bbox2 = []
        for item in range(len(roi)):
            roii = roi.tolist()
            roiii = roii[item]
            if class_id[item] == 1:
                bbox1.append(roiii)
            else:
                bbox2.append(roiii)
            cate_dict["1"] = bbox1
            cate_dict["2"] = bbox2
        return {nameFile : cate_dict}

    def keepBBox(self):
        with open( self.coco_instances) as json_file:
            data = json.load(json_file)
            for item in range(len(data["images"])):
                dict_image = dict()
                file_images = data["images"][item]["file_name"]
                self.bbox_image[str(file_images)] = []
                cate_1, cate2 = [], []
                for jtem in range(len(data["annotations"])):
                    annotation = data["annotations"][jtem]
                    if data["images"][item]["id"] == annotation["image_id"]:
                        bbox = annotation["bbox"]
                        cate_id = annotation["category_id"]
                        if cate_id == 1:
                            cate_1.append(self.sortList(bbox))
                        if cate_id == 2:
                            cate2.append(self.sortList(bbox))
                        
                dict_image["1"], dict_image["2"] = cate_1, cate2
                self.bbox_image[str(file_images)] = dict_image

        return self.bbox_image

    #########New evaluation that return miss and over mask to confusion matraix.

    def compareTwoObj(self, answer, predObj, checkedObj):
        compareResult = dict()
        compareResult['miss'] = 0
        compareResult['over'] = 0
        compareResult["correct"]  = len(checkedObj) 
        print("allPred: {}, \nold {}, \ncheck {}\n".format(len(answer), len(predObj), len(checkedObj))) #6,6,5 ,,3,2,2

        if len(answer) > len(checkedObj):
            result = len(answer) - len(checkedObj)
            compareResult['miss'] = result 

            if len(predObj) > len(checkedObj):
                result =  len(predObj) - len(checkedObj)
                compareResult["over"] = result

            else:
                result = len(checkedObj) - len(predObj)
                compareResult["correct"] = len(checkedObj) - result

        elif len(answer) < len(checkedObj):
            result = len(checkedObj) - len(answer)
            compareResult['over'] = result 
            
            if len(predObj) > len(checkedObj):
                result =  len(predObj) - len(checkedObj)
                compareResult["over"] = result

            else:
                result = len(checkedObj) - len(predObj)
                compareResult["correct"] = len(checkedObj) - result

        elif len(answer) == len(checkedObj):
            result = len(answer) - len(checkedObj)
            compareResult["correct"]  = len(checkedObj)

            if len(predObj) > len(checkedObj):
                result =  len(predObj) - len(checkedObj)
                compareResult["over"] = result

            else:
                result = len(checkedObj) - len(predObj)
                compareResult["correct"] = len(checkedObj) - result
    
        # compareResult["correct"]  


        return compareResult

    def checkMaskInResult(self, nameFile, pred, ans):
        returnDict = dict()
        ans_name = ans[nameFile]
        pred_name = pred[nameFile]

        for key, value in ans_name.items():
            temp = []
            print("key: ",key)
            for eAns in value:
                for ePred in pred_name[key]:
                    status = self.IOU(eAns, ePred)
                    if key == '1':
                        if status >= 0.50:
                            temp.append(ePred)
                            break
                    elif key == '2':
                        if status >= 0.50:
                            temp.append(ePred)
                            break

            resultComp = self.compareTwoObj(ans_name[key], pred_name[key], temp)      
            returnDict[key] = resultComp

        return returnDict

    def main(self, nameFile, predFormat):
        answer = self.keepBBox()
        # predFormat = self.resultFormat(nameFile, rois, class_ids)
        self.result_IoU[nameFile] = self.calculate_IoU(nameFile, predFormat, answer)
        self.result_confusion[nameFile] = self.checkMaskInResult(nameFile, predFormat, answer)
        print("mask result: ", self.result_confusion[nameFile])

        print("\nIOU results: ", self.result_IoU)
        
    def saveJson(self):
        output_path = self.saveFileIOU
        with open(output_path, 'w+') as output_file:
            json.dump(self.result_IoU, output_file)

        confusion = self.saveconFusion
        with open(confusion, 'w+') as output_file:
            json.dump(self.result_confusion, output_file)


if __name__ == "__main__":
    val_path = r"C:/Users/wilat/OneDrive/Desktop/Real_val"
    evaluate = evaluation(val_path)
    answer = evaluate.keepBBox()
    nameFile = '22.png'
    import numpy as np
    roi = np.array([[295, 457, 314, 473],
       [213, 367, 324, 532],
       [238,  80, 343, 251],
       [235, 400, 272, 444],
       [262,  61, 300, 110],
       [275, 161, 307, 196],
       [248, 498, 260, 523],
       [301, 511, 331, 537]])
    print("22.png ", answer["22.png"])
    class_ids = np.array(([1, 2, 2, 1, 1, 1, 1, 1]))

    predFormat = evaluate.resultFormat(nameFile, roi, class_ids)
    print("predFormat: ", predFormat)
    x = evaluate.checkMaskInResult(nameFile, predFormat, answer)
    print(x)
        # return self.result_confusion
    # evaluate.main('22.png', predFormat)
    
