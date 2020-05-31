import cv2
import numpy as np
import math
import os
class rotate_trans:
    def __init__(self, image):
        self.img = image
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.ch   = image.shape[2]

    def rotation(self):
        M = cv2.getRotationMatrix2D((self.width/2,self.height/2),45,1)
        dst = cv2.warpAffine(self.img,M,(self.width,self.height))
        return dst

    def wrap(self, list_point_input, list_point_output):

        pts1 = np.float32(list_point_input)
        pts2 = np.float32(list_point_output)

        M = cv2.getAffineTransform(pts1,pts2)
        dst = cv2.warpAffine(self.img,M,(self.width, self.height))
        return dst

    #Wrapper of Rotation a Image
    def rotate_along_axis(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):

        #Get radius of rotationd along 3 axes.
        rtheta, rphi, rgamma = self.get_rad(theta, phi, gamma)

        #Get ideal focal length on z axis
        # Note: Change this sections to other axis.
        d = np.sqrt(self.height**2 + self.width**2)
        self.focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz = self.focal

        #Get projection matrix
        mat = self.get_rotate(rtheta, rphi, rgamma, dx, dy, dz)

        return cv2.warpPerspective(self.img.copy(), mat, (self.width, self.height))
    
    #Get Perspective Projection Matrix
    def get_rotate(self, theta, phi, gamma, dx, dy, dz):

        w = self.width
        h = self.height
        f = self.focal

        #Projection 2D --> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])

        # Rotation matrix around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(theta), -np.sin(theta), 0],
                        [0, np.sin(theta), np.cos(theta), 0],
                        [0, 0, 0, 1]])

        RY = np.array([ [np.cos(phi), 0, np.sin(phi), 0],
                        [0, 1, 0, 0],
                        [-np.sin(phi), 0, np.cos(phi), 0],
                        [0, 0, 0, 1]])

        RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                        [np.sin(gamma), np.cos(gamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        #Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        #Translation matrix
        T = np.array([  [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])

        #Projection 3D --> 2D matrix
        A2 = np.array([ [f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])

        #Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))

    def get_rad(self, theta, phi, gamma):
        return (self.deg_to_rad(theta),
                self.deg_to_rad(phi),
                self.deg_to_rad(gamma))

    def deg_to_rad(self, deg):
        return deg * math.pi /180.0

class load_save_img:

    def __init__(self, image_path):
        self.img_path = image_path

    def load_image(self, shape=None):
        img = cv2.imread(self.img_path)
        if shape != None:
            img = cv2.resize(img, shape)
        return img

    def save_image(self, img_path, img):
        cv2.imwrite(img_path, img)


if __name__ == "__main__":
    # load_save = load_save_img("C:\\Users\\wilat\\OneDrive\\Desktop\\Edit_Image\\grid.jpg")
    # exist_folder = os.listdir('C:\\Users\\wilat\\OneDrive\\Desktop\\images')
    # print(exist_folder)
    # for item in exist_folder:
    
    load_save = load_save_img("C:\\Users\wilat\\Downloads\\cocosynth\\datasets\\Minbury_Dataset\\test\\86.png")
    img = load_save.load_image()
    count = 0
    for item in  range(-60,70,20):  #-50 to 60

        if(item != 0):
            count += 1
            rotateImg = rotate_trans(img).rotate_along_axis(gamma = item)
            # load_save.save_image("C:\\Users\\wilat\\OneDrive\\Desktop\\rotate_img\\degree_phi\\{}.png".format(item), rotateImg)
            cv2.imshow("img{}".format(count),rotateImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
                        