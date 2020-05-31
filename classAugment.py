import cv2
from rotate_trans import *
from tqdm import tqdm
import numpy as np
import os

class augment:

    def __init__(self, img_read, img_write, mask_read, mask_write):
        self.img_read = img_read
        self.img_write = img_write
        self.mask_read = mask_read
        self.mask_write = mask_write
        self.count = 1


    def rotateX(self, min = -40, max = 50, step = 40):

        for item in  tqdm(range(min, max, step), ascii = True, desc="rotateX"): 

            if item != 0:
                rotate_theta_img = rotate_trans(self.Readimg).rotate_along_axis(theta = item)
                rotate_theta_mask = rotate_trans(self.Readmask).rotate_along_axis(theta = item)
                self.Cload_images.save_image(r"{}\{}.png".format(self.img_write, self.count), rotate_theta_img)
                self.Cload_masks.save_image(r"{}\{}.png".format(self.mask_write, self.count), rotate_theta_mask)
            
                self.count += 1

                # self.filp_img(rotate_theta_img, rotate_theta_mask, self.count)
                self.brightness(rotate_theta_img, rotate_theta_mask, self.count, [0.5, 1.5])
                # self.noise_img(rotate_theta_img, rotate_theta_mask, self.count)


    def rotateY(self, min = -40, max = 50, step = 40):

        for item in  tqdm(range(min, max, step), ascii = True, desc="rotateY"): 

            if item != 0:
                rotate_phi_img = rotate_trans(self.Readimg).rotate_along_axis(phi = item)
                rotate_phi_mask = rotate_trans(self.Readmask).rotate_along_axis(phi = item)
                self.Cload_images.save_image(r"{}\{}.png".format(self.img_write, self.count), rotate_phi_img)
                self.Cload_masks.save_image(r"{}\{}.png".format(self.mask_write, self.count), rotate_phi_mask)

                self.count += 1

                # self.filp_img(rotate_phi_img, rotate_phi_mask, self.count)
                self.brightness(rotate_phi_img, rotate_phi_mask, self.count, [0.5, 1.5])
                # self.noise_img(rotate_phi_img, rotate_phi_mask, self.count)


    def rotateZ(self, min = -20, max = 30, step = 20):

        for item in  tqdm(range(min, max, step), ascii = True, desc="rotateZ"): 

            if item != 0:
                rotate_gamma_img = rotate_trans(self.Readimg).rotate_along_axis(gamma = item)
                rotate_gamma_mask = rotate_trans(self.Readmask).rotate_along_axis(gamma = item)
                self.Cload_images.save_image(r"{}\{}.png".format(self.img_write, self.count), rotate_gamma_img)
                self.Cload_masks.save_image(r"{}\{}.png".format(self.mask_write, self.count), rotate_gamma_mask)
                
                self.count += 1

                # self.filp_img(rotate_gamma_img, rotate_gamma_mask, self.count)
                self.brightness(rotate_gamma_img, rotate_gamma_mask, self.count, [0.5, 1.5])
                # self.noise_img(rotate_gamma_img, rotate_gamma_mask, self.count)


    def transX(self, min = -150, max = 160, step = 50):

        for item in  tqdm(range(min, max, step), ascii = True, desc="transX"): 
            
            if item != 0:
                tran_dx_img = rotate_trans(self.Readimg).rotate_along_axis(dx = item)
                tran_dx_mask = rotate_trans(self.Readmask).rotate_along_axis(dx = item)
                self.Cload_images.save_image(r"{}\{}.png".format(self.img_write, self.count), tran_dx_img)
                self.Cload_masks.save_image(r"{}\{}.png".format(self.mask_write, self.count), tran_dx_mask)

                self.count += 1

                # self.filp_img(tran_dx_img, tran_dx_mask, self.count)
                self.brightness(tran_dx_img, tran_dx_mask, self.count, [0.5, 1.5])
                # self.noise_img(tran_dx_img, tran_dx_mask, self.count)


    def transY(self, min = -150, max = 160, step = 50):

        for item in  tqdm(range(min, max, step), ascii = True, desc="transY"): 

            if item != 0:
                tran_dy_img = rotate_trans(self.Readimg).rotate_along_axis(dy = item)
                tran_dy_mask = rotate_trans(self.Readmask).rotate_along_axis(dy = item)
                self.Cload_images.save_image(r"{}\{}.png".format(self.img_write, self.count), tran_dy_img)
                self.Cload_masks.save_image(r"{}\{}.png".format(self.mask_write, self.count), tran_dy_mask)

                self.count += 1

                # self.filp_img(tran_dy_img, tran_dy_mask, self.count)
                self.brightness(tran_dy_img, tran_dy_mask, self.count, [0.5, 1.5])
                # self.noise_img(tran_dy_img, tran_dy_mask, self.count)


    def filp_img(self, readImg, readMask, count):

        flip_img = cv2.flip(readImg, 1)
        flip_mask = cv2.flip(readMask, 1)
        self.Cload_images.save_image(r"{}\{}.png".format(self.img_write, count), flip_img)
        self.Cload_masks.save_image(r"{}\{}.png".format(self.mask_write, count), flip_mask)

        self.count = count + 1

        return self.count

    def adjust_gamma(self, image, gamma=1.0):

        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(image, table)

    def brightness(self, readImg, readMask, count, listbrigh): #listbrigh = [0.5, 1.5]

        for item in listbrigh:
            gamma = item                                
            adj_img = self.adjust_gamma(readImg, gamma=gamma)
            self.Cload_images.save_image(r"{}\{}.png".format(self.img_write, count), adj_img)
            self.Cload_masks.save_image(r"{}\{}.png".format(self.mask_write, count), readMask)
            count += 1

            # self.filp_img(adj_img, readMask, count)
            # count += 1

        self.count = count
        return self.count

    def noise_img(self, readImg, readMask, count):
        kernel = np.ones((3,3),np.float32)/9
        dst = cv2.filter2D(readImg, -1, kernel)
        self.Cload_images.save_image(r"{}\{}.png".format(self.img_write, count), dst)
        self.Cload_masks.save_image(r"{}\{}.png".format(self.mask_write, count), readMask)
        count += 1

        # self.filp_img(dst, readMask, count)
        # count += 1

        self.count = count 
        return self.count

    def main(self):
        
        exist_images = os.listdir(self.img_read)
    
        for image in exist_images:
            if os.path.splitext(image)[1].lower() in ['.png']:
                print("Name Files : ", image)
                self.Cload_images = load_save_img("{}\{}".format(self.img_read, image))
                self.Readimg = self.Cload_images.load_image()
                self.Cload_images.save_image(r"{}\{}.png".format(self.img_write, self.count), self.Readimg) #save original image.

                self.Cload_masks = load_save_img("{}\{}".format(self.mask_read, image))
                self.Readmask = self.Cload_masks.load_image()
                self.Cload_masks.save_image(r"{}\{}.png".format(self.mask_write, self.count), self.Readmask) #save original mask.

                self.count += 1
                # self.filp_img(self.Readimg, self.Readmask, self.count)
                self.brightness(self.Readimg, self.Readmask, self.count, [0.5, 1.5])
                # self.noise_img(self.Readimg, self.Readmask, self.count)
                

                self.rotateX()
                self.rotateY()
                self.rotateZ()

                self.transX()
                # self.transY()

                # self.brightness([0.5, 1.5])
                # self.noise_img()

        print("Created Files")



if __name__ == "__main__":


    Caugment = augment("C:\\Users\\wilat\\OneDrive\\Desktop\\validation\\train\\read\\images", "C:\\Users\\wilat\\OneDrive\\Desktop\\validation\\train\\write\\images", "C:\\Users\\wilat\\OneDrive\\Desktop\\validation\\train\\read\\masks", "C:\\Users\\wilat\\OneDrive\\Desktop\\validation\\train\\write\\masks")     
    Caugment.main()
    # Caugment = augment(r"C:\Users\wilat\Downloads\cocosynth\datasets\Minbury_Dataset\val\images", r"C:\Users\wilat\OneDrive\Desktop\AugmentIV\val\images", r"C:\Users\wilat\OneDrive\Desktop\Cyst_Paws\val\read\masks", r"C:\Users\wilat\OneDrive\Desktop\AugmentIV\val\masks")     
    # Caugment.main()C:\Users\wilat\OneDrive\Desktop\validation\train\read