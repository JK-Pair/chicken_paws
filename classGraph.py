import matplotlib.pyplot as plt
import numpy as np

class plotGraph():

    def __init__(self, path_text):
        self.text = path_text
        self.trainList = []
        self.valList = []
        self.switch = 0
        self.number = []
        self.count = 0

    def readText(self, textFile):
        text = self.text + "\\{}.txt".format(textFile)
        with open(text, 'r') as reader:
            # Further file processing goes here
            for line in reader:
                for word in line.split():
                    if word == "loss:":
                        self.switch = 1
                    elif self.switch == 1:
                        self.trainList.append(float(word))
                        self.count += 1
                        self.number.append(self.count)
                        self.switch = 0


        with open(text, 'r') as reader:
            # Further file processing goes here
            for line in reader:
                for word in line.split():
                    if word == "val_loss:":
                        self.switch = 1
                    elif self.switch == 1:
                        self.valList.append(float(word))
                        self.switch = 0

        return self.number, self.trainList, self.valList

    def plot_LossPerEpoch_dot(self, title = "Loss", saveFig = False, nameFig = ""):

        fig = plt.figure()    
        # plotting points as a scatter plot 
        plt.scatter(self.number, self.trainList, color= "blue",  
                    marker= "o", s=30, label = "Train") 

        plt.scatter(self.number, self.valList, color= "red",  
                    marker= "o", s=30, label = "Validation") 
        plt.ylim(0, 10)
        # x-axis label 
        plt.xlabel('Number of epoch') 
        # frequency label 
        plt.ylabel('Log Loss') 
        # plot title 
        plt.title(title) 
        # showing legend 
        plt.legend() 

        # function to show the plot 
        plt.show()
        if saveFig == True:
            fig.savefig(r'C:\Users\wilat\OneDrive\Desktop\graph\Epoch_Dot_{}.png'.format(nameFig), dpi=fig.dpi)
            

    def plot_LossPerEpoch_line(self, title = 'ResNet-101', saveFig = False, nameFig = ""):

        # plot the graph
        fig = plt.figure()
        plt.plot(self.number, self.trainList, 'b', label='Loss of training')
        plt.plot(self.number, self.valList, 'r', label='Loss of validation')
        plt.ylim(0, 10)
        plt.xlabel('Number of epoch')
        plt.ylabel('Log loss')
        plt.title(title)
        plt.legend()
        # show grid
        plt.grid(True) # try changing this values to False

        plt.show()
        if saveFig == True:
            fig.savefig(r'C:\Users\wilat\OneDrive\Desktop\graph\Epoch_Line_{}.png'.format(nameFig), dpi=fig.dpi)

    def plot_mAP_IoU_(self, mAP, label, title = "Evaluating the model", saveFig = False, nameFig = "01"):
        # plot the graph
        numIoU = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        fig = plt.figure()
        color = ['lightcoral', 'cornflowerblue', 'r', 'b', 'grey']
        
        for count in range(len(mAP)):
            if count == (len(mAP)-1):
                plt.plot(numIoU, mAP[count], color[2], label = label[count])
            else:
                plt.plot(numIoU, mAP[count], color[-1], label = label[count])

        plt.ylim(0, 1)
        plt.xlim(50,100)
        plt.xlabel('IoU Threshold (%)')
        plt.ylabel('Mean Average Precision')
        plt.title(title)
        plt.legend()
        # show grid
        plt.grid(True) # try changing this values to False

        plt.show()

        if saveFig == True:
            fig.savefig(r'{}\mAP_IoU_{}.png'.format(self.text, nameFig), dpi=fig.dpi)

if __name__ == "__main__":

    ##Test plot_mAP_IoU function
    plotGH = plotGraph(r"C:\Users\wilat\OneDrive\Desktop\graph")
    mAP_range = [[0.4723431249832114, 0.4508550295357903, 0.39389989964974415, 0.3522218895206849, 0.28139540894577897, 0.14228059065217774, 0.07327742590258518, 0.00895906200322012, 0.0, 0.0],
        [0.6510437915420958, 0.6186628389624612, 0.6077261574442188, 0.5354249316187841, 
		0.4485767114410798, 0.3243749151636092, 0.1939249053136224, 0.09839748631098441, 
		0.01529761946627072, 0.0]]

    # label = ['ExperimentI_50', 'ExperimentII_50', 'ExperimentI_101', 'ExperimentII_101']
    label = ['ExperimentI', 'ExperimentII']
    plotGH.plot_mAP_IoU_(mAP_range,label, saveFig=False, nameFig="50_VS_101")

    ##Test readText function
    # plotGH.readText("ResNet-101-300_0907")
    # plotGH.plot_LossPerEpoch_dot(saveFig = False, title = "Experiment2 (ResNet-101)",nameFig="198_New_100")
    # plotGH.plot_LossPerEpoch_line(saveFig = False, title = "Experiment2 (ResNet-101)",nameFig="15444_New")
