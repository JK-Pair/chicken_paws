#######################################User Interface Part.#################################
import tkinter
from tkinter import *
from database import *
import pandas as pd
import numpy as np
import matplotlib
from datetime import datetime
matplotlib.use("TKAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import urllib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from main import *
LARGE_FONT =(None, 30)
bottom_FONT = (None, 20)
now = datetime.now()
date_time = now.strftime("%d/%m/%Y")
fig = Figure(figsize=(5,5), dpi=100)
_plotImg = fig.add_subplot(121)
_plot = fig.add_subplot(122) #222

gradeGlobal = [0, 0, 0, 0]
def animate(i):
    gradeList = []

    DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 'FPD_Database_.db')
    dbFPD = databaseFPD(DEFAULT_PATH)
    callRes = dbFPD.callProducts()

    for grade in range(len(callRes)):
        gradeList.append(callRes[grade][-1])

    _plot.clear()
    _plot.title.set_text('Histogram on {}'.format(date_time))
    _plot.set_xlabel('Grade')
    _plot.set_ylabel('The number of paws')
    _plot.hist(gradeList, bins=10, color = 'darkblue')

    _plotImg.clear()
    _plotImg.set_title('The mask prediction part of Mask R-CNN',fontsize= 20)
    img = Image.open(r"C:\Users\wilat\Downloads\Mask_RCNN-master\Mask_RCNN-master\chicken_paws_project\images\0.png")
    _plotImg.imshow(img)

    gradeGlobal[0] = gradeList.count(1)
    gradeGlobal[1] = gradeList.count(2)
    gradeGlobal[2] = gradeList.count(3)
    gradeGlobal[3] = gradeList.count(4)


class SeaofBTCapp(tkinter.Tk):

    def __init__(self, *args, **kwargs):

        tkinter.Tk.__init__(self, *args, **kwargs)
        tkinter.Tk.wm_title(self, "Foot Pad Dermatitis (FPD) Inspection System")
        container = tkinter.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frame = {}
        self.show_frame(showPage)

    def show_frame(self, cont):
        frame = self.frame[cont]
        frame.tkraise()

    def comfirmExit(self):
        response = tkinter.messagebox.askquestion("Comfirm Exit", "Are you sure you want to exit?")
        if response == 'yes':
            tkinter.Tk.quit(self)
            mainDetection(self.result_entry[0], self.result_entry[1], self.result_entry[2], self.result_entry[3], self.result_entry[4], self.result_entry[5], self.result_entry[6], self.result_entry[7])

    def comfirmSetting(self):
        response = tkinter.messagebox.askquestion("Comfirm Setting", "Are you already checked setting?")
        print(self.result_entry)
        if response == 'yes':
            self.show_frame(showPage)

class showPage(tkinter.Frame):
    def __init__(self, parent, controller):
        tkinter.Frame.__init__(self, parent, bg='white')

        label = tkinter.Label(self, text="Foot Pad Dermatitis (FPD) Inspection System \n Show Page", font=LARGE_FONT, bg="white")
        label.pack(pady=30)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

        button1 = tkinter.Button(self, text="Close Programe", font=(None, 18), command=lambda : controller.comfirmExit())
        button1.pack(side = 'right', pady=10)

        button3 = tkinter.Button(self, text="Start Programe", font=(None, 18), command=lambda : controller.startMain())
        button3.pack(side = 'right', pady=10, padx=10)

        self.label1 = tkinter.Label(self, text="Grade A: {}  Grade B: {}  Grade C: {}  Grade D: {}  Sum: {}".format(gradeGlobal[0], gradeGlobal[1], gradeGlobal[2], gradeGlobal[3], sum(gradeGlobal)), font=(None, 26), bg='white')
        self.label1.pack(side = 'right', pady=10, padx=10)

        self.label1.after(1000, self.changeLabelText)
    

    def changeLabelText(self):
        self.label1.config(text="Grade A: {}  Grade B: {}  Grade C: {}  Grade D: {}  Sum: {}".format(gradeGlobal[0], gradeGlobal[1], gradeGlobal[2], gradeGlobal[3], sum(gradeGlobal)), font=(None, 26), bg='white')   
        self.label1.after(1000, self.changeLabelText)

        
app = SeaofBTCapp()
ani = animation.FuncAnimation(fig, animate, interval=1000)
app.mainloop()