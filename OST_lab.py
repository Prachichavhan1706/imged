from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import cv2
import numpy as np
import PIL.Image
import PIL.ImageTk
import matplotlib.pyplot as plt
from skimage import exposure

class Editor(Tk):

    def __init__(self):
        Tk.__init__(self)
        self.title('Image Editor')
        self.resizable(0, 0)

        self.imageLabel = Label(self)
        self.imageLabel.grid(row=0, column=0)

        self.openImage('cameraman.jpg')

        self.butFrame = Frame(self)
        self.butFrame.grid(row=0, column=1)

        self.buttonQuit = Button(self.butFrame, text='Quit', command=self.quit).pack(fill=BOTH)
        self.buttonOpen = Button(self.butFrame, text='Open image', command=self.openImage).pack(fill=BOTH)
        self.buttonRevert = Button(self.butFrame, text='Revert to original', command=self.revertImage).pack(fill=BOTH)
        self.buttonSave = Button(self.butFrame, text='Save image', command=self.saveImage).pack(fill=BOTH)

        self.flipLabel = Label(self.butFrame)
        self.flipLabel.pack()
        self.flipText = Label(self.flipLabel, text='Flip:').pack(fill=BOTH, side='left')
        self.buttonFlipHor = Button(self.flipLabel, text='Horiz', command=lambda: self.flip('HORIZONTAL')).pack(fill=BOTH, side='left')
        self.buttonFlipVer = Button(self.flipLabel, text='Vert', command=lambda: self.flip('VERTICAL')).pack(fill=BOTH, side='right')

        self.brightLabel = Label(self.butFrame)
        self.brightLabel.pack()
        self.brightText = Label(self.brightLabel, text='Brightness:').pack(fill=BOTH, side='left')
        self.buttonBrightUp = Button(self.brightLabel, text='Up', command=lambda: self.brightness('UP')).pack(fill=BOTH, side='left')
        self.buttonBrightDown = Button(self.brightLabel, text='Down', command=lambda: self.brightness('DOWN')).pack(fill=BOTH, side='right')

        self.contrastLabel = Label(self.butFrame)
        self.contrastLabel.pack()
        self.contrastText = Label(self.contrastLabel, text='Contrast:').pack(fill=BOTH, side='left')
        self.buttonContrastUp = Button(self.contrastLabel, text='Up', command=lambda: self.contrast('UP')).pack(fill=BOTH, side='left')
        self.buttonContrastDown = Button(self.contrastLabel, text='Down', command=lambda: self.contrast('DOWN')).pack(fill=BOTH, side='right')

        self.buttonGray = Button(self.butFrame, text='Grayscale', command=self.grayscale).pack(fill=BOTH)
        self.buttonQuant = Button(self.butFrame, text='Quantization', command=self.quantization).pack(fill=BOTH)
        self.buttonNeg = Button(self.butFrame, text='Negative', command=self.negative).pack(fill=BOTH)
        self.buttonHist = Button(self.butFrame, text='Histogram', command=self.histogram).pack(fill=BOTH)
        self.buttonEqualize = Button(self.butFrame, text='Equalize Histo', command=self.equalizeHistograms).pack(fill=BOTH)
        self.buttonMatch = Button(self.butFrame, text='Match Histos', command=self.matchHistograms).pack(fill=BOTH)

        self.zoomLabel = Label(self.butFrame)
        self.zoomLabel.pack()
        self.zoomText = Label(self.zoomLabel, text='Zoom:').pack(fill=BOTH, side='left')
        self.buttonZoomIn = Button(self.zoomLabel, text='In ', command=lambda: self.zoom('IN')).pack(fill=BOTH, side='left')
        self.buttonZoomOut = Button(self.zoomLabel, text='Out', command=lambda: self.zoom('OUT')).pack(fill=BOTH, side='right')

        self.rotateLabel = Label(self.butFrame)
        self.rotateLabel.pack()
        self.rotateText = Label(self.rotateLabel, text='Rotate:').pack(fill=BOTH, side='left')
        self.buttonRotLeft = Button(self.rotateLabel, text='Left', command=lambda: self.rotate('LEFT')).pack(fill=BOTH, side='left')
        self.buttonRotRight = Button(self.rotateLabel, text='Right', command=lambda: self.rotate('RIGHT')).pack(fill=BOTH, side='right')

    def updateLabel(self, img):
        tempImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        tempImg = PIL.Image.fromarray(tempImg)
        tempImg = PIL.ImageTk.PhotoImage(image=tempImg)
        self.imageLabel.configure(image=tempImg)
        self.imageLabel.image = tempImg

    def openImage(self, filename=None):
        if filename is None:
            try:
                filename = filedialog.askopenfilename(initialdir='~/Pictures', title='Open image')
            except (OSError, FileNotFoundError):
                messagebox.showerror('Error', 'Unable to find or open file <filename>')
            except Exception as error:
                messagebox.showerror('Error', f'An error occurred: {error}')

        if filename:
            self.image = cv2.imread(filename)
            self.origImage = self.image.copy()
            self.updateLabel(self.image)

    def revertImage(self):
        self.image = self.origImage.copy()
        self.updateLabel(self.image)

    def saveImage(self):
        try:
            filename = filedialog.asksaveasfilename(initialdir='~/Pictures', title='Save image')
        except Exception as error:
            messagebox.showerror('Error', f'An error occurred: {error}')

        if filename:
            cv2.imwrite(filename, self.image)

    def flip(self, option):
        h, w, _ = self.image.shape
        temp = np.zeros((h, w, 3), np.uint8)
        if option == 'HORIZONTAL':
            for i in range(w):
                temp[:, i, :] = self.image[:, w-i-1, :]
        elif option == 'VERTICAL':
            for j in range(h):
                temp[j, :, :] = self.image[h-j-1, :, :]
        self.image = temp
        self.updateLabel(self.image)

    def grayscale(self):
        b = self.image[:, :, 0]
        g = self.image[:, :, 1]
        r = self.image[:, :, 2]
        gray = 0.114*b + 0.587*g + 0.299*r
        self.image[:, :, 0] = self.image[:, :, 1] = self.image[:, :, 2] = gray
        self.updateLabel(self.image)

    def quantization(self):
        pass

    def brightness(self, option):
        bias = 20 if option == 'UP' else -20
        self.image = cv2.addWeighted(self.image, 1, np.zeros(self.image.shape, self.image.dtype), 0, bias)
        self.updateLabel(self.image)

    def contrast(self, option):
        gain = 1.25 if option == 'UP' else 0.8
        self.image = cv2.addWeighted(self.image, gain, np.zeros(self.image.shape, self.image.dtype), 0, 0)
        self.updateLabel(self.image)

    def negative(self):
        self.image[:, :, :] = 255 - self.image[:, :, :]
        self.updateLabel(self.image)

    def histogram(self):
        if len(self.image.shape) == 3:  # Color image
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
                plt.xlim([0, 256])
        else:  # Grayscale image
            hist = cv2.calcHist([self.image], [0], None, [256], [0, 256])
            plt.plot(hist, color='k')
            plt.xlim([0, 256])

        plt.title('Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.show()

    def equalizeHistograms(self):
    #check if grayscale, if yes directly apply histogram equalization
        if len(self.image.shape) == 2:
            equalized_image = cv2.equalizeHist(self.image)
        else:
            # For color images, equalize each channel separately
            ycrcb_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCrCb)
            ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
            # Convert back to BGR color space
            equalized_image = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        
        # Update the current image with the equalized image
        self.image = equalized_image
        self.updateLabel(self.image)

    def matchHistograms(self):
        #reference image to match histograms with
        reference_filename = filedialog.askopenfilename(title='Reference image')
        
        if reference_filename:
            reference_image = cv2.imread(reference_filename)

            # Check if the reference image has the same number of channels as the current image
            if reference_image.shape[-1] != self.image.shape[-1]:
                messagebox.showerror("Error", "The reference image must have the same number of channels as the current image.")
                return

            if len(self.image.shape) == 3:  # Color image
                matched = exposure.match_histograms(self.image, reference_image, channel_axis=-1)
            else:  # Grayscale image
                matched = exposure.match_histograms(self.image, reference_image)

            # Convert matched image back to uint8
            matched = (matched * 255).astype(np.uint8)

            # Update the current image with the matched image
            self.image = matched
            # Update the displayed image
            self.updateLabel(self.image)

    def zoom(self, option):
        h, w, _ = self.image.shape
        if option == 'IN':
            temp1 = np.zeros((h, 2*w, 3), np.uint8)
            for i in range(w):
                temp1[:, 2*i, :] = self.image[:, i, :]
                temp1[:, 2*i+1, :] = self.image[:, i, :]
            temp = np.zeros((2*h, 2*w, 3), np.uint8)
            for j in range(h):
                temp[2*j, :, :] = temp1[j, :, :]
                temp[2*j+1, :, :] = temp1[j, :, :]
        elif option == 'OUT' and h > 10 and w > 10:
            temp1 = np.zeros((h, int(w/2), 3), np.uint8)
            for i in range(int(w/2)):
                temp1[:, i, :] = (self.image[:, 2*i, :] + self.image[:, 2*i+1, :]) / 2
            temp = np.zeros((int(h/2), int(w/2), 3), np.uint8)
            for j in range(int(h/2)):
                temp[j, :, :] = (temp1[2*j, :, :] + temp1[2*j+1, :, :]) / 2

        self.image = temp
        self.updateLabel(self.image)

    def rotate(self, direction):
        h, w, _ = self.image.shape
        temp = np.zeros((w, h, 3), np.uint8)
        if direction == 'RIGHT':
            for i in range(w):
                temp[i, :, :] = self.image[h-1::-1, i, :]
        elif direction == 'LEFT':
            for i in range(w):
                temp[i, :, :] = self.image[:, w-i-1, :]
        self.image = temp
        self.updateLabel(self.image)

editor = Editor()
editor.mainloop()
