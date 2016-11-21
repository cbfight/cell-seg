#paint
import tkinter as tk
import io
from PIL import Image,ImageTk
#frontend
import os
import re
#segmenter
from skimage import io as io2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.morphology import watershed, dilation, erosion, closing, opening, disk
from scipy import ndimage as ndi
import numpy as np
from skimage import measure
from skimage.measure import regionprops
from skimage.filters import gaussian
import matplotlib.patches as mpatches





class ImageCrawler():
    """finds all the images as a first step before curation. doesn't do anything more than provide the initial paths. Only run crawl one time on a given directory tree to initialize it"""
    def __init__(self):
        self.jpegs = re.compile('.*[.]jpg')
        self.all_channels = re.compile('.*c1\+2\+3.*')
        self.already_masked = re.compile('.*mask.*')
        self.dirs = []
        self.target_images = []
        self.strip_pattern = '_c1+2+3.jpg'
        
    
    def crawl(self,path): #on initialize only
        walk = os.walk(path)
        folders = []
        
        for item in walk:
        #a is path b is folders in path c is files in path
            

            folders.append(item[0])
        
        files_i_want = []
        
        for folder in folders:
            contents = os.listdir(folder)
           
                    
            for file in contents:
                if re.search(self.all_channels,file):
                    original = folder + '/' + file
                    mask = folder+'/'+file.replace(self.strip_pattern,'')+'_mask.png'
                    results = folder
                    files_i_want.append((file,original,mask,results)) #result is folder path
        self.target_images = files_i_want
        self.dirs = folders
        
        
    def update_rm(self,path): #updates list of directories by removing whatever you put in here from it.
        for item in self.target_images:
            index = self.target_images.index(item)
            if item[3]==path:
                del self.target_images[index]
    def fresh_filter(self,path):
            if re.search(self.already_masked,path):
                return False
        






class Paint(tk.Tk):
    """Opens an input path in a Paint object for manual segmentation, then outputs the result after the user saves to the output path"""
    def __init__(self,impath,imout):
        
        tk.Tk.__init__(self)
        self.impath = impath
        self.color1 = '#%02x%02x%02x' % (0, 255, 0)
        self.color2 = '#%02x%02x%02x' % (255, 0, 0)
        self.color3 = '#%02x%02x%02x' % (0, 0, 255)
        self.color4 = '#%02x%02x%02x' % (255, 255, 255)
        self.color5 = '#%02x%02x%02x' % (0, 0, 0)
        
        self.size_s = 5
        self.size_m = 10
        self.size_l = 15
        self.size_xl = 25
        self.size_xxl = 125
        
        self.color = self.color1
        self.size = 10
        self.strsize = 'M'
        self.mode = 'Including'
        self.title(' | '.join([self.impath,self.mode,self.strsize]))
        
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        filemenu1 = tk.Menu(menubar)
        filemenu2 = tk.Menu(menubar)
        filemenu3 = tk.Menu(menubar)
        filemenu1.add_command(label="Small | 7",command=lambda:self.setSize(self.size_s))
        filemenu1.add_command(label="Medium | 8",command=lambda:self.setSize(self.size_m))
        filemenu1.add_command(label="Large | 9",command=lambda:self.setSize(self.size_l))
        filemenu1.add_command(label="X-Large | 0",command=lambda:self.setSize(self.size_xl))
        filemenu1.add_command(label="XXL-Large",command=lambda:self.setSize(self.size_xxl))
        filemenu2.add_command(label="Green | 1", command=lambda:self.setColor(self.color1))
        filemenu2.add_command(label="Red | 2",command=lambda:self.setColor(self.color2))
        
        filemenu3.add_command(label="Reset | r",command=self.reset)
        filemenu3.add_command(label="Remove Image | backspace",command=self.clearimage)
        filemenu3.add_command(label="Save | s",command=self.savefile)
        filemenu3.add_command(label="Quit | q",command=self.quit)
        
        menubar.add_cascade(label="Brush Size",menu=filemenu1)
        menubar.add_cascade(label="Brush Color",menu=filemenu2)
        menubar.add_cascade(label="Auxillary",menu=filemenu3)
        
        
        
        self.imout = imout
        self.impath = impath
        self.openedimage = Image.open(self.impath)
        self.imagex = self.openedimage.width
        self.imagey = self.openedimage.height
        self.resizedimage = self.openedimage.resize((694,520),resample=Image.LANCZOS)
        self.image= ImageTk.PhotoImage(self.resizedimage)
        
        
        self.lastx,self.lasty=None,None
        
        
        self.canvas = tk.Canvas(master=self, width=694, height=520,highlightthickness=-1,background='gray')
        
        self.canvas.bind("<Button-1>", self.xy)
        self.canvas.bind("<B1-Motion>", self.addLine)
        self.bind("s", lambda x: self.savefile())
        self.bind("r", lambda x: self.reset())
        self.bind("1",lambda x: self.setColor(self.color1))
        self.bind("2",lambda x: self.setColor(self.color2))
        self.bind("<BackSpace>",lambda x: self.clearimage())
        self.bind("q",lambda x: self.quit())
        self.bind("7",lambda x: self.setSize(self.size_s))
        self.bind("8",lambda x: self.setSize(self.size_m))
        self.bind("9",lambda x: self.setSize(self.size_l))
        self.bind("0",lambda x: self.setSize(self.size_xl))
        
        self.opened_image = tk.PhotoImage(master = self.canvas)
        self.canvas.create_image(0,0, image=self.image,anchor=tk.NW,tags='image')
        self.canvas.pack()
        
    def setSize(self,newsize):
        self.size = newsize
        if newsize == self.size_s:
            self.strsize = 'S'
        if newsize == self.size_m:
            self.strsize = 'M'
        if newsize == self.size_l:
            self.strsize = 'L'
        if newsize == self.size_xl:
            self.strsize = 'XL'
        if newsize == self.size_xxl:
            self.strsize = 'XXL'
        self.title(' | '.join([self.impath,self.mode,self.strsize]))
    def test(self):
        pass
        
        
    def quit(self):
        self.destroy()
    def savefile(self):
        ps = self.canvas.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img = img.resize((1388,1040),resample=Image.LANCZOS)
        #img = img.convert(mode="L")
        img.save(self.imout)
        
    
    def setColor(self,newcolor):
        self.color = newcolor
        if newcolor == self.color1:
            self.mode='Including'
        if newcolor == self.color2:
            self.mode='Excluding'
        self.title(' | '.join([self.impath,self.mode,self.strsize]))
    def addLine(self,event):
        self.canvas.create_line((self.lastx, self.lasty, event.x, event.y), fill=self.color, width=self.size, tags='drawn_line',smooth=1,splinesteps=50,joinstyle=tk.ROUND)
        self.lastx, self.lasty = event.x, event.y  
    def xy(self,event):
        self.lastx,self.lasty=event.x,event.y
    def reset(self):
        self.canvas.delete('drawn_line')
    def clearimage(self):
        self.canvas.delete('image')





class Segmenter():
    """Takes mask path, original file path, result path, and filename arguments and segments out epithelium and dermal layers of tissue with positive cell counts in each stratum. Epidermis is masked in green and dermis is masked in red."""
    def __init__(self,filename,orig_path,mask_path,result_path):
        self.path_mask = mask_path
        self.path_orig = orig_path
        self.path_result = result_path
        self.strip_pattern = '_c1+2+3.jpg'
        self.filename = filename.replace(self.strip_pattern,'')
        self.genpath = []
    def segment(self):
        self.im1 = io2.imread(self.path_mask)
        self.im2 = gaussian(io2.imread(self.path_orig),sigma=1,multichannel=True)
        im2r = np.copy(self.im2[:,:,0])
        im2g = np.copy(self.im2[:,:,1])
        im2b = np.copy(self.im2[:,:,2])
        self.im2gray = rgb2gray(self.im2)

        rt = 0.3*im2r.max()
        gt = 0.3*im2g.max() #0.3 before
        bt = 0.3*im2b.max()

        rm = im2r > rt
        gm = im2g > gt
        bm = im2b > bt

        rm_bm = (rm == 1) & (bm == 1)
        gm_bm = (gm == 1) & (bm == 1)
        rm_gm_bm = (gm == 1) &(rm == 1) & (bm == 1)

        rm_bm = np.ma.masked_where(rm_bm == 0,rm_bm)
        rm_bm = closing(rm_bm,disk(3))
        gm_bm = np.ma.masked_where(gm_bm == 0,gm_bm)
        gm_bm = closing(gm_bm,disk(3))
        rm_gm_bm = np.ma.masked_where(rm_gm_bm == 0,rm_gm_bm)
        rm_gm_bm = closing(rm_gm_bm,disk(3))
        
        self.epi_seeds = np.copy((self.im1[:,:,0] == 0) & (self.im1[:,:,1] > 250) & (self.im1[:,:,2] == 0))
        
        #self.epi_seeds = (self.im1[:,:,0] == 123) & (self.im1[:,:,1] == 123) & (self.im1[:,:,2] == 0)
        self.epi_seeds = dilation(self.epi_seeds,disk(15))
        self.epi_seeds_mask = np.ma.masked_where(self.epi_seeds==0,self.epi_seeds)
        self.bad_seeds = np.copy((self.im1[:,:,0] > 250) & (self.im1[:,:,1] == 0) & (self.im1[:,:,2] == 0))
        #self.bad_seeds = (self.im1[:,:,0] == 0) & (self.im1[:,:,1] == 0) & (self.im1[:,:,2] == 0)
        self.bad_seeds = dilation(self.bad_seeds,disk(15))
        self.bad_seeds_mask = np.ma.masked_where(self.bad_seeds==0,self.bad_seeds)
    
        self.epi_brdu = np.ma.masked_where(self.epi_seeds==0,gm_bm)
        self.epi_vim = np.ma.masked_where(self.epi_seeds==0,rm_bm)
        self.derm_mask = np.copy((self.bad_seeds==1) + (self.epi_seeds==1))
        self.derm_vim = np.ma.masked_where((self.derm_mask==1),rm_bm)
        self.derm_brdu = np.ma.masked_where((self.derm_mask==1),gm_bm)
 
        #need to add a remove small objects step
        self.epi_count = measure.label(dilation(~np.ma.getmask(self.epi_brdu),disk(2)))
        self.ec = len(np.unique(self.epi_count))-1
        self.derm_count = measure.label(dilation(~np.ma.getmask(self.derm_vim),disk(2)))
        self.dc = len(np.unique(self.derm_count))-1
        
    def plot1(self):
        fig = plt.figure(figsize=(10,10))
        plt.subplot(1,2,1)
        plt.imshow(self.im2)
        plt.hold(True)
        plt.imshow(self.epi_seeds_mask,alpha=0.2,cmap='cool')
        plt.title('Inclusion Area: Epithelium')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1,2,2)
        plt.imshow(self.im2)
        plt.hold(True)
        plt.imshow(self.bad_seeds_mask,alpha=0.2,cmap='cool')
        plt.title('Exclusion Area: Follicle')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=0.5)
        #plt.show()
        figsavepath = self.path_result+'/'+self.filename+'_plot1.png'
        fig.savefig(figsavepath,dpi=150,bbox_inches='tight')
        self.genpath.append(figsavepath)
        plt.clf()
    def plot2(self):
        fig = plt.figure(figsize=(10,10))
        plt.subplot(221)
        plt.imshow(self.im2)
        plt.title('Original')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(222)
        plt.imshow(self.im2)
        plt.hold(True)
        plt.imshow(self.epi_brdu,cmap='cool')
        plt.title('BrdU+DAPI Double Positive (Epithelium)')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(223)
        plt.imshow(self.im2)
        plt.hold(True)
        plt.imshow(self.derm_vim,cmap='cool')
        plt.title('Vim+DAPI Double Positive (Dermis exl. Follicles)')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=0.5)
        #plt.show()
        figsavepath = self.path_result+'/'+self.filename+'_plot2.png'
        fig.savefig(figsavepath,dpi=150,bbox_inches='tight')
        self.genpath.append(figsavepath)
        '''
        plt.subplot(224)
        plt.imshow(im2)
        plt.hold(True)
        plt.imshow(self.derm_vim,cmap='cool')
        plt.title('Vim+BrdU+DAPI Triple Positive (Dermis exl. Follicles)') #make this actually show triple positive
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=0.5)
        plt.show()
        '''
        plt.clf()
    def plot3(self):
        fig = plt.figure(figsize=(10,10))
        plt.subplot(1,2,1)
        plt.title('Epidermal Counts: '+str(self.ec))
        plt.imshow(self.epi_count,cmap='viridis')
        plt.subplot(1,2,2)
        plt.title('Dermal Counts: '+str(self.dc))
        plt.imshow(self.derm_count,cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=0.5)
        #plt.show()
        figsavepath = self.path_result+'/'+self.filename+'_plot3.png'
        fig.savefig(figsavepath,dpi=150,bbox_inches='tight')
        self.genpath.append(figsavepath)
        plt.clf()
    def plot4(self):
        derm_bbox = self.derm_count
        epi_bbox = self.epi_count
        fig,ax = plt.subplots(figsize=(10,10))
        ax.imshow(self.im2)

        for region in regionprops(derm_bbox):

            # skip small images
            #if region.area < 5:
                #continue

            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='white', linewidth=1)
            ax.add_patch(rect)
        for region in regionprops(epi_bbox):
            minr,minc,maxr,maxc = region.bbox
            rect = mpatches.Rectangle((minc,minr),maxc-minc,maxr-minr,fill=False,edgecolor='magenta',linewidth=1)
            ax.add_patch(rect)
        plt.title('Bounding Boxes')
        plt.xticks([])
        plt.yticks([])
        #plt.show()
        figsavepath = self.path_result+'/'+self.filename+'_plot4.png'
        fig.savefig(figsavepath,dpi=150,bbox_inches='tight')
        self.genpath.append(figsavepath)
        plt.clf()

    def result_summary(self):
        summary_info = (self.path_orig,self.path_mask,self.path_result,self.ec,self.dc)
        return summary_info
#(Inclusion Area: Epi, Inclusion Area: Derm), (Original, Double1, Double2, Triple), (Epi Counts, Derm Counts)

#write out the results from segment to a report... 
#import openpyxl
import csv

class SegReport():
    def __init__(self,segment_summary,output):
        self.output = output #path for report
        self.segment_summary = segment_summary
    
    def report(self):
        #walk the same directories using ImageCrawl(should be in the segqueue?) and get the data.
        #need to make segment class generate a pickle of a results dictionary for each image.
        #digest_pattern = re.compile('^[/].*[/].*[.].*$')
        with open(self.output,'w') as fout:
            report_file = csv.writer(fout)
            report_file.writerow(['original path','mask path','result path','metric 1','metric 2'])
            for entry in self.segment_summary:
                report_file.writerow([entry[0],entry[1],entry[2],entry[3],entry[4]])
                
            
    
