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

color_dict = {'red':(255, 0, 0),'green':(0, 255, 0),'blue':(0, 0, 255),'magenta':(255, 255, 0),'cyan':(0, 255, 255)}


class ImageCrawler():
    """finds all the images as a first step before curation. doesn't do anything more than provide the initial paths. Only run crawl one time on a given directory tree to initialize it"""
    def __init__(self):
        self.jpegs = re.compile('.*[.]jpg')
        self.all_channels = re.compile('.*c1\+2\+3.*')
        self.already_masked = re.compile('.*mask.*')
        self.dirs = []
        self.target_images = []
        self.strip_pattern = '_c1+2+3.jpg'
        
    
    def crawl(self, path):  # on initialize only
        walk = os.walk(path)
        folders = []
        
        for item in walk:
        # a is path b is folders in path c is files in path
            

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

    def update_rm(self, path):  # updates list of directories by removing whatever you put in here from it.
        for item in self.target_images:
            index = self.target_images.index(item)
            if item[3] == path:
                del self.target_images[index]

    def fresh_filter(self, path):
            if re.search(self.already_masked, path):
                return False
        






class Paint(tk.Tk):
    """Opens an input path in a Paint object for manual segmentation, then outputs the result after the user saves to the output path"""
    def __init__(self, impath, imout):
        
        tk.Tk.__init__(self)
        self.impath = impath
        self.color1 = '#%02x%02x%02x' % color_dict['green']
        self.color2 = '#%02x%02x%02x' % color_dict['red']
        self.color3 = '#%02x%02x%02x' % color_dict['blue']
        self.color4 = '#%02x%02x%02x' % color_dict['magenta']
        self.color5 = '#%02x%02x%02x' % color_dict['cyan']
        
        self.size_s= 5
        self.size_m = 10
        self.size_l = 15
        self.size_xl = 25
        self.size_xxl = 125
        
        self.color = self.color1
        self.size = 10
        self.strsize = 'M'
        self.mode = 'Including'
        self.title(' | '.join([self.impath, self.mode, self.strsize]))
        
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        filemenu1 = tk.Menu(menubar)
        filemenu2 = tk.Menu(menubar)
        filemenu3 = tk.Menu(menubar)
        filemenu1.add_command(label="Small | 7", command=lambda: self.setSize(self.size_s))
        filemenu1.add_command(label="Medium | 8", command=lambda: self.setSize(self.size_m))
        filemenu1.add_command(label="Large | 9", command=lambda: self.setSize(self.size_l))
        filemenu1.add_command(label="X-Large | 0", command=lambda: self.setSize(self.size_xl))
        filemenu1.add_command(label="XXL-Large", command=lambda: self.setSize(self.size_xxl))

        filemenu2.add_command(label="Green | 1", command=lambda: self.setColor(self.color1))
        filemenu2.add_command(label="Red | 2", command=lambda: self.setColor(self.color2))
        filemenu2.add_command(label="Blue", command=lambda: self.setColor(self.color3))
        filemenu2.add_command(label="White", command=lambda: self.setColor(self.color4))
        filemenu2.add_command(label="Black", command=lambda: self.setColor(self.color5))
        
        filemenu3.add_command(label="Reset | r", command=self.reset)
        filemenu3.add_command(label="Remove Image | backspace", command=self.clearimage)
        filemenu3.add_command(label="Save | s", command=self.savefile)
        filemenu3.add_command(label="Quit | q", command=self.quit_paint)
        
        menubar.add_cascade(label="Brush Size", menu=filemenu1)
        menubar.add_cascade(label="Brush Color", menu=filemenu2)
        menubar.add_cascade(label="Auxillary", menu=filemenu3)
        
        
        
        self.imout = imout
        self.impath = impath
        self.openedimage = Image.open(self.impath)
        self.imagex = self.openedimage.width
        self.imagey = self.openedimage.height
        self.resizedimage = self.openedimage.resize((694, 520), resample=Image.LANCZOS)
        self.image = ImageTk.PhotoImage(self.resizedimage)
        
        
        self.lastx, self.lasty=None,None

        self.canvas = tk.Canvas(master=self, width=694, height=520, highlightthickness=-1, background='gray')
        
        self.canvas.bind("<Button-1>", self.xy)
        self.canvas.bind("<B1-Motion>", self.addLine)
        self.bind("s", lambda x: self.savefile())
        self.bind("r", lambda x: self.reset())
        self.bind("1", lambda x: self.setColor(self.color1))
        self.bind("2", lambda x: self.setColor(self.color2))
        self.bind("<BackSpace>", lambda x: self.clearimage())
        self.bind("q", lambda x: self.quit_paint())
        self.bind("7", lambda x: self.setSize(self.size_s))
        self.bind("8", lambda x: self.setSize(self.size_m))
        self.bind("9", lambda x: self.setSize(self.size_l))
        self.bind("0", lambda x: self.setSize(self.size_xl))
        
        self.opened_image = tk.PhotoImage(master=self.canvas)
        self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW, tags='image')
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


    def quit_paint(self):
        self.destroy()
    def savefile(self):
        ps = self.canvas.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img = img.resize((1388, 1040), resample=Image.LANCZOS)
        #img = img.convert(mode="L")
        img.save(self.imout)
        
    
    def setColor(self, newcolor):
        self.color = newcolor
        if newcolor == self.color1:
            self.mode ='Including'
        if newcolor == self.color2:
            self.mode ='Excluding'
        self.title(' | '.join([self.impath, self.mode, self.strsize]))
    def addLine(self, event):
        self.canvas.create_line((self.lastx, self.lasty, event.x, event.y), fill=self.color, width=self.size, tags='drawn_line',smooth=1,splinesteps=50,joinstyle=tk.ROUND)
        self.lastx, self.lasty = event.x, event.y  
    def xy(self,event):
        self.lastx, self.lasty = event.x, event.y
    def reset(self):
        self.canvas.delete('drawn_line')
    def clearimage(self):
        self.canvas.delete('image')





class Segmenter():
    """Takes mask path, original file path, result path, and filename arguments and segments out epithelium and dermal layers of tissue with positive cell counts in each stratum. Epidermis is masked in green and dermis is masked in red."""
    def __init__(self, filename, orig_path, mask_path, result_path):
        self.path_mask = mask_path
        self.path_orig = orig_path
        self.path_result = result_path
        self.im1 = io2.imread(self.path_mask)  # read in the mask file
        self.im2 = io2.imread(self.path_orig)  # read in the original image
        self.strip_pattern = '_c1+2+3.jpg'  # images taken on the zeiss have this suffix that needs to be adjusted for each microscope
        self.filename = filename.replace(self.strip_pattern, '')
        self.genpath = []  # a record of all the file paths produced by this object
        self.result_dict = None

    def correction(self, disk_size=15, T=0.25, gaussian_sigma=3):
        """apply preprocessing steps here to the parent image; currently performs H-dome bg subtr and a gaussian filter"""
        im = self.im2
        uncorr_red = np.copy(im[:, :, 0])
        uncorr_green = np.copy(im[:, :, 1])
        uncorr_blue = np.copy(im[:, :, 2])
        uncorr_cols = (uncorr_red, uncorr_green, uncorr_blue)
        corrected_cols = []

        for uncorrected in uncorr_cols:
            bg = opening(uncorrected, disk(disk_size))
            corrected = ndi.gaussian_filter(uncorrected - bg, gaussian_sigma)
            corrected = corrected > T * corrected.max()
            corrected_cols.append(corrected)
        self.im2 = np.dstack(corrected_cols)





    def get_regions(self):
        """works on the mask file only to determine highlighted regions"""
        # gets the regions in the mask image and prepares it
        self.zone1 = np.copy((self.im1[:, :, 0] < 10) & (self.im1[:, :, 1] > 250) & (self.im1[:, :, 2] < 10))  # g
        self.zone2 = np.copy((self.im1[:, :, 0] > 250) & (self.im1[:, :, 1] < 10) & (self.im1[:, :, 2] < 10))  # r
        self.zone3 = np.copy((self.im1[:, :, 0] < 10) & (self.im1[:, :, 1] < 10) & (self.im1[:, :, 2] > 250))  # b
        self.zone4 = np.copy((self.im1[:, :, 0] > 250) & (self.im1[:, :, 1] > 250) & (self.im1[:, :, 2] < 10))  # r+g (mag)
        self.zone5 = np.copy((self.im1[:, :, 0] < 10) & (self.im1[:, :, 1] > 250) & (self.im1[:, :, 2] > 250))  # g+b (cy)
        self.zone6 = np.copy((self.im1[:, :, 0] > -1) & (self.im1[:, :, 1] > -1) & (self.im1[:, :, 2] > -1))  # all colors
        # flesh out the masked zones
        self.zone1 = dilation(self.zone1, disk(15))
        self.zone2 = dilation(self.zone2, disk(15))
        self.zone3 = dilation(self.zone3, disk(15))
        self.zone4 = dilation(self.zone4, disk(15))
        self.zone5 = dilation(self.zone5, disk(15))
        self.zone6 = dilation(self.zone6, disk(15))

        # make them into actual masks for use in overlays

        self.zone1_mask = np.ma.masked_less(self.zone1, 1)
        self.zone2_mask = np.ma.masked_less(self.zone2, 1)
        self.zone3_mask = np.ma.masked_less(self.zone3, 1)
        self.zone4_mask = np.ma.masked_less(self.zone4, 1)
        self.zone5_mask = np.ma.masked_less(self.zone5, 1)
        self.zone6_mask = np.ma.masked_less(self.zone6, 1)

        self.zone_list = [
            (self.zone1_mask, 'zone 1'),
            (self.zone2_mask, 'zone 2'),
            (self.zone3_mask, 'zone 3'),
            (self.zone4_mask, 'zone 4'),
            (self.zone5_mask, 'zone 5'),
            (self.zone6_mask, 'zone 6')]

    def color_rules(self):
        """works on the original (or corrected original) image to segregate color features"""
        im2r = np.copy(self.im2[:, :, 0])  # copy the original layer because weird things happened before when I didn't
        im2g = np.copy(self.im2[:, :, 1])  # copy the original layer because weird things happened before when I didn't
        im2b = np.copy(self.im2[:, :, 2])  # copy the original layer because weird things happened before when I didn't

        color_t = {'red': 0.3 * im2r.max(), 'green': 0.3 * im2g.max(), 'blue': 0.3 * im2b.max()}

        r = im2r > color_t['red']  # get the bool mask of the high spots of red channel
        g = im2g > color_t['green']  # get the bool mask of the high spots of green channel
        b = im2b > color_t['blue']  # get the bool mask of the high spots of blue channel

        rb = (r == 1) & (b == 1)
        gb = (g == 1) & (b == 1)
        rgb = (r == 1) & (g == 1) & (b == 1)

        self.color_rule_list = [(b,'b'), (rb, 'r+b'), (gb, 'g+b'), (rgb, 'r+g+b')]

    def segment(self):
        # for a zone, process it for all color combinations, and do all the zones like so
        templist = []
        b_counts = {}
        gb_counts = {}
        rb_counts = {}
        rgb_counts = {}
        color_counts = {'b': b_counts, 'r+b': rb_counts, 'g+b': gb_counts, 'r+g+b': rgb_counts}
        for zone in self.zone_list:
            for rule in self.color_rule_list:
                #print('_'.join([zone[1],rule[1]]))  # diagnostic
                color_layer = closing(rule[0], disk(2))
                mask = np.ma.getmask(zone[0])
                mask_mult_color = np.multiply(~mask, color_layer)
                label, counts = ndi.label(mask_mult_color)
                templist.append((zone, rule[1], counts, label))  # zone, color rule, # counted, array
                if rule[1] == 'b':
                    b_counts[zone[1]] = counts
                if rule[1] == 'r+b':
                    rb_counts[zone[1]] = counts
                if rule[1] == 'g+b':
                    gb_counts[zone[1]] = counts
                if rule[1] == 'r+g+b':
                    rgb_counts[zone[1]] = counts

        for item in templist:
            fig, ax = plt.subplots(figsize=(10,10))
            fname = '_'.join([str(item[0][1]), str(item[1]), str(item[2])])

            plt.imshow(self.im2)  # the original image as background
            plt.hold(True)
            plt.imshow(item[0][0], alpha=0.3, cmap='cool')  # the region counted as overlay
            plt.title(fname)
            plt.xticks([])
            plt.yticks([])

            for region in regionprops(item[3]):
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='white', linewidth=1)
                ax.add_patch(rect)

            plt.tight_layout()


            figsavepath = self.path_result + '/' + self.filename + '_' + fname + ' diagnostic.png'
            fig.savefig(figsavepath, dpi=150, bbox_inches='tight')  # disabling save for testing
            plt.clf()
            self.genpath.append(figsavepath)
        #print(color_counts)  # diagnostic
        self.result_dict = color_counts # a dictionary contain break down of results by color then zone

    def plot1(self):
        fig = plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(self.im2)
        plt.hold(True)
        plt.imshow(self.zone1_mask, alpha=0.75, cmap='cool')
        plt.title('Inclusion Area: Epithelium')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 2, 2)
        plt.imshow(self.im2)
        plt.hold(True)
        plt.imshow(self.zone2_mask, alpha=0.2, cmap='cool')
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
        plt.imshow(self.zone1_green_blue,cmap='cool')
        plt.title('BrdU+DAPI Double Positive (Epithelium)')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(223)
        plt.imshow(self.im2)
        plt.hold(True)
        plt.imshow(self.zone2_red_blue,cmap='cool')
        plt.title('Vim+DAPI Double Positive (Dermis exl. Follicles)')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(224)
        plt.imshow(self.im2)
        plt.hold(True)
        plt.imshow(self.zone1_count2, cmap='cool')
        plt.title('Only Epi DAPI')  # make this actually show triple positive
        plt.xticks([])
        plt.yticks([])

        #plt.show()
        plt.tight_layout(pad=0.5)
        #plt.show()
        figsavepath = self.path_result+'/'+self.filename+'_plot2.png'
        fig.savefig(figsavepath,dpi=150,bbox_inches='tight')
        self.genpath.append(figsavepath)



        plt.clf()
    def plot3(self):
        fig = plt.figure(figsize=(10, 10))
        plt.subplot(1,2,1)
        plt.title('Epidermal Counts: '+str(self.z1c1))
        plt.imshow(self.zone1_count1, cmap='viridis')
        plt.subplot(1,2,2)
        plt.title('Dermal Counts: '+str(self.z2c1))
        plt.imshow(self.zone2_count1, cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=0.5)
        #plt.show()
        figsavepath = self.path_result+'/'+self.filename+'_plot3.png'
        fig.savefig(figsavepath,dpi=150,bbox_inches='tight')
        self.genpath.append(figsavepath)
        plt.clf()
    def plot4(self):
        z1c1_bbox = self.zone2_count1
        z1c2_bbox = self.zone2_count2
        z2c1_bbox = self.zone1_count1
        z2c2_bbox = self.zone1_count2
        fig,ax = plt.subplots(figsize=(10,10))
        ax.imshow(self.im2)

        for region in regionprops(z1c1_bbox):

            # skip small images
            # if region.area < 5:
                # continue

            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='green', linewidth=1)
            ax.add_patch(rect)

        for region in regionprops(z2c1_bbox):
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc-minc, maxr-minr, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)
        for region in regionprops(z1c2_bbox):
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc-minc, maxr-minr, fill=False, edgecolor='white', linewidth=1)
            ax.add_patch(rect)
        for region in regionprops(z2c2_bbox):
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc-minc, maxr-minr, fill=False, edgecolor='yellow', linewidth=1)
            ax.add_patch(rect)
        plt.title('Bounding Boxes: z1c1: grn, z1c2: wht, z2c1: rd, z2c2: yel')
        plt.xticks([])
        plt.yticks([])
        #plt.show()
        figsavepath = self.path_result+'/'+self.filename+'_plot4.png'
        fig.savefig(figsavepath, dpi=150, bbox_inches='tight')
        self.genpath.append(figsavepath)
        plt.clf()

    def result_summary(self):
        summary_info = [self.path_result,
                        self.result_dict['b'],
                        self.result_dict['g+b'],
                        self.result_dict['r+b'],
                        self.result_dict['r+g+b']]
        return summary_info



import csv

class SegReport():
    """this takes the segment_summary either directly or from a pickle and writes a csv for all the done images"""
    def __init__(self, segment_summary, output):
        self.output = output  # path for report
        self.segment_summary = segment_summary
    
    def report(self):
        #walk the same directories using ImageCrawl(should be in the segqueue?) and get the data.
        #need to make segment class generate a pickle of a results dictionary for each image.
        #digest_pattern = re.compile('^[/].*[/].*[.].*$')
        with open(self.output, 'w') as fout:
            report_file = csv.writer(fout)
            report_file.writerow(['root directory',
                                  'z1 b', 'z2 b', 'z3 b', 'z4 b',
                                  'z5 b', 'z6 b', 'z1 g+b', 'z2 g+b',
                                  'z3 g+b', 'z4 g+b', 'z5 g+b', 'z6 g+b',
                                  'z1 r+b', 'z2 r+b', 'z3 r+b', 'z4 r+b',
                                  'z5 r+b', 'z6 r+b', 'z1 r+g+b', 'z2 r+g+b',
                                  'z3 r+g+b', 'z4 r+g+b', 'z5 r+g+b', 'z6 r+g+b',])

            for entry in self.segment_summary:
                report_file.writerow([entry[0],
                                      entry[1]['zone 1'], entry[1]['zone 2'], entry[1]['zone 3'], entry[1]['zone 4'],
                                      entry[1]['zone 5'], entry[1]['zone 6'], entry[2]['zone 1'], entry[2]['zone 2'],
                                      entry[2]['zone 3'], entry[2]['zone 4'], entry[2]['zone 5'], entry[2]['zone 6'],
                                      entry[3]['zone 1'], entry[3]['zone 2'], entry[3]['zone 3'], entry[3]['zone 4'],
                                      entry[3]['zone 5'], entry[3]['zone 6'], entry[4]['zone 1'], entry[4]['zone 2'],
                                      entry[4]['zone 3'], entry[4]['zone 4'], entry[4]['zone 5'], entry[4]['zone 6']])
                
            
    
