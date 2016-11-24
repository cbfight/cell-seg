import matplotlib
matplotlib.use('TkAgg')
import sys
import datetime
import os
import pickle
from CellSegmentTest import *

def existmake(path):
    if not os.path.exists(path):
        print('Creating: %s' %path)
        os.mkdir(path)

####
#master_dir = '/'.join([os.path.expanduser('~/'),'CellSegmentBeta'])
master_dir = os.path.expanduser('~/')+'CellSegmentBeta'
report_dir = '/'.join([master_dir,'reports'])
pq_pickle = '/'.join([master_dir,'pq_pickle.bin'])
seg_pickle = '/'.join([master_dir,'seg_pickle.bin'])
summ_pickle = '/'.join([master_dir,'summ_pickle.bin'])
###
existmake(master_dir)
existmake(report_dir)
###
paintqueue = []
segqueue = []
summary_list = []
###
print('Manual Annotation Segmentation Tool Rev 11-15-16')
sess_type = input('To start a NEW session enter NEW. To RESUME the previous session, enter RESUME: ')
if sess_type == 'RESUME':
    #Load Paint on a queued image
    try:
        with open(pq_pickle,'rb') as fin:
            paintqueue = pickle.load(fin)
    except:
        print('Could not find previous session files! Bye!')
        sys.exit('NO PAINT PICKLE FOUND')
    try:
        with open(seg_pickle,'rb') as fin:
            segqueue = pickle.load(fin)
    except:
        print('Could not find previous session files! Bye!')
        sys.exit('NO SEG PICKLE FOUND')
        
    num_ims_tot = len(paintqueue)
    num_ims_done = 1
    
    print('Images remaining in current session: %d' % num_ims_tot)
    request_next = input('Would you like the next image? Y/N: ')
    while request_next != 'N':
        print('Loading: %s' % paintqueue[-1][0])
        print('Count: %d/%d' % (num_ims_done,num_ims_tot))
        print('Index: %d' % paintqueue.index(paintqueue[-1]))
        painter = Paint(paintqueue[-1][1],paintqueue[-1][2])
        painter.mainloop()
        print('Deleting Previous Item...')
        segqueue.append(paintqueue[-1])
        del paintqueue[-1]
        print('Saving...')

        with open(pq_pickle,'wb') as fout:
            pickle.dump(paintqueue,fout)

        with open(seg_pickle,'wb') as fout:
            pickle.dump(segqueue,fout)
        print('Saved')
        num_ims_done += 1
        print('Images remaining in current session: %d' % len(paintqueue))
        request_next = input("Continue? Y/N: ")
    
    #print('Session Terminated. Goodbye')
    #sys.exit('SESSION CLOSED BY USER')
    
if sess_type == 'NEW':
    warning = input('********************\nWARNING: PREVIOUS SESSION FILES WILL BE WIPED: Y/N \n********************\n')
    if warning == 'Y':
        print('Removing previous session files...')
        try:
            os.remove(pq_pickle)
        except:
            print("No pq_pickle found! Skipping.")
        try:
            os.remove(seg_pickle)
        except:
            print("No seg_pickle found! Skipping.")
        try:
            os.remove(summ_pickle)
        except:
            print("No summ_pickle found! Skipping.")
    else:
        sys.exit('SESSION CLOSED BY USER')
        
        
    
    work_path = input('Type full path of top level directory: ')
    if os.path.isdir(work_path):
        print('Path verified to lead to existing directory')
        print('Searching for images...')
        image_obj = ImageCrawler()
        image_obj.crawl(work_path)
        images = image_obj.target_images
        print('Search completed')
        print('Creating new save files...')
        try:
            with open(pq_pickle,'xb') as fout:
                pickle.dump(images,fout)
        except: 
            print("Problem creating pickle! Bye!")
            sys.exit("CAN'T MAKE PQ_PICKLE. MANUALLY DELETE PICKLES AND TRY AGAIN.")
        
    else:
        print('I could not verify this directory exists. Session canceled. Goodbye.')
        sys.exit('NO DIRECTORY FOUND')
        
    
    
    #Load Paint on a queued image
    with open(pq_pickle,'rb') as fin:
        paintqueue = pickle.load(fin)    
    num_ims_done = 1
    num_ims_tot = len(paintqueue)
    print('Images remaining in current session: %d' % len(paintqueue))
    request_next = input('Would you like the next image? Y/N: ')
    while request_next == 'Y':
        print('Loading: %s' % paintqueue[-1][0])
        print('Count: %d/%d' % (num_ims_done,num_ims_tot))
        print('Index: %d' % paintqueue.index(paintqueue[-1]))
        painter = Paint(paintqueue[-1][1],paintqueue[-1][2])
        painter.mainloop()
        print('Deleting Previous Item...')
        segqueue.append(paintqueue[-1])
        del paintqueue[-1]
        print('Saving...')

        with open(pq_pickle,'wb') as fout:
            pickle.dump(paintqueue,fout)

        with open(seg_pickle,'wb') as fout:
            pickle.dump(segqueue,fout)
        print('Saved')
        num_ims_done += 1
        print('Images remaining in current session: %d' % len(paintqueue))
        request_next = input("Continue? Y/N: ")
        
print('Preparing list of images ready for segmentation...')
try:
    with open(seg_pickle,'rb') as fin:
        segqueue = pickle.load(fin)
except: 
    print('UNABLE TO LOAD SEGQUEUE')
    sys.exit('NO SEG_PICKLE')

num_to_seg = len(segqueue)
seg_time_est = ((num_to_seg*0.25),(num_to_seg*1.0))
if input('There are %d images annotated and ready to process. Proceed? Estimated time required: %d-%d min. Y/N: ' % (num_to_seg,seg_time_est[0],seg_time_est[1])) == 'Y':
    print('********************\nWARNING: THERE IS NO GRACEFUL RESUME FOR SEGMENTATION!\n********************')
    print('Segmenting. Please be patient...')
    for image in segqueue:
        print('Segmenting: %s' % image[0])
        segApp = Segmenter(image[0],image[1],image[2],image[3])
        segApp.color_rules()
        segApp.get_regions()
        segApp.segment()
        #segApp.plot1()
        #segApp.plot2()
        #segApp.plot3()
        #segApp.plot4()
        summary_list.append(segApp.result_summary())
        #plt.close('all')
        print('Done')
    try:
        with open(summ_pickle,'wb') as fout:
            print('Saving summary')
            pickle.dump(summary_list,fout)
    except:
        print('UNABLE TO WRITE SUMMARY')
        sys.exit("CAN'T WRITE SUMM_PICKLE")
else:
	sys.exit('SESSION CLOSED BY USER')
if input('Write report? Y/N: ') == 'Y':
    try: 
        with open(summ_pickle,'rb') as fin:
            summary_list = pickle.load(fin)
    except:
        print("UNABLE TO LOAD SUMMARY")
        sys.exit("CAN'T LOAD SUMM_PICKLE")
        
    ts = str(datetime.datetime.now()).replace(':','-').replace('.','-')
    report_fname = '.'.join([ts,'csv'])
    report = SegReport(summary_list,output='/'.join([report_dir,report_fname]))
    report.report()
    print('Wrote report to: %s' % report_fname)
    
print('Session Terminated. Goodbye')
sys.exit('SESSION CLOSED BY USER')

        

        

