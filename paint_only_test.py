import matplotlib
matplotlib.use('TkAgg')
from CellSegmentTest import *
import os

test_data = {'fname': 'Snap-8904-Image Export-06',
             'path': '/Users/wesleywong/Desktop/segment_test/Snap-8904-Image Export-06/Snap-8904-Image Export-06_c1+2+3.jpg',
             'mask': '/Users/wesleywong/Desktop/segment_test/Snap-8904-Image Export-06/Snap-8904-Image Export-06_mask.png',
             'output': '/Users/wesleywong/Desktop/test'}

fout = '/'.join([test_data['output'],'test.png'])

painter = Paint(test_data['path'],fout)
painter.mainloop()

#print('cleaning up')
#os.remove(fout)