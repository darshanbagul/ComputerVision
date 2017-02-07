import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import scipy.ndimage as ndi
import math

def plot_input(img, title):
    plt.imshow(img, cmap = 'gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()

class boundaryMelting:
    
    def __init__(self, filename):
        self.filename = filename
        self.image_preprocessing()
        print "Image Preprocessed"
        self.label_pixels()
        print "Pixels Labelled"
        self.calculate_region_avg_intensity()
        print "Average intensity per region calculated"
        self.initialise_crack_edge_data()
        print "Initialise crack edges to average intensities"
        self.fill_crack_edge_values()
        print "Calculated crack edges"
        self.update_perimeters()
        print "Perimeters updated between regions"
        self.elim_weak_edges()
        print "Eliminated weak edges by given criteria"
        self.superimpose_edges()
        print "Edges superimposed on main image, to show distinct regions"
        
        plt.imsave("result_gray.jpg",self.original_img, cmap='gray')
        plt.imsave("result_color.jpg", self.original_img)

        Or plot the output to visualise it
        plot_input(self.original_img, 'Region Segmentation - Boundary Melting')
        plt.imshow(self.original_img)
        plt.show()

    def image_preprocessing(self):
        self.original_img = cv2.imread(self.filename, 0)
        self.gaussian_smooth_img = ndi.filters.gaussian_filter(self.original_img, 1.2)
        [self.M,self.N] = self.original_img.shape
        self.adj_values = np.array([ {'left': 0, 'right':0,'top':0,'down':0} for x in range(self.M * self.N + 1)])
        self.region_label = np.array([[ 0 for i in range(self.N)] for j in range(self.M)])
        self.avg_intensity_reg = []
        for i in range(self.M * self.N +1):
            self.avg_intensity_reg.append({'min':0,'max':0,'avg':0})

        self.crack_edge_data = np.zeros((2*self.M-1,2*self.N-1))
        self.label_data = np.zeros((2*self.M-1,2*self.N-1))

    def label_pixels(self):
        label = 0
        for i in range(self.M):
            for j in range(self.N):
                if self.region_label[i,j] == 0:
                    label = label + 1
                    pixel = self.original_img[i,j]
                    self.adj_values[label]['left'] = i
                    self.adj_values[label]['right'] = i
                    self.adj_values[label]['top'] = j
                    self.adj_values[label]['down'] = j
                    self.split_label_recursively(i,j,label,pixel)
        self.fin_label = label

    def split_label_recursively(self,i,j,label,pixel):
        [M, N] = [self.M,self.N]
        T1 = 5 # Tested for all values from 1 to 10, the higher the values the more lenient the merging, lower values create more regions and have high noise. 
        if (i < 0 or i >= M or j < 0 or j >= N) or self.region_label[i,j] != 0 or self.original_img[i,j] < pixel - T1 or self.original_img[i,j] > pixel + T1:
            return 

        pixel = self.original_img[i,j]

        if self.avg_intensity_reg[label]['min'] > pixel:
            self.avg_intensity_reg[label]['min'] = pixel
        elif self.avg_intensity_reg[label]['max'] < pixel:
            self.avg_intensity_reg[label]['max'] = pixel

        self.region_label[i,j] = label
        if self.adj_values[label]['left'] > i:
            self.adj_values[label]['left'] = i 
        elif self.adj_values[label]['right'] < i: 
            self.adj_values[label]['right'] = i 
        if self.adj_values[label]['top'] > j:
            self.adj_values[label]['top'] = j 
        elif self.adj_values[label]['down'] < j: 
            self.adj_values[label]['down'] = j
        self.split_label_recursively(i-1,j+1,label,pixel)
        self.split_label_recursively(i,j+1,label,pixel)
        self.split_label_recursively(i+1,j+1,label,pixel)
        self.split_label_recursively(i+1,j,label,pixel)
        self.split_label_recursively(i+1,j-1,label,pixel)


    def calculate_region_avg_intensity(self):
        for i in range(len(self.avg_intensity_reg)):
            self.avg_intensity_reg[i]['avg'] = (self.avg_intensity_reg[i]['max'] + self.avg_intensity_reg[i]['min'])/2

    def initialise_crack_edge_data(self):
        for i in range(self.M):
            for j in range(self.N):
                self.crack_edge_data[2*i,2*j] = self.avg_intensity_reg[self.region_label[i,j]]['avg']
                self.label_data[2*i,2*j] = self.region_label[i,j]

    def fill_crack_edge_values(self):
        for i in range(self.M):
            for j in range(self.N):
                if 2*j < len(self.crack_edge_data[0]) and 2*i+2 < len(self.crack_edge_data):
                    delta = self.crack_edge_data[2*i,2*j] - self.crack_edge_data[2*i+2,2*j]
                    if delta != 0:
                        self.crack_edge_data[2*i+1,2*j] = 0
                        self.label_data[2*i+1,2*j] = 0
                    else:
                        self.crack_edge_data[2*i+1,2*j] = self.crack_edge_data[2*i,2*j]
                        self.label_data[2*i+1,2*j] = self.label_data[2*i,2*j]
                if 2*j+2 < len(self.crack_edge_data[0]) and 2*i < len(self.crack_edge_data):
                    delta = self.crack_edge_data[2*i,2*j] - self.crack_edge_data[2*i,2*j+2]
                    if delta != 0:
                        self.crack_edge_data[2*i,2*j+1] = 0
                        self.label_data[2*i,2*j+1] = 0
                    else:
                        self.crack_edge_data[2*i,2*j+1] = self.crack_edge_data[2*i,2*j]
                        self.label_data[2*i,2*j+1] = self.label_data[2*i,2*j]
                if 2*i+1 < len(self.crack_edge_data) and 2*j+1 < len(self.crack_edge_data[0]):
                    self.crack_edge_data[2*i+1,2*j+1],self.label_data[2*i+1,2*j+1] = self.check_neighbours_crack_edge(2*i+1,2*j+1)


    def check_neighbours_crack_edge(self,i,j):
        pix_val = 0
        pixel_label = 0
        count = 0
        if i-1 >= 0:
            if self.crack_edge_data[i-1,j] != 0:
                pix_val = self.crack_edge_data[i-1,j]
                pixel_label = self.label_data[i-1,j]
            else:
                count = count + 1
        if j-1 >= 0:
            if self.crack_edge_data[i,j-1] != 0:
                pix_val = self.crack_edge_data[i,j-1]
                pixel_label = self.label_data[i,j-1]
            else:
                count = count + 1
        if i+1 < len(self.crack_edge_data):
            if self.crack_edge_data[i+1,j] != 0:
                pix_val = self.crack_edge_data[i+1,j]
                pixel_label = self.label_data[i+1,j]
            else:
                count = count + 1
        if j+1 < len(self.crack_edge_data[0]):
            if self.crack_edge_data[i,j+1] != 0:
                pix_val = self.crack_edge_data[i,j+1]
                pixel_label = self.label_data[i,j+1]
            else:
                count = count + 1
        if count >2:
            return 0,0
        else:
            return pix_val,pixel_label


    def adjacent_reg(self, i, j):
        lab1 = 0
        lab2 = 0
        test_labels = [0,0,0,0]
        if i-1 >= 0:
            if self.crack_edge_data[i-1,j] != 0 :
                test_labels[0] = self.label_data[i-1,j]
        if j-1 >= 0:
            if self.crack_edge_data[i,j-1] != 0 :
                test_labels[1] = self.label_data[i,j-1]
        if i+1 < len(self.crack_edge_data):
            if self.crack_edge_data[i+1,j] != 0 :
                test_labels[2] = self.label_data[i+1,j]
        if j+1 < len(self.crack_edge_data[0]):
            if self.crack_edge_data[i,j+1] != 0 :
                test_labels[3] = self.label_data[i,j+1]
        lab1 = test_labels[0]
        if lab1 == 0:
            lab1 = test_labels[1]
        else:
            lab2 = test_labels[1]
        if lab1 == 0:
            lab1 = test_labels[2]
        else:
            lab2 = test_labels[2]
        if lab1 == 0:
            lab1 = test_labels[3]
        else:
            lab2 = test_labels[3]
        return lab1,lab2

    def update_perimeters(self):
        self.common_perimeter = np.array([[ 0 for i in range(self.fin_label+1)] for j in range(self.fin_label+1)])
        self.region_perimeter = np.array([ 0 for i in range(self.fin_label+1)]) 

        for i in range(len(self.crack_edge_data)):
            for j in range(len(self.crack_edge_data[0])):
                R1 = 0
                R2 = 0
                if self.crack_edge_data[i,j] == 0:
                    [R1,R2] = self.adjacent_reg(i,j)
                    if R1 != 0:
                        self.region_perimeter[int(R1)] = self.region_perimeter[int(R1)] + 1
                    if R2 != 0:
                        self.region_perimeter[int(R2)] = self.region_perimeter[int(R2)] + 1
                        if R1 != 0:
                            self.common_perimeter[int(R1),int(R2)] = self.common_perimeter[int(R1),int(R2)] + 1
                            self.common_perimeter[int(R2),int(R1)] = self.common_perimeter[int(R2),int(R1)] + 1


    def merge_reg(self,R1, R2):
        reg = 0
        if self.avg_intensity_reg[int(R1)]['avg'] < self.avg_intensity_reg[int(R2)]['avg']:
            reg = self.avg_intensity_reg[int(R1)]['avg']
            replacement = self.avg_intensity_reg[int(R2)]['avg']
            R = R2
        else:
            reg = self.avg_intensity_reg[int(R2)]['avg']
            replacement = self.avg_intensity_reg[int(R1)]['avg']
            R = R1

        for i in range(2* self.adj_values[int(reg)]['left'],2*self.adj_values[int(reg)]['right']):
            for j in range(2*self.adj_values[int(reg)]['top'],2*self.adj_values[int(reg)]['down']):
                if self.crack_edge_data[i,j] == reg:
                    self.crack_edge_data[i,j] = replacement
                    self.label_data[i,j] = R
                if self.crack_edge_data[i,j] == 0:
                    [R3,R4] = self.adjacent_reg(i,j)
                    if (R3 == R1 and R4 == R2) or (R3 == R2 and R4 == R1):
                        self.crack_edge_data[i,j] = replacement
                        self.label_data[i,j] = R


    def elim_weak_edges(self):
        edge_rem_threshold = 0.8
        for i in range(len(self.crack_edge_data)):
            for j in range(len(self.crack_edge_data[0])):
                R1 = 0
                R2 = 0
                if self.crack_edge_data[i,j] == 0:
                    [R1,R2] = self.adjacent_reg(i,j)
                    if R1 != 0 and R2 != 0:
                        W = self.common_perimeter[int(R1),int(R2)]
                        L1 = self.region_perimeter[int(R1)]
                        L2 = self.region_perimeter[int(R2)]
                        if L1 <= L2:
                            min_perimeter = L1
                        else:
                            min_perimeter = L2
                        score = W/min_perimeter
                        if score >= edge_rem_threshold:
                            self.merge_reg(R1,R2)


    def superimpose_edges(self):
        for i in range(len(self.crack_edge_data)):
            for j in range(len(self.crack_edge_data[0])):
                if self.crack_edge_data[i,j] == 0:
                    self.original_img[int((i+1)/2),int((j+1)/2)] = 255

if __name__ == "__main__":
    solution = boundaryMelting('./MixedVegetables.jpg');