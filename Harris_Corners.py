#importing libraries
import cv2 
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# Opening and displaying the original image
InputIm_FileName="image3.jpg" #opening an image
InImg=cv2.imread(InputIm_FileName,0) #reading the original image in grayscale

H,W=InImg.shape #defining the values of width and height of the image 
matrix_R = np.zeros((H, W)) #defining the matrix of the image 
DPI=96 #setting the value of the dots per inch parameter of the image /// image1,2,5 - 96, image3,6 - 72, image4 - 300
plt.figure(figsize=(W/DPI+1,H/DPI+1)) #setting the figure size to the size of the image with the option figsize =(s1,s2) as argument
plt.imshow(InImg,cmap = 'gray') #displaying the image
plt.suptitle('The gray scale input image') #adding a subtitle for the figure
plt.show()
InImg

    
# Step 1 - Computing image gradients 
# Computing the image gradients on x
dx = cv2.Sobel(InImg, cv2.CV_64F, 1, 0, ksize=3) #computing the image gradient on x (image derivative dx)
print('x-gradient', dx) #displaying the array of the image gradient on x
plt.imshow(dx,cmap = 'gray') #displaying the plot of the image gradient on x
plt.suptitle('The image gradients on x axis')
plt.show()

# Computing the image gradients on y
dy = cv2.Sobel(InImg, cv2.CV_64F, 0, 1, ksize=3) #computing the image gradient on y (image derivative dy)
print('y-gradient', dy) #displaying the array of the image gradient on y
plt.imshow(dy,cmap = 'gray') #displaying the plot of the image gradient on y
plt.suptitle('The image gradients on y axis')
plt.show()


# Step 2 - Subtract mean from each image gradient to center the data (remove the DC offset)
InImg_sobel=np.hypot(dx,dy) # Computing square root of the sum of squares
print('Sobel', InImg_sobel) #displaying the array of the image gradient on both axes
plt.imshow(InImg_sobel,cmap = 'gray') #displaying the plot of the square root of the sum 
plt.suptitle('The image gradients on both axis')
plt.show()


#Step 3 - Compute the elements of the covariance matrix

#Calculate product and second derivatives (dxx, dyy and dxy)
dxx = gaussian_filter(dx**2, sigma=1)
dxy = gaussian_filter(dy*dx, sigma=1)
dyy = gaussian_filter(dy**2, sigma=1)


#Step 4 - Harris response calculation
k = 0.05 #sensitivity factor to separate corners from edges
window_size = 5 #setting the window size
offset = int( window_size / 2 ) #setting the offset to half the window size

for y in range(offset, H-offset):
        for x in range(offset, W-offset):
            #calculate the sum of squares at each pixel by shifting a window over all the pixels in our image
            Sxx = np.sum(dxx[y-offset:y+1+offset, x-offset:x+1+offset]) 
            Syy = np.sum(dyy[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(dxy[y-offset:y+1+offset, x-offset:x+1+offset])
           
            window_matrix = np.array([[Sxx,Sxy],[Sxy,Syy]]) #defining the window matrix
            # print('Window Matrix', window_matrix)
           
            #Harris response calculation for each pixel
            det = (Sxx * Syy) - (Sxy**2) #compute the determinant
            trace = Sxx + Syy #compute the trace 
            r = det - k*(trace**2) #harris response
            matrix_R[y-offset, x-offset] = r #array of peak values of each row in the image 
     
            
#Step 5 - Use threshold on eigenvalues to detect corners
new_img = cv2.cvtColor(InImg,cv2.COLOR_GRAY2RGB) #converting the image from grayscale to rgb
threshold = 0.25 #setting the threshold ///good for all images

cv2.normalize(matrix_R, matrix_R, 0, 1, cv2.NORM_MINMAX)
for y in range(offset, H-offset):
        for x in range(offset, W-offset):
            value=matrix_R[y, x] 
            if value>threshold:
                  cv2.circle(new_img,(x,y),3,(0,255,0))  #adding green dots where a corner is found

#Step6 - Displaying the final result 
plt.figure(figsize=(W/DPI+1,H/DPI+1)) 
plt.imshow(new_img) #displaying the image with the corners found
plt.title("Harris corners")
plt.savefig('Harris_corner_detection_result_%s.png' %(threshold), bbox_inches='tight')
plt.show()





