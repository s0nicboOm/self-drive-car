import numpy as np

from PIL import ImageGrab

import cv2

import imutils

import time

from threading import Thread

import sys
from Queue import Queue

import pyautogui

from directkeys import PressKey,ReleaseKey, W, A, S, D


class Main:
    
	def convert_hls(self,image):
        
		return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

   
	def apply_smoothing(self, image, kernel_size=19):

	        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    
	def convert_gray_scale(self,image):
        
		return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    
	def detect_edges(self,image, low_threshold=50, high_threshold=150):
       
		 image=self.convert_gray_scale(image)
        
		 image=self.apply_smoothing(image)
        
		 return cv2.Canny(image, low_threshold, high_threshold)

    
	def filter_region(self,image, vertices):
      
        
		mask = np.zeros_like(image)
        
		if len(mask.shape)==2:
            
			cv2.fillPoly(mask, vertices, 255)
       
		else:
            
			cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) 
        
		return cv2.bitwise_and(image, mask)

        
    
	def select_region(self,image):

      
          bottom_left  = [75,700]

       	        top_left     = [75,400]

        	bottom_right = [825,700]
	      
  top_right    = [825,400]

                vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        		return self.filter_region(image, vertices)



    	def select_white_yellow(self,image):

        	converted = self.convert_hls(image)
        # white color mask

        	lower = np.uint8([  0, 200,   0])

        	upper = np.uint8([255, 255, 255])

        	white_mask = cv2.inRange(converted, lower, upper)
        # yellow color mask

        	lower = np.uint8([ 10,   0, 100])
        upper = np.uint8([ 40, 255, 255])

        	yellow_mask = cv2.inRange(converted, lower, upper)
        # combine the mask

        	mask = cv2.bitwise_or(white_mask, yellow_mask)

        	return cv2.bitwise_and(image, image, mask = mask)



        def hough_lines(self,image):
     
		minLineLength = 100

        	maxLineGap = 10

        	image = cv2.Canny(image,50,150,4)

        	lines = cv2.HoughLines(image,1,np.pi/90,40)

        	return lines

    

        def find_x_y(self,rho,theta):

     	   	a = np.cos(theta)

        	b = np.sin(theta)

        	x0 = a*rho

        	y0 = b*rho

        	x1 = int(x0 + 1000*(-b))

        	y1 = int(y0 + 1000*(a))

        	x2 = int(x0 - 1000*(-b))

        	y2 = int(y0 - 1000*(a))

        	return x1,y1,x2,y2

          
    
   	 def slope(self,p1, p2) :

   	        return (p2[1] - p1[1]) * 1. / (p2[0] - p1[0])


	 def y_intercept(self,slope, p1) :

       		return p1[1] - 1. * slope * p1[0]



    	 def process(self,image):

        	color_select = np.copy(image)

       
 	red_threshold = 200
  
      	green_threshold = 200  

        	blue_threshold = 200  
 
        	rgb_threshold = [red_threshold, green_threshold, blue_threshold]
  

        	thresholds = (image[:, :, 0] < rgb_threshold[0]) | (image[:, :, 1] < rgb_threshold[1]) | (image[:, :, 2] < rgb_threshold[2]) 
 
        	color_select[thresholds] = [0, 0, 0]

        	return color_select
 
 
        
	 def intersect(self,line1, line2) :

       		min_allowed = 1e-5   # guard against overflow

	        big_value = 1e10     # use instead (if overflow would have occurred)

                m1 = self.slope(line1[0], line1[1])
       #print( 'm1: %d' % m1 )

                b1 = self.y_intercept(m1, line1[0])
       #print( 'b1: %d' % b1 )

                m2 = self.slope(line2[0], line2[1])
       #print( 'm2: %d' % m2 )

                b2 = self.y_intercept(m2, line2[0])
       #print( 'b2: %d' % b2 )

                if abs(m1 - m2) < min_allowed :

          		x = big_value
       
		else :
          
			x = (b2 - b1) / (m1 - m2)

       		y = m1 * x + b1

	        y2 = m2 * x + b2
       #print( '(x,y,y2) = %d,%d,%d' % (x, y, y2))

                return (int(x),int(y))

    
	
	def draw_lines(self,image,lines):

        
        try:
            
			acount = 1

            		ocount = 1

            		ax1=ay1=ax2=ay2=0.0

            		ox1=oy1=ox2=oy2=0.0

            		x = y = 0.0

            
            for line in lines:

                		for rho,theta in line:

                    			if theta > -np.pi/18 and theta < np.pi/18:

                        			continue
                    
					if theta < np.pi/2:

			                        ax1,ay1,ax2,ay2 = self.find_x_y(rho,theta)

			                        acount += 1

			                        cv2.line(image,(ax1,ay1),(ax2,ay2),(255,255,255),2)
 
			                        #PressKey(D)

			                        #time.sleep(0.1*(ocount-acount))
			
                        #ReleaseKey(D)

                    			if theta > np.pi/2:

			                        #PressKey(A)

			                        ocount += 1

			                        #time.sleep(0.1*(acount-ocount))
			
                        #ReleaseKey(A)
			
                        ox1,oy1,ox2,oy2 = self.find_x_y(rho,theta)
			
                        cv2.line(image,(ox1,oy1),(ox2,oy2),(255,255,255),2)

	                if acount > ocount:

		                PressKey(D)
		
                time.sleep(0.1)
		
                ReleaseKey(D)

		                elif acount < ocount:
			
                PressKey(A)
			
                time.sleep(0.1)

                			ReleaseKey(A)

                
        		except Exception as e:

            				pass



        def steering_thread(self,q):

	        point = q.get()
    
	        x = point[0]

                y = point[1]
 
                tryno = 0

                centerPointx = 350
	
        #print("Inside the steering thread")
	
        if(x>centerPointx):
		
            PressKey(D)
		
            PressKey(S)
		
            time.sleep(0.1)
		
            ReleaseKey(D)
 
		            ReleaseKey(S)

		elif(x<centerPointx):

		            PressKey(A)
		
            PressKey(S)
		
            time.sleep(0.1)

		            ReleaseKey(A)
		
            ReleaseKey(S)

		            tryno+=1

 
   
    def accelerate(self,count):

      	        PressKey(W)
	
        time.sleep(8)

	        ReleaseKey(W)

	        count+=1

	        if count == 5:

              		    count = 0
 
		            PressKey(S)

            		    time.sleep(5)

            		    ReleaseKey(S)
    

   def main(self):

        	count = 0

	        time.sleep(5)
	
        queue = Queue()
	
        while(True):
		
            #PressKey(W)
		
            #time.sleep(0.3)
		
            thread1 = Thread( target=self.accelerate, args=(count,) )

		            thread1.start()
		
            screen=np.array(ImageGrab.grab(bbox=(0,0,900,900)))
		
            original=screen
		
            original=self.select_region(original)

		            screen=self.select_white_yellow(self.process(screen))

		            screen=self.detect_edges(screen)
		
            screen=self.select_region(screen)

		            #screen = process(screen)
		
            #screen = connn(screen)
		
            lines=self.hough_lines(screen)
		
            self.draw_lines(screen,lines)

		            #corner=select_region_corner(screen)

		            #draw_lines(screen,lines)
		
            #left,right=lane_lines(screen,lines)

		            #screen=draw_lane_lines(original,(left,right))

		            cv2.imshow('grabbed_screen',screen)
		
            #cv2.imshow('original',original)
		
            #cv2.imshow('original',corner)

		            if cv2.waitKey(25) & 0xFF == ord('q'):

		                cv2.destroyAllWindows()

	   	            break

		m = Main()

		m.main()