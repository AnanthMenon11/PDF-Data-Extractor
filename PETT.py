#Imports
import os
import camelot
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from pdf2image import convert_from_path
import PIL
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.summarization.summarizer import summarize
pytesseract.pytesseract.tesseract_cmd = r'D:\Users\ASreeku1\AppData\Local\Tesseract-OCR\tesseract.exe'



def save_Txt(cD,txt,pnum):
    text_file = open(str(cD)+"/TextFiles/Page"+str(pnum)+"_Text.txt", "w")
    n = text_file.write(txt)
    text_file.close()

def viewPage(PageNum,ctables): #Displays the Tables and Text on a particular Sheet
    print("THE EXTRACTED TEXT IS BELOW:\n")
    for c in dict_extracted_text[PageNum].replace("/n","").split("#.#.#.#.#"):
        if c!="":
			print(c)
    print("\nEND OF EXTRACTED TEXT.\n\n")
    i=0
    for tab in ctables:
        i=i+1
        if tab.parsing_report['page']==PageNum:
            print("Table "+str(tab.parsing_report['order']))
            print(tab.df)

def get_Order_Page(ctab,dest):#Returns the Number of tables in each page of the document
    i=0
    dict_tab=dict()
    for tab in ctab:
        if tab.df.shape[0]>2:# & tab.df.shape[1]>2:
            if tab.parsing_report['page'] in dict_tab.keys():
                if dict_tab[tab.parsing_report['page']]<tab.parsing_report['order']:
                    dict_tab[tab.parsing_report['page']]=tab.parsing_report['order']
            else:
                dict_tab[tab.parsing_report['page']]=tab.parsing_report['order']
            tab.to_excel(str(dest)+"/ExcelFiles/"+"Page"+str(tab.parsing_report['page'])+"_Table"+str(tab.parsing_report['order'])+'.xlsx')
    return dict_tab

def resize(yo):
    scale_percent = 35 # percent of original size
    width = int(yo.shape[1] * scale_percent / 100)
    height = int(yo.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(yo, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Resized image" , resized)
    cv2.waitKey(0)
	cv2.destroyAllWindows()

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def find_tables(img,num_t):
    img = np.array(img)
    Originalimg = img.copy()# Read the image
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #grayscale

    ret, mask = cv2.threshold(img, 2, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img, img, mask=mask)
    ret, img_bin = cv2.threshold(image_final, 240, 255, cv2.THRESH_BINARY_INV)  # for black text , cv.THRESH_BINARY_INV

    #resize(Originalimg)
    #resize(img_bin)

    replicate = cv2.copyMakeBorder(img_bin,10,10,10,10,cv2.BORDER_REPLICATE)
    #resize(replicate)

    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//80

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    #cv2.imwrite("verticle_lines.jpg",verticle_lines_img)
    #resize(verticle_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    #cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)
    #resize(horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    #cv2.imwrite(cropped_dir_path,img_final_bin)
    #resize(img_final_bin)
    Black_ened= blacken_tables(Originalimg,img_final_bin,num_t)
    return Black_ened

def blacken_tables(image,line_image,num_tab):
    OriginalImage=image.copy()
    image=np.array(image)
    line_image = np.array(line_image)
    #img2gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY) #grayscale
    ret, mask = cv2.threshold(line_image, 240, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(line_image, line_image, mask=mask)
    ret, thresh = cv2.threshold(image_final, 240, 255, cv2.THRESH_BINARY_INV)  # for black text , cv.THRESH_BINARY_INV
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,3))
    dilate = cv2.dilate(thresh,kernel,iterations=2)

    new=OriginalImage.copy()
    cnts= cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts)==2 else cnts[1]
    area = [cv2.contourArea(c) for c in cnts]
    Z = pd.Series(data=cnts,index=area).sort_index().tolist()
    Z = Z[len(Z)-num_tab:]
    for i in Z:
        print("Blackening Table at Position (x,y,w,h) ",cv2.boundingRect(i))
        cv2.drawContours(image, [i], -1, (0,0,0), thickness=-1)
    return image


def extract_text(z):
    #pytesseract.pytesseract.tesseract_cmd = r'D:\Users\ASreeku1\AppData\Local\Tesseract-OCR\tesseract.exe'
    extracted_text = ''

    open_cv_image = np.array(z)
    img = open_cv_image[:,:,::-1].copy()
    image = img

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)
    thresh= cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    dilate = cv2.dilate(thresh,kernel,iterations=4)
    cnts= cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts)==2 else cnts[1]
    cnts,boundingBoxes =sort_contours(cnts, method="left-to-right")
    cnts,boundingBoxes =sort_contours(cnts, method="top-to-bottom")
    i=0
    for c in cnts:
        i=i+1
        x,y,w,h = cv2.boundingRect(c)
        if(cv2.contourArea(c)>3400):
            cv2.rectangle(image,(x,y),(x+w,y+h),(36,255,12),2)
            cv2.putText(image,str(i)+" Area: "+str(cv2.contourArea(c)), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            result = pytesseract.image_to_string(gray[y:y+h,x:x+w])
            extracted_text = extracted_text +'/n#.#.#.#.#/n'+result

    #resize(image)
    return extracted_text



def extractData_PDF(path):
    #Create Folders to Save the Extracted Data
    createDir=path[0:len(path)-4]
    os.mkdir(createDir)
    os.mkdir(os.path.join(createDir, "Text_LOC"))
	os.mkdir(os.path.join(createDir, "Excel_LOC"))

    #Extract Pages
    pages = convert_from_path(path,150)
    print("Pages Extracted.")

    #Extract all tables from PDF
    ctables = camelot.read_pdf(path, pages="all")
    print("Tables Extracted. The number of tables: ",len(ctables))
    i=0 #Page Number Count

    #Get the page number and number of tables on the page
    #Save the necessary tables into files
    dict_table_loc=get_Order_Page(ctables,createDir)
    print("Tables Saved.")
    dict_extracted_text=dict()
    #Traverse each page to see if any table exists on the page. If so, Blacken it so that text is not read from the table.
    for page in pages:
        i=i+1
        if i in dict_table_loc:
            new_page=find_tables(page,dict_table_loc[i])
            page_text=extract_text(new_page)
        else:
            page_text=extract_text(page)
        dict_extracted_text[i]=page_text
        print("Extraction From Page "+str(i)+ " Complete.")
        save_Txt(createDir,page_text,i)
        print("Text from Page Saved.")
    print("All Tables and Text extracted from "+str(path))


extractData_PDF("<Enter file path here>")
# ctab=camelot.read_pdf("GUS-C-19-GBR-001-V01-M01_Approval.pdf", pages="all")

