#imports
import fitz
import numpy as np
import math
import cv2
import pytesseract
from pytesseract import Output
import time
from datetime import timedelta

# Path to the location of the Tesseract-OCR executable/cmd
pytesseract.pytesseract.tesseract_cmd =r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#CONSTANTS
DATE ='Date'
METRO_DICT = 'Metro'
IFT_DICT = 'IFT'
SUBR_DICT ='SubR'
SCHEDULED = 'Scheduled'
STAFFED = 'Staffed'
BLS ='BLS'
TANGO = 'Tango'
TESSERACT_DATE_CONFIG = r'--psm 7 -c tessedit_char_whitelist=/0123456789'
TESSERACT_NUMBER_CONFIG = r'--psm 7 -c tessedit_char_whitelist=0123456789'
CSV_HEADEERS = "Page, Date, Metro Scheduled, Metro Staffed, Metro BLS, Sub/Rural Scehduled, Sub/Rural Staffed, Sub/Rural BLS, IFT Scheduled, IFT Staffed, IFT BLS, IFT Tango\n"
TRIM_HEIGHT = 120

#MAGIC_NUMBERS
X_THRESHOULD = 120
DATE_BOX_SIZE_RANGE = (145,35,160,50) #min width, min Height, max width, max Height
DATA_TABLE_SIZE_RANGE = (1000,185,1100,200) #min width, min height, max width, max height
DATA_CELL_SIZE_RANGE =(80,25, 140, 40) #min width, min height, max width, max height
MASK_CELL_SIZE_RANGE =(70, 20, 135, 30) #min width, min height, max width, max height
MIN_CELL_WIDTH = 15
ROW_ONE_Y_MIN = 10
ROW_THREE_Y_MIN = 50 
CROP_PIXEL = (5,4)
TABLE_SHIFT_START = (104,38)
TABLE_SHIFT_END = (568,150)

#Objects
def sortOnX(elm):
    x = elm['x']
    return x

class TableDataContainer:
    def __init__(self):
        self.ROW001 = []
        self.ROW002 = []
        self.ROW003 = []
    
    def addData(self,cellDict):
        y = cellDict['y']
        if(y<ROW_ONE_Y_MIN):
            self.ROW001.append(cellDict)
        if(y > ROW_ONE_Y_MIN and y < ROW_THREE_Y_MIN):
            self.ROW002.append(cellDict)
        if y >ROW_THREE_Y_MIN:
            self.ROW003.append(cellDict) 

    def sortData(self):
        self.ROW001.sort(key=sortOnX)
        self.ROW002.sort(key=sortOnX)
        self.ROW003.sort(key=sortOnX)

    def toCSVstring(self,strinVer=0):
        csvString = ""
        for elem1 in self.ROW001:
            csvString = csvString + elem1['data'] + ", "
        for elem2 in self.ROW002:
            csvString = csvString + elem2['data'] + ", "
        for elem3 in self.ROW003:
            csvString = csvString + elem3['data'] + ", "
        
        if len(csvString)>2:
            csvString = csvString[:-2]
            csvString = csvString +'\n'

        return csvString


class PageData:
    def __init__(self):
        self.dateString = ""
        self.thisTableDataContainer = None
    
    def addData(self,data_value,container):
        self.dateString = data_value
        self.thisTableDataContainer = container
    
    def toCSVstring(self):
        outString = self.dateString
        if self.thisTableDataContainer != None:
            outString = outString +", " + self.thisTableDataContainer.toCSVstring()
        return outString


class StateAndMem:
    #class to manage some global data
    #want to be able to print recoreded data if things go catastrophicly awry
    def __init__(self):
        self.asked = 0
        self.pageDataArray =[]
    
    def askforhelp(self):
        self.asked = self.asked +1

    def howMuchHelpAsked(self):
        helped = self.asked
        return helped
    
    def appendData(self,pageData):
        self.pageDataArray.append(pageData)
    
    def getData(self):
        return self.pageDataArray

theWorld = StateAndMem()

def build3PartDict(scheduled="NaN",staffed="NaN",bls="Nan"):
    return{SCHEDULED:scheduled,STAFFED:staffed,BLS:bls}

def build4PartDict(scheduled="NaN",staffed="NaN",bls="Nan",tango="Nan"):
    return{SCHEDULED:scheduled,STAFFED:staffed,BLS:bls,TANGO:tango}
def buildEmptyResultDic():
    metroDict = build3PartDict()
    iftDict = build4PartDict()
    subrDict = build3PartDict()
    return{DATE:"NaN,",METRO_DICT:metroDict,IFT_DICT:iftDict,SUBR_DICT:subrDict}

def buildResultDict(date,metroDict,iftDict,subRDict):
      return{DATE:date,METRO_DICT:metroDict,IFT_DICT:iftDict,SUBR_DICT:subRDict}

def editDictKeybyIndex(dict,index,value):
    try:   
        match index:
            case 0: dict[SCHEDULED] = value
            case 1: dict[STAFFED] = value
            case 2: dict[BLS] = value
            case 3: dict[TANGO]=value
    except KeyError as e:
        print("Edit Dict Key by Index - Invalid Key",e)


#Converts Pix to np
def pix2np(pix):
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    return im

def trimHeaderFooter(image):
    #remove garbage header and footer data
    w,h,d = image.shape
    startPoint = (0,0+TRIM_HEIGHT)
    endPoint = (w,h-TRIM_HEIGHT)
    trim = image[startPoint[1]:endPoint[1],startPoint[0]:endPoint[0]]
    return trim

def detectAndcorrectSkew(image,thresh):
    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    correctedAngle = False
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
  
    #pull back to square - assumes roated clockwise
    if angle !=0:
        angle = 90 - angle
        correctedAngle = True
        # rotate the image to deskew it
        ( h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
        #Debug drawing code
        #rotated = np.copy(image)
        # draw the correction angle on the image so we can validate it
        #cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
        # show the output image
        #print("[INFO] angle: {:.3f}".format(angle))
        #cv2.imshow("Input", image)
        #cv2.imshow("Rotated", rotated)
        #cv2.waitKey(0)
    
    #olde code for counterclockwise? roatations
    #if angle < -45:
    #    angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    #else:
    #    angle = -angle 
    
    return correctedAngle, image

#Return Masks that can be used to find Tables
def makeTableMask(image):
    #greyscale and thresh for cvOpen operations
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    #check for scew and correct it
    didCorrectAngle,rotated = detectAndcorrectSkew(image,thresh)
    if didCorrectAngle:
        #refresh the Thresh
        image = rotated
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(thresh)
    vertical = np.copy(thresh)
    
    cols = horizontal.shape[1]
    horizontal_size = math.ceil(cols / 50)

    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = math.ceil(rows / 50)

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    res = vertical + horizontal

    return res, thresh

def confirmContourIsQuadrilateral(element):
    hasFourSides = False
    peri = cv2.arcLength(element, True)
    approx = cv2.approxPolyDP(element, 0.02 * peri, True)
    print()
    if len(approx ==4):
        hasFourSides = True

    return hasFourSides

def confirmBoxDimension(w,h,sizeRange):
    isDateBox = False
    if w>sizeRange[0] and h>sizeRange[1] and w<sizeRange[2] and h<sizeRange[3]:
        isDateBox = True
    
    return isDateBox

def captureCell(img, areaBox):
    x, y, w, h = cv2.boundingRect(areaBox)
    startPoint = (x+CROP_PIXEL[0],y+CROP_PIXEL[0])
    endPoint = (x+w-CROP_PIXEL[1],y+h-CROP_PIXEL[1])
    areaROI = img[startPoint[1]:endPoint[1],startPoint[0]:endPoint[0]]
    return areaROI

#Isolate and pre-Process the ROI representating the data table
# @img = Master Image representing the whole page
# @area box - provides dimensions of ROI
def captureAndCleanDataTable(img,areBox):
    x, y, w, h = cv2.boundingRect(areBox)
    startPoint = (x+TABLE_SHIFT_START[0],y+TABLE_SHIFT_START[1])
    endPoint = (x+TABLE_SHIFT_END[0],y+TABLE_SHIFT_END[1])

    #crop master image down to ROI
    tableROI = img[startPoint[1]:endPoint[1],startPoint[0]:endPoint[0]]
    
    #creat a new mask for cropped image
    maskROI,thresh = makeTableMask(tableROI)
    
    #Prevent Masking of cell contents / Only mask cell boarders
    maskContours = cv2.findContours(maskROI, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]

    #Make background all white
    imgH, imgW = maskROI.shape
    startPoint = (0,0)
    endPoint = (imgW,imgH)
    cornerDownLeft = startPoint[0],endPoint[1]
    cornerUpRight = endPoint[0],startPoint[1]
    contoursFillWhite = np.array([startPoint, cornerDownLeft, endPoint, cornerUpRight])
    cv2.fillPoly(maskROI, pts =[contoursFillWhite], color=(255,255,255))

    #Place black squares on onto white background where data will be located
    for element in maskContours:
        x, y, w, h = cv2.boundingRect(element)
        if confirmBoxDimension(w,h,DATA_CELL_SIZE_RANGE):
            cornerUpLeft = x+CROP_PIXEL[0],y+CROP_PIXEL[0]
            cornerDownLeft = x+CROP_PIXEL[0],y+h-CROP_PIXEL[1]
            cornerDownRight = x+w-CROP_PIXEL[1],y+h-CROP_PIXEL[1]
            cornerUpRight = x+w-CROP_PIXEL[1],y+CROP_PIXEL[0]
            contoursFillBlack = np.array([cornerUpLeft, cornerDownLeft, cornerDownRight, cornerUpRight])
            cv2.fillPoly(maskROI, pts =[contoursFillBlack], color=(0,0,0))

    clean = thresh - maskROI
    table = clean.copy()

    #draw back in rows and columns
    # white color in BGR
    whiteLines = (255, 255, 255)
    # Line thickness of 2 px
    thickness = 1
    #draw 2 Horizontal Lines
    horizonalLineSpacing = math.ceil(imgH / 3)
    lineStartPoint =(0,horizonalLineSpacing)
    lineEndPoint = (imgW,horizonalLineSpacing)
    cv2.line(table,lineStartPoint,lineEndPoint,whiteLines,thickness)

    lineStartPoint = (0, horizonalLineSpacing + horizonalLineSpacing)
    lineEndPoint = (imgW, horizonalLineSpacing + horizonalLineSpacing)
    cv2.line(table,lineStartPoint,lineEndPoint,whiteLines,thickness)
    
    #draw 3 Vertical Lines
    verticalLineSpaceing = math.ceil(imgW / 4)
    lineStartPoint = (verticalLineSpaceing,0)
    lineEndPoint = (verticalLineSpaceing,imgH)
    cv2.line(table,lineStartPoint,lineEndPoint,whiteLines,thickness)
    
    lineStartPoint= (verticalLineSpaceing+verticalLineSpaceing,0)
    lineEndPoint= (verticalLineSpaceing + verticalLineSpaceing,imgH)
    cv2.line(table,lineStartPoint,lineEndPoint,whiteLines,thickness)
    
    lineStartPoint =  (verticalLineSpaceing+verticalLineSpaceing+verticalLineSpaceing, 0)
    lineEndPoint = (verticalLineSpaceing+verticalLineSpaceing+verticalLineSpaceing,imgH)
    cv2.line(table,lineStartPoint,lineEndPoint,whiteLines,thickness)
   
    #Find conntours works with black background / white objects
    table  = cv2.bitwise_not(table)
    maskROI = cv2.bitwise_not(maskROI)
    #Table ROI - image taken striaght from page
    #Mask ROI - mask to isolate cells
    #clean - image with just numbers
    #table - Grid lines drawn on clean
    return tableROI, maskROI, clean, table

# for a sanitized Data Table
# Pro - might be faster
# Con - unable to ask a human for help
def proccessImageToData(img,configString):
    #Pass in clean image / is already black & white

    processedImg = processImageAlgorithmDefault(img)
   # cv2.imshow("Original Window",img)
   # cv2.imshow("Pocessed IMage for Image to data",processedImg)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()
    data = pytesseract.image_to_data(processedImg,config=configString,output_type=Output.DICT)

    #iterate over returned dictionary and create a list for each row
    rowOneList =[]
    rowTwoList = []
    rowThreeList = []
    fillNumber =0
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        readText = data['text'][i]
        textLength = len(readText)

        if fillNumber ==0 and textLength >0:
            #found first block of text in dictionary
            fillNumber = fillNumber +1
        
        if textLength == 0 and fillNumber>0:
            #found a space after a block of text, move to next row
            fillNumber = fillNumber +1

        if textLength >0:
            #put found text in appropriate row list
            match fillNumber:
                case 1: 
                    rowOneList.append(readText)
                case 2: 
                    rowTwoList.append(readText)
                case 3: 
                    rowThreeList.append(readText)
        
    
    returnList = []
    returnList.append(rowOneList)
    returnList.append(rowTwoList)
    returnList.append(rowThreeList)
    return returnList

#for a data Table, iterate over table and examine each cell
#for each cell extarct a string and store it 
#return a storage object containing all the data - sorted
def processImagetoString(imgDataTable,imgDataMask):
  
    imageTableData = TableDataContainer()
    countoursData = cv2.findContours(imgDataMask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    for element in countoursData:
        x, y, w, h = cv2.boundingRect(element)
        isCellSizeCorect = confirmBoxDimension(w,h,MASK_CELL_SIZE_RANGE)
        startPoint = (x,y)
        endPoint = (x+w,y+h)

        if isCellSizeCorect:
            # crop to isolates cell
            cellImg = imgDataTable[startPoint[1]:endPoint[1],startPoint[0]:endPoint[0]]
            #cv2.imshow("Cell Image",cellImg)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            #OCR to read individual Cell
            cellString = processCellToString(cellImg,TESSERACT_NUMBER_CONFIG)
            #print("String Found = ",x,y,cellString)
            if len(cellString) ==0 :
                #unable to OCR the cell, ask a human
                cellString = askAHuman(cellImg)
            
            #Build a dict to hold the data & add data to container object
            cellDict = {'x':x,'y':y,'data':cellString}
            imageTableData.addData(cellDict)

    #finished capturing all the data, sort it
    imageTableData.sortData()
    return imageTableData

#for an individual ROI/Cell
#will greyscale coloured image
def processCellToString(img,configString):
    returnString ="NaN"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blackandwhite = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    if np.mean(blackandwhite) == 255:
        print("Cell Empty - Returning 0")
        returnString="0"
        return returnString
    
    processedImg = processImageAlgorithmDefault(gray)
    returnString = pytesseract.image_to_string(processedImg,config=configString)
    returnString = returnString.strip()
    return returnString

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

#Best guess at effective image processing
def processImageAlgorithmDefault(grayImg):
    #optimize character height to 30p - Tesseract quirk
    resizeCubic = cv2.resize(grayImg, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    #sharpen = unsharp_mask(resizeCubic,amount=2.0)
    #cv2.imshow("UnsharpMaks",sharpen)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #asked for help 11 times in 20 pages
    #kernel = np.ones((1, 1), np.uint8)
    #dilated = cv2.dilate(resizeCubic, kernel, iterations=1)
    #eroded = cv2.erode(dilated, kernel, iterations=1)
    #filtered = cv2.threshold(cv2.bilateralFilter(eroded, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # asked for help 10 times in 20 pages, very accurate otherwise
    #deNoised = resizeCubic.copy()
    #cv2.fastNlMeansDenoising(resizeCubic,deNoised,15.0)
    #thresh = cv2.threshold(deNoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #flipThresh = cv2.bitwise_not(thresh)

    #clean the noise and then dialate a little
    deNoised = resizeCubic.copy()
    cv2.fastNlMeansDenoising(resizeCubic,deNoised,15.0)
    thresh = cv2.threshold(deNoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    flipThresh = cv2.bitwise_not(dilated)
    return flipThresh

def askAHuman(failImg):
    cv2.imshow("Help Please",failImg)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    print("Help Please")
    returnString = input("Picture Value is ? ")
    theWorld.askforhelp()

    return returnString

def addPageDataToMem(pageData):
    theWorld.appendData(pageData)


def processPdfPage(pageImg):
    #Step 1 Trim Headers to isolate scanned image
    trimmed = trimHeaderFooter(pageImg)

    #step 2 make a Mask to find countours/bounding rectanges for Regions of interest on page
    imgMask,res = makeTableMask(trimmed)

    #find Date Box and Main Data Tabel
    contourMain = cv2.findContours(imgMask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    desiredBoxes = []
    boxesIndex =0
    dateBoxIndexList =[]
    tableBoxIndexList = []
    foundBoxes =0
    for element in contourMain:
        x, y, w, h = cv2.boundingRect(element)
        if x < X_THRESHOULD:
            #Want Boxes on left side of page
            if confirmBoxDimension(w,h,DATE_BOX_SIZE_RANGE):
                #grab the Date Box
                desiredBoxes.append(element)
                dateBoxIndexList.append(tableBoxIndex)
                foundBoxes=foundBoxes+1
        
            if confirmBoxDimension(w,h,DATA_TABLE_SIZE_RANGE):
                #grab the Data Table
                desiredBoxes.append(element)
                tableBoxIndexList.append(tableBoxIndex)
                foundBoxes=foundBoxes+1

        tableBoxIndex = boxesIndex+1
    
    if(foundBoxes != 2):
        # Fix this issues when Dealing with 2020 Data
        #Debug Code
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 4
        if foundBoxes ==0:
            print("No data found on page")
            cv2.imshow("NO Data Found", pageImg)
            cv2.imshow("Failed Mask", res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        for element in desiredBoxes:
            x, y, w, h = cv2.boundingRect(element)
            startPoint = (x,y)
            endPoint = (x+w,y+h)
            failROI = cv2.rectangle(pageImg, startPoint, endPoint, color, thickness)
            print(x,y,w,h)
            cv2.imshow("Broken",failROI)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print("Huston, We have a Problem -", foundBoxes ,"Boxes")
    
    #2017-2020 data has 2 table boxes
    dateBox = desiredBoxes[1]
    tableBox = desiredBoxes[0]

    dateROI = captureCell(trimmed,dateBox)
    date_Value = processCellToString(dateROI,TESSERACT_DATE_CONFIG)

    dataTableROI,dataTableROIMask,cleanData,cleanTabled = captureAndCleanDataTable(trimmed,tableBox)

    
    #Algorithm 1) Grab Data from the Table as a whole
  
    #cleanDataStringConfig = r'--psm 6 -c tessedit_char_whitelist=0123456789'
    #dataList = proccessImageToData(cleanData,cleanDataStringConfig)
    #addPageDataToMem(dataList,date_Value)
    #not very accurate
    
    #Algorithm 2) Grab Data from the Tabel, focusing Cell ROI by Cell ROI, <2020 version
    thisPageData = PageData()
    thisdataContainer = processImagetoString(dataTableROI,dataTableROIMask)
    thisPageData.addData(date_Value,thisdataContainer)
    addPageDataToMem(thisPageData)
    return

    
# write to file as CSV.txt
def writeOutCSVFile(dictDataArray,filepath):
    outputFile =  open(filepath,"a")
    outputFile.write(CSV_HEADEERS)
    pageIndex = 1
    for elms in dictDataArray:
        rowString = str(pageIndex) +", " + elms.toCSVstring()
        outputFile.write(rowString)
        pageIndex = pageIndex+1

    outputFile.close()
###################################################################################
# MAIN SCRIPT
print('pdf2csv')
filepath = ""
path2017FileRead = "2022-G-203 (2017).pdf"
path2017FileWrite = "2017DataCSV.txt"
#TODO - add argument parsing
filepath = path2017FileRead
outputFilePath = path2017FileWrite
pdf = fitz.open(filepath)
pageCount = pdf.page_count
print('Document Pages -',pageCount)


#debug value
#iDicts = 1 #start processing at specific page
pageCount = 20 # only process X amount of records
startTime = time.time()
for iDicts in range(pageCount):
#if iDicts:
    image_List = pdf.get_page_images(iDicts)
    numImageOnPage = len(image_List)
    pageDictonary =  buildEmptyResultDic()
    if(numImageOnPage ==1):
        #get the Image that represents the Page
        img = image_List[0]
        xref = img[0]
        pix = fitz.Pixmap(pdf,xref)
        pageImageCV2 = pix2np(pix)
        #Process Image and extact text
        #try:
        processPdfPage(pageImageCV2)
        #except Exception as e:
        #    print("My Code explode!")
        #    print(e)
        print("completed page", iDicts)
    # percent = iDicts+1
    # percent = round((percent/pageCount)*100,2)
        #print("Prcessing - ", percent,"% (", iDicts," of ",pageCount, " )",end='\r')
endTime = time.time()
elapsedTime = endTime-startTime
delta = timedelta(seconds=elapsedTime)

pageDataArray = theWorld.getData()
writeOutCSVFile(pageDataArray,outputFilePath)
helped = theWorld.howMuchHelpAsked()
print("Completed in, ", delta," Helps Asked = ",helped)