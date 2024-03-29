#imports
import fitz
import numpy as np
import math
import cv2
import pytesseract
from pytesseract import Output
import time
from datetime import timedelta
import pandas as pd

# Path to the location of the Tesseract-OCR executable/cmd
pytesseract.pytesseract.tesseract_cmd =r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#CONSTANTS
DATE ='Date'
METRO = 'Metro'
IFT = 'IFT'
SUBURBAN ='Suburban'
SCHEDULED = 'Scheduled'
STAFFED = 'Staffed'
BLS ='BLS'
TANGO = 'Tango'
TESSERACT_DATE_CONFIG = r'--psm 6'
TESSERACT_NUMBER_CONFIG = r'--psm 7 -c tessedit_char_whitelist=0123456789'
CSV_HEADEERS = "Page, Date, Metro Scheduled, Metro Staffed, Metro BLS, IFT Scheduled, IFT Staffed, IFT BLS, IFT Tango,Sub/Rural Scehduled, Sub/Rural Staffed, Sub/Rural BLS\n"


#MAGIC_NUMBERS
CELL_SAMPLE_WIDTH = 50 
CELL_SAMPLE_HEIGHT = 20
CROP_PIXEL = (10,6,6,10) #Left,Top,bottom,right
TRIM_HEIGHT = 120
HORIZONTAL_SPECIFICITY = 20
VERTICAL_SPECIFICITY_FIND_TABLE = 20
VERTICAL_SPECIFICITY_FIND_CELL = 10

#Objects
def sortOnX(elm):
    x = elm['x']
    return x
def sortAreaOnX(elm):
    x =cv2.boundingRect(elm)[0]
    return x
def sortAreaOnY(elm):
    y =cv2.boundingRect(elm)[1]
    return y

class TableDataContainer:
    def __init__(self):
        self.Datalist = []
        self.dataFrame = None

    
    def addTableData(self,cellDict):
        self.Datalist.append(cellDict)

    def sortTableData(self):
        seenX = set()
        seenY = set()
        for element in self.Datalist:
           seenX.add(element[0])
           seenY.add(element[1])
        
        cols = list(seenX)
        cols.sort()
        rows = list(seenY)
        rows.sort()
        emptyDF = pd.DataFrame(0,rows,cols)     
        for element in self.Datalist:
            emptyDF.at[element[1],element[0]] = element[2]

        #place where version of file matters
        pageDataVersion = theWorld.getDataVersion()
        emptyDF.rename(columns={cols[0]:SCHEDULED, cols[1]:STAFFED, cols[2]:BLS, cols[3]:TANGO},inplace=True)
        match pageDataVersion:
            case 0 :
                emptyDF.rename(index = {rows[0]:METRO, rows[1]:IFT, rows[2]:SUBURBAN},inplace=True)
            case 1:
                #back part of 2020 and the future 
                emptyDF.rename(index = {rows[0]:METRO, rows[1]:SUBURBAN, rows[2]:IFT},inplace=True)
            
        self.dataFrame = emptyDF

    def toCSVstring(self):
        csvString = ""
        #Metro / IFT / Suburban-Rural
        metroScheduledString = self.dataFrame.at[METRO,SCHEDULED]
        metroStaffedString = self.dataFrame.at[METRO,STAFFED]
        metroBLSString = self.dataFrame.at[METRO,BLS]
        metroString = metroScheduledString + ", " + metroStaffedString + ", " + metroBLSString + ", "
        
        iftScheduledString = self.dataFrame.at[IFT,SCHEDULED]
        iftStaffedString = self.dataFrame.at[IFT,STAFFED]
        iftBLSString = self.dataFrame.at[IFT,BLS]
        iftTangoString = self.dataFrame.at[IFT,TANGO]
        iftString = iftScheduledString + ", " + iftStaffedString + ", " + iftBLSString + ", " + iftTangoString + ", "

        subrScheduleString = self.dataFrame.at[SUBURBAN,SCHEDULED]
        subrStaffedString = self.dataFrame.at[SUBURBAN,STAFFED]
        subrBLSString = self.dataFrame.at[SUBURBAN,BLS]
        subrString = subrScheduleString + ", " + subrStaffedString + ", " + subrBLSString

        csvString = metroString + iftString + subrString + "\n"

        return csvString
    
    def fillBrokenContainer(self):
        cols = [SCHEDULED,STAFFED,BLS,TANGO]
        rows = [METRO,IFT,SUBURBAN]
        self.dataFrame = pd.DataFrame("Nan",rows,cols)
        
        
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
        self.crashed =0
        self.pageDataArray =[]
        self.dataVersion =0 # default for 2017/2018/2019/ half of 2020
        self.versionSwapIndex =-1
        self.willSwap = False
        self.swapped2020 = False
    
    def defineWhereSwap(self,swapIndex):
        self.willSwap = True
        self.versionSwapIndex = swapIndex
        
    def swapDataVersion(self, pageIndex):
        #deal with the formating change in 2020 file
        if self.willSwap:
            if pageIndex >= self.versionSwapIndex:
                self.swapped2020 = True
                print("The World Swapped to Data Version =",self.dataVersion)
                self.willSwap = False # no more swaps until told to swap again
    
    def setDataVersion(self, version):
        #Set Version for 2017-202x
        self.dataVersion = version
    

    def getDataVersion(self):
        currentdataVersion = 0 # the Default 2017-2019

        if self.dataVersion > 2020: #first part of 2020
            currentdataVersion = 1
        if self.dataVersion == 2020 and self.swapped2020: #latter part of 2020
            currentdataVersion = 1

        return currentdataVersion

    def askforhelp(self):
        self.asked = self.asked +1

    def howMuchHelpAsked(self):
        helped = self.asked
        return helped
    
    def exceptionDetected(self):
        self.crashed = self.crashed+1

    def howManyExceptionsDetected(self):
        return self.crashed
    
    def appendData(self,pageData):
        self.pageDataArray.append(pageData)
    
    def getData(self):
        return self.pageDataArray 

theWorld = StateAndMem()

#Converts Pix to np
def pix2np(pix):
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    return im

def detectAndcorrectSkew(tableContour,skewedPage):
    
    #Testing the the squarness of the Table
    rect = cv2.minAreaRect(tableContour)
    needsCorrectedAngle = False
    rotated = None
    rectAngle = rect[2]
    if rectAngle != 0 and rectAngle !=90:
        needsCorrectedAngle = True
        # rotate the image to deskew it
        ( h, w) = skewedPage.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rectAngle, 1.0)
        rotated = cv2.warpAffine(skewedPage, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return needsCorrectedAngle, rotated

#Return Masks that can be used to find Tables
def makeTableMask(image,verticalGravity):
    #greyscale and thresh for cvOpen operations
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(thresh)
    vertical = np.copy(thresh)
    
    cols = horizontal.shape[1]
    horizontal_size = math.ceil(cols / HORIZONTAL_SPECIFICITY)

    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = math.ceil(rows / verticalGravity)

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    res = vertical + horizontal

    return res, thresh

#All rows are the Height, just need to know how many are in table
def countTableRows(tableMesh):
    rowsArray = []
    hImg = tableMesh.shape[0]
    startPoint =(0,0)
    endPoint = (CELL_SAMPLE_WIDTH,hImg)
    whiteBoarder = (255,255,255)
    thickness = 10
    firstColumnOnlyimg = np.copy(tableMesh[startPoint[1]:endPoint[1],startPoint[0]:endPoint[0]])
    cv2.rectangle(firstColumnOnlyimg,startPoint,endPoint,whiteBoarder,thickness)

    rowContours = cv2.findContours(firstColumnOnlyimg, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
    for elements in rowContours:
        x,y,w,h = cv2.boundingRect(elements)
        if h<hImg:
            rowsArray.append(elements)


    rowsArray.sort(key=sortAreaOnY)
    return rowsArray

def verifyDesiredCell(contourElement,wMesh):
    wantCell = False
    x,y,w,h = cv2.boundingRect(contourElement)
    midpoint = 0
    fileDataVersion = theWorld.getDataVersion()
    match fileDataVersion:
        case 0:
            midpoint = math.ceil(wMesh/2)
        case 1:
            midpoint = math.ceil(wMesh/2) -100

    if x< midpoint and w < wMesh:
        wantCell = True


    return wantCell



#each column has varying lenght, capture the area for each relevent column
def countTableColumns(tableMesh):
    colsArray = []
    wMesh = tableMesh.shape[1]
    #Isolate just the top row of the teable
    startPoint =(0,0)
    endPoint = (wMesh,CELL_SAMPLE_HEIGHT)
    whiteBoarder = (255,255,255)
    thickness = 5
    topRowOnlyimg = np.copy(tableMesh[startPoint[1]:endPoint[1],startPoint[0]:endPoint[0]])
    cv2.rectangle(topRowOnlyimg,startPoint,endPoint,whiteBoarder,thickness)

    columnContours = cv2.findContours(topRowOnlyimg, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]

    for element in columnContours:

        isCellwanted = verifyDesiredCell(element,wMesh)
        if isCellwanted:
            colsArray.append(element)

    if len(colsArray) !=5:
        raise IndexError("Count Cols - mask failed")

    colsArray.sort(key=sortAreaOnX)
    return colsArray

#Isolate and pre-Process the ROI representating the data table
# @img = Master Image representing the whole page
# @area box - provides dimensions of ROI
# Returns ROI of Table and Mask to isolate each cell
def captureAndCleanDataTable(pageImage,areaBox):
    #1) Crop working aread down to Table Range of Interest
    x, y, w, h = cv2.boundingRect(areaBox)
    startPoint = (x,y)
    endPoint = (x +w,y+h)
    tableROI = pageImage[startPoint[1]:endPoint[1],startPoint[0]:endPoint[0]]

    #creat a new mask for cropped image - Masking out cells of the Table
    maskROI = makeTableMask(tableROI,VERTICAL_SPECIFICITY_FIND_CELL)[0]

    #calcualate placement of rows and columns
    rows = countTableRows(maskROI)
    cols = countTableColumns(maskROI)

    #Make Mask background all white
    imgH, imgW = maskROI.shape
    startPoint = (0,0)
    endPoint = (imgW,imgH)
    cornerDownLeft = startPoint[0],endPoint[1]
    cornerUpRight = endPoint[0],startPoint[1]
    contoursFillWhite = np.array([startPoint, cornerDownLeft, endPoint, cornerUpRight])
    cv2.fillPoly(maskROI, pts =[contoursFillWhite], color=(255,255,255))

    #place black cells on white background
    colIndex = 1 #skip row headers, only interested in next 4 columns 
    while colIndex <5:
        rowIndex = 1 #skip column headers
        colX, colY, colW, colH = cv2.boundingRect(cols[colIndex])
        while rowIndex < 4:
            #only interested in first 4 rows after headers
            rowX,rowY,rowW,rowH = cv2.boundingRect(rows[rowIndex])
            cornerUpLeft = colX+CROP_PIXEL[0],rowY+CROP_PIXEL[1]
            cornerDownLeft = colX+CROP_PIXEL[0],rowY+rowH-CROP_PIXEL[2]
            cornerDownRight = colX+colW-CROP_PIXEL[3],rowY+rowH-CROP_PIXEL[2]
            cornerUpRight = colX+colW-CROP_PIXEL[3],rowY+CROP_PIXEL[1]
            contoursFillBlack = np.array([cornerUpLeft, cornerDownLeft, cornerDownRight, cornerUpRight])
            cv2.fillPoly(maskROI, pts =[contoursFillBlack], color=(0,0,0))
            rowIndex = rowIndex+1 #increment
        colIndex = colIndex+1

    #create conntours works with black background / white objects
    maskROI = cv2.bitwise_not(maskROI)
    return tableROI,maskROI

# for a sanitized Data Table
# Pro - might be faster
# Con - unable to ask a human for help
def proccessImageToData(img,configString):
    #Pass in clean image / is already black & white
    processedImg = processImageAlgorithmDefault(img)
    data = pytesseract.image_to_data(processedImg,config=configString,output_type=Output.DICT)
    return data 

#for a data Table, iterate over table and examine each cell
#for each cell extarct a string and store it 
#return a storage object containing all the data - sorted
def processImagetoString(imgDataTable,imgDataMask):
    imageTableData = TableDataContainer()
    countoursData = cv2.findContours(imgDataMask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    for element in countoursData:
        x, y, w, h = cv2.boundingRect(element)
        startPoint = (x,y)
        endPoint = (x+w,y+h)

        # crop to isolates cell
        cellImg = imgDataTable[startPoint[1]:endPoint[1],startPoint[0]:endPoint[0]]

        #OCR to read individual Cell
        cellString, testedImg = processCellToString(cellImg,TESSERACT_NUMBER_CONFIG)
        if len(cellString) ==0 :
            #unable to OCR the cell, ask a human
            cellString = askAHuman(testedImg)
            
        #Build a tuple to hold the data & add data to container object
        cellTuple = (x,y,cellString)
        imageTableData.addTableData(cellTuple)

    #finished capturing all the data, sort it
    imageTableData.sortTableData()
    return imageTableData

#for an individual ROI/Cell
#will greyscale coloured image
def processCellToString(img,configString):
    returnString ="NaN"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Clean up any possible boarder smuges
    h,w =gray.shape
    startpoint = (0,0)
    endpoint = (w,h)
    whiteBoarder = (255,255,255)
    thickness = 4
    cv2.rectangle(gray,startpoint,endpoint,whiteBoarder,thickness)
    blackandwhite = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    #if blank white cell - return 0
    if np.mean(blackandwhite) == 255:
        returnString = "0"
        return returnString,blackandwhite
    
    processedImg = processImageAlgorithmDefault(gray)
    returnString = pytesseract.image_to_string(processedImg,config=configString)
    returnString = returnString.strip()
    return returnString, processedImg

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

    #clean the noise and then dialate a little
    deNoised = resizeCubic.copy()
    cv2.fastNlMeansDenoising(resizeCubic,deNoised,15.0)
    thresh = cv2.threshold(deNoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    flipThresh = cv2.bitwise_not(dilated)
    return flipThresh

def askAHuman(failImg):
    #failed on OCR - Ask a Human
    cv2.imshow("Help Please",failImg)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    print("Help Please")
    returnString = input("Picture Value is ? ")
    theWorld.askforhelp()

    return returnString

def addPageDataToMem(pageData):
    theWorld.appendData(pageData)

def captureAndCleanDate(pageImg,tableElement):
    date_String = "01/01/1901"
    x, y, w, h = cv2.boundingRect(tableElement)
    targetWidth = math.ceil(w/2)
    targetHeight = 0
    if theWorld.dataVersion ==0:
        targetHeight = h
    else:
        targetHeight = math.ceil(h/2) 

    startPoint = (x,y-targetHeight)
    endPoint = (targetWidth,y)
    dateRegion = pageImg[startPoint[1]:endPoint[1],startPoint[0]:endPoint[0]]

    gray = cv2.cvtColor(dateRegion, cv2.COLOR_BGR2GRAY)

    blackandwhite = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    imageDataDict = proccessImageToData(blackandwhite,TESSERACT_DATE_CONFIG)
    n_boxes = len(imageDataDict['text'])
    foundDateKey = False
    for i in range(n_boxes):
        if int(imageDataDict['conf'][i]) > 60: #% confidence in identification
            testword = imageDataDict['text'][i]
            if foundDateKey:
                #found date keyword previous iteration, this iteration pull the date
                date_String = testword
                break
            if "Date" in testword:
                foundDateKey = True
    return date_String

def findTableDimensions(pageMask):
    fountElement = False
    pageContours = cv2.findContours(pageMask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    widestW = 0
    widestElement = None
    for element in pageContours:
        x, y, w, h = cv2.boundingRect(element)
         #Find the widest table contour on the page
        if w > widestW:
            widestW = w
            widestElement = element
            fountElement = True
 
    #return true/false, and the widest element found
    return fountElement, widestElement

def processPdfPage(pageImg):
    thisPageData = PageData()
    thisdataContainer = None
    date_Value = "01/01/1901"
    try:
        #step 1 create a Mask to find countour rectanges for possible Regions of interest on page
        pageMask = makeTableMask(pageImg,VERTICAL_SPECIFICITY_FIND_TABLE)[0]

        #step 2 find the Region of the Table using the mask
        didFindTable,tableElement = findTableDimensions(pageMask)
        if didFindTable == False :
            print("Found no Contures on page")
            raise Exception("No Contours","Empty Page")
        
        #Check for Skew - skew most detectable when inspecting the table
        isSkewed,roatated =  detectAndcorrectSkew(tableElement,pageImg)

        if isSkewed:
            #fixed skew so lets start all over
            pageImg = roatated
            newpageMask = makeTableMask(pageImg,VERTICAL_SPECIFICITY_FIND_TABLE)[0]
            didFindTable,tableElement = findTableDimensions(newpageMask)
            if didFindTable == False :
                print("Lost Cotures when correcting skew")
                raise Exception("No Contours", "All Skewed up")

        #step 5 create a mask which allows the isolation of individual cells
        dataTableROI,dataTableROIMask = captureAndCleanDataTable(pageImg,tableElement)

        #step 6 date is located just above table box
        date_Value = captureAndCleanDate(pageImg,tableElement)
        thisdataContainer = processImagetoString(dataTableROI,dataTableROIMask)
    except Exception as inst:
        #create an empty page to add
        print("Caught Exception processing Page")
        theWorld.exceptionDetected()
        print(type(inst))    # the exception instance
        print(inst.args)     # arguments stored in .args
        thisdataContainer = TableDataContainer()
        thisdataContainer.fillBrokenContainer()


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

#
def processData(readPath):
    print("Processing Data")
    pdf = fitz.open(readPath)
    pageCount = pdf.page_count
    print('Document Pages -',pageCount)
    #debug values
    #iDicts = 200#start processing at specific page (0 for begining of file)
    #pageCount = iDicts +25 # only process X amount of records


    for iDicts in range(pageCount): # prferend loop method
    #while iDicts< pageCount: # Debug while loop
    #if iDicts:
        theWorld.swapDataVersion(iDicts)
        
        image_List = pdf.get_page_images(iDicts)
        numImageOnPage = len(image_List)
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
            #iDicts = iDicts+1 #While debug code

###################################################################################
# MAIN SCRIPT
print('pdf2csv')
readfilepath = ""
writeFilePath = "default.txt"

command = "2023-1"
match command:
    case "2017":
        readfilepath = "2022-G-203 (2017).pdf"
        writeFilePath = "2017DataCSV.txt"
        theWorld.setDataVersion(2017)
    case "2018":
        readfilepath = "2022-G-203 (2018).pdf"
        writeFilePath = "2018DataCSV.txt"
        theWorld.setDataVersion(2018)
    case "2019":
        readfilepath = "2022-G-203 (2019).pdf"
        writeFilePath = "2019DataCSV.txt"
        theWorld.setDataVersion(2019)
    case "2020":
        readfilepath = "2022-G-203 (2020).pdf"
        writeFilePath = "2020DataCSV.txt"
        theWorld.setDataVersion(2020)
        theWorld.defineWhereSwap(195)
    case "2021":
        readfilepath = "2022-G-203 (2021).pdf"
        writeFilePath = "2021DataCSV.txt"
        theWorld.setDataVersion(2021)
    case "2022-1":
        readfilepath = "2022-01-08.pdf"
        writeFilePath = "2022DataCSV-01.txt"
        theWorld.setDataVersion(2022)
    case "2022-2":
        readfilepath = "2022-09-12.pdf"
        writeFilePath = "2022DataCSV-02.txt"
        theWorld.setDataVersion(2022)
    case "2023-1":
        readfilepath = "2023-04-30.pdf"
        writeFilePath = "2023DataCSV-01.txt"
        theWorld.setDataVersion(2022)

if len(readfilepath) ==0:
    print("invalid data path")
    exit(0)

startTime = time.time()

#do some work
processData(readfilepath)

endTime = time.time()
elapsedTime = endTime-startTime
delta = timedelta(seconds=elapsedTime)

pageDataArray = theWorld.getData()
writeOutCSVFile(pageDataArray,writeFilePath)
helped = theWorld.howMuchHelpAsked()
crashes = theWorld.howManyExceptionsDetected()
print("Completed in, ", delta," Helps Asked = ",helped)
print("Exceptions caught =",crashes)