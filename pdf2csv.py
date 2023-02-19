#imports
import fitz
import numpy as np
import math
import cv2
import pytesseract
from pytesseract import Output

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
MIN_CELL_WIDTH = 15
ROW_ONE_Y_MIN = 10
ROW_THREE_Y_MIN = 50 
CROP_PIXEL = (2,4)
TABLE_SHIFT_START = (104,38)
TABLE_SHIFT_END = (568,155)

#Global

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

def sortOnX(elm):
    x = cv2.boundingRect(elm)[0]
    return x


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

#Return a Mask that can be used to find Tables
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
    horizontal_size = math.ceil(cols / 20)

    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = math.ceil(rows / 70)

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    res = vertical + horizontal

    return res, image

def confirmBoxDimension(w,h,sizeRange):
    isDateBox =False
    if w>sizeRange[0] and h>sizeRange[1] and w<sizeRange[2] and h<sizeRange[3]:
        isDateBox = True
    
    return isDateBox

def captureCell(img, areaBox):
    x, y, w, h = cv2.boundingRect(areaBox)
    startPoint = (x+CROP_PIXEL[0],y+CROP_PIXEL[0])
    endPoint = (x+w-CROP_PIXEL[1],y+h-CROP_PIXEL[1])
    areaROI = img[startPoint[1]:endPoint[1],startPoint[0]:endPoint[0]]
    return areaROI

def captureDataTable(img,mask,areBox):
    x, y, w, h = cv2.boundingRect(areBox)
    startPoint = (x+TABLE_SHIFT_START[0],y+TABLE_SHIFT_START[1])
    endPoint = (x+TABLE_SHIFT_END[0],y+TABLE_SHIFT_END[1])

    tableROI = img[startPoint[1]:endPoint[1],startPoint[0]:endPoint[0]]
    maskROI = mask[startPoint[1]:endPoint[1],startPoint[0]:endPoint[0]]
    return tableROI, maskROI

def processImageToString(img,configString):
    returnString ="NaN"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blackandwhite = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    if np.mean(blackandwhite) == 255:
        print("Cell Empty - Returning 0")
        returnString="0"
        return returnString
    
    returnString = processImageAlgorithmDefault(gray,configString)
    return returnString

#Best guess at effective image processing
def processImageAlgorithmDefault(grayImg,configString):
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(grayImg, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    filtered = cv2.threshold(cv2.bilateralFilter(eroded, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    returnString = pytesseract.image_to_string(filtered,config=configString)
    returnString = returnString.strip()
    return returnString

def askAHuman(failImg):
    cv2.imshow("Help Please",failImg)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    print("Help Please")
    returnString = input("Picture Value is ? ")

    return returnString

def processPdfPage(pageImg):
    #Debug Code
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 4

    trimmed = trimHeaderFooter(pageImg)

    res,pageImg = makeTableMask(trimmed)

    #find Date Box and Main Data Tabel
    contourMain = cv2.findContours(res, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
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
        if foundBoxes ==0:
            cv2.imshow("Skew?", pageImg)
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
    
    dateBox = desiredBoxes[1]
    tableBox = desiredBoxes[0]

    dateROI = captureCell(pageImg,dateBox)
    date_Value = processImageToString(dateROI,TESSERACT_DATE_CONFIG)

    dataTableROI,dataTableROIMask = captureDataTable(pageImg,res,tableBox)

    #Grab Data from the Tabel, <2020 version
    dataBoxMetro = []
    dataBoxIft = []
    dataBoxSubR = []
    countoursData = cv2.findContours(dataTableROIMask, cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)[0]
    for element in countoursData:
        x, y, w, h = cv2.boundingRect(element)
        if confirmBoxDimension(w,h,DATA_CELL_SIZE_RANGE):
            if(y<ROW_ONE_Y_MIN):
                dataBoxMetro.append(element)
            if(y> ROW_ONE_Y_MIN and y<ROW_THREE_Y_MIN):
                dataBoxIft.append(element)
            if y>ROW_THREE_Y_MIN:
              dataBoxSubR.append(element)  

    #sort the output
    dataOutput = []
    dataBoxMetro.sort(key=sortOnX)
    dataOutput = dataOutput+dataBoxMetro
    dataBoxIft.sort(key=sortOnX)
    dataOutput = dataOutput+dataBoxIft
    dataBoxSubR.sort(key=sortOnX)
    dataOutput = dataOutput+dataBoxSubR    

    #Process and Store Page's Data
    metroDict = build3PartDict()
    iftDict = build4PartDict()
    subRDict = build3PartDict()
    tableIndex =0
    for element in dataOutput:
        cell = captureCell(dataTableROI,element)
        cellString = processImageToString(cell,TESSERACT_NUMBER_CONFIG)
        if len(cellString) ==0 :
            cellString = askAHuman(cell)
        match tableIndex:
            case 0:
                metroDict[SCHEDULED]=cellString
            case 1:
                metroDict[STAFFED] = cellString
            case 2:
                metroDict[BLS] = cellString
            case 3:
                iftDict[SCHEDULED] = cellString
            case 4:
                iftDict[STAFFED] = cellString
            case 5:
                iftDict[BLS] = cellString
            case 6:
                iftDict[TANGO] = cellString
            case 7:
                subRDict[SCHEDULED] = cellString
            case 8:
                subRDict[STAFFED] = cellString
            case 9:
                subRDict[BLS] = cellString
        tableIndex = tableIndex+1
    thisPageDict = buildResultDict(date_Value,metroDict,iftDict,subRDict)
    print("this Page",thisPageDict)
    return thisPageDict
    
# write to file as CSV.txt
def writeOutCSVFile(dictDataArray,filepath):
    iRecords = len(dictDataArray)
    outputFile =  open(filepath,"a")
    outputFile.write(CSV_HEADEERS)

    for index in range(iRecords):
        #add Date
        row = []
        row.append(str(index+1))
        row.append(",")
        baseDictonary = dictDataArray[index]
        row.append(baseDictonary.get(DATE))
        row.append(",")
        
        #Add Metro Stats
        metroDict = baseDictonary.get(METRO_DICT)
        row.append(metroDict.get(SCHEDULED))
        row.append(",")
        row.append(metroDict.get(STAFFED))
        row.append(",")
        row.append(metroDict.get(BLS))
        row.append(",")

        #add Suburban / Rural Stats
        subRDict = baseDictonary.get(SUBR_DICT)
        row.append(subRDict.get(SCHEDULED))
        row.append(",")
        row.append(subRDict.get(STAFFED))
        row.append(",")
        row.append(subRDict.get(BLS))
        row.append(",")

        #add IFT
        iftDict = baseDictonary.get(IFT_DICT)
        row.append(iftDict.get(SCHEDULED))
        row.append(",")
        row.append(iftDict.get(STAFFED))
        row.append(",")
        row.append(iftDict.get(BLS))
        row.append(",")
        row.append(iftDict.get(TANGO))
        row.append("\n")

        #put it all together
        outputString = "".join(row)
        outputFile.write(outputString)

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

dictList =[]
#debug value
#iDicts =
pageCount = 1
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
        try:
            pageDictonary = processPdfPage(pageImageCV2)
        except Exception as e:
            print("My Code explode!")
            print(e)

        dictList.append(pageDictonary)
        print("completed", iDicts)
    # percent = iDicts+1
    # percent = round((percent/pageCount)*100,2)
        #print("Prcessing - ", percent,"% (", iDicts," of ",pageCount, " )",end='\r')

writeOutCSVFile(dictList,outputFilePath)
print("Completed")