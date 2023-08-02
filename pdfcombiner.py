import fitz

def merge2022Part001():
    # First Part of 2022
    file001 = "2022-01.pdf"
    file002 = "2022-02 to 04 - part 1.pdf"
    file003 = "2022-02 to 04 - part 2.pdf"
    file004 = "2022-05 to 08 - part 1.pdf"
    file005 = "2022-05 to 08 - part 2.pdf"
    file006 = "2022-05 to 08 - part 3.pdf"
    fileArray = [file001,file002,file003,file004,file005,file006]
    outputString = "2022-01-08.pdf"
    return fileArray,outputString

def merge2022Part002():
    #Second Part of 2022
    file007 = "2022 - Sept FOIP 2023-G-069.pdf"
    file008 = "2022 - Oct FOIP 2023-G-069.pdf"
    file009 = "2022 - Nov FOIP 2023-G-069.pdf"
    file010 = "2022 - Dec FOIP 2023-G-069.pdf"
    fileArray = [file007,file008,file009,file010]
    outputString = "2022-09-12.pdf"
    return fileArray,outputString

def merge2023Part001():
    print("Merging Jan-April 2023")
    file202301 = "2023-G-127 - 001.pdf"
    file202302 = "2023-G-127 - 002.pdf"
    file202303 = "2023-G-127 - 003.pdf"
    file202304 = "2023-G-127 - 004.pdf"
    fileArray = [file202301,file202302,file202303,file202304]
    outputString = "2023-04-30.pdf"
    return fileArray,outputString

print("Merge some PDFs")

fileArray,outputPath = merge2023Part001()

results = fitz.open()

for pdf in fileArray:
    with fitz.open(pdf) as mfile:
        results.insert_pdf(mfile)
    
results.save(outputPath)