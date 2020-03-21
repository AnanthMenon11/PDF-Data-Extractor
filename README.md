# Types of PDF files used
During the building of the POC, we realised that Character recognition algorithms react differently to different kinds of PDF 
files. So we tested the above process flow on 2 different kinds of PDF Documents.
# The 2 PDF documents are as follows:</b>
&emsp;&emsp;<b>1) Tabular_PDF- </b><br>
&emsp;&emsp;&emsp;Predominantly Tabular Data<br>
&emsp;&emsp;&emsp;Contains Text on a few pages<br>
&emsp;&emsp;&emsp;Tables continue across pages<br>
<br>
&emsp;&emsp;<b>2) Textual_PDF-</b><br>
&emsp;&emsp;&emsp;Predominantly Text Data<br>
&emsp;&emsp;&emsp;Contains Highlighed Text on a few pages<br>
&emsp;&emsp;&emsp;Contains a few tables<br>


# Function Definitions and Process Flow
Apart from pandas and numpy, we are using a few python packages like Tesseract, Camelot, PIL & PDF2Image.These package are being utilized by various functions in the following order-<br>
# Step 1:
Call function <b>extractData_PDF()</b> by passing the address of the PDF file which needs to be extracted.This function is used to extract pages from the PDF and to create a list of dataframes from the PDF.
# Step 2:
Once the tables are generated, function <b>get_Order_Page()</b> is called with 2 parameters. The list of extracted tables (DataFrames) and the path of the folder to which the tables need to be exported. This function filters the tables, creates/returns a dictionary (Page Number:Number of Tables) and exports the tables with data to the <b>"ExcelFiles"</b>.
# Step 3:
The next step in the pipeline is traversal of each page of the PDF and extracting text. If a table exists on the page then page is first sent through 2 functions. Namely, <b>find_tables()</b> & <b>blacken_tables()</b>. These functions as their names suggest find the tables on the page (<b>find_tables()</b>) and then proceed with blackening them out (<b>blacken_tables()</b>) so that text from the tables doesn't reappear.
# Step 4:
The page is then sent to the function <b>extract_text()</b> which uses Tesseract to decode the text on the page. <b>extract_text()</b> calls a function called <b>sort_contours()</b> to order the contours found on the image so that the textual output is in readable format. <b>extract_text()</b> returns a string with all the text from that page appended to it.
# Step 5:
The returned string from function <b>extract_text()</b> is sent to function <b>save_Txt()</b> which inturn saves the text per 
page in the <b>"TextFiles"</b> folder.
