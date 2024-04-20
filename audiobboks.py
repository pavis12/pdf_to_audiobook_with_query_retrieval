import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from pdfminer.high_level import extract_text
from gtts import gTTS
import speech_recognition as sr
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import time
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from googletrans import Translator
pytesseract.pytesseract.tesseract_cmd=r'C:\Users\Sathish\AppData\Local\Programs\Python\Python312\Lib\site-packages\tesseract.exe'
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAHFvDRL0tNbP9yX0WjCeXsuIci8ueGKQg'


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        print(pdf,end="\n")
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def play(fn,dir):
    #for fn in fcn.keys():
        # predict images
    filename = get_filename_without_extension(fn)
    path = dir+ "/"+fn
    print(path,fn,os.path.exists(path),AUDIO_DIR + "/"+filename+".mp3")
    if os.path.exists(AUDIO_DIR + "/"+filename+".mp3"):
       st.audio(AUDIO_DIR + "/"+filename+".mp3" ,format="audio/mp3")
    else:
        filenames = pdf2Image(path)
            #filenames=convert_from_path(path)
        outFile = convertImagesToText(filenames,filename)
        text = open(outFile).read()
            #text=get_pdf_text(text)
        try:
            with st.spinner('Bringing it to Life....'):
                #time.sleep(5)
                #st.success('Done!')
                ta_tts = gTTS(text)
                ta_tts.save(AUDIO_DIR +"/"+ filename + '.mp3')
                    #audio = AudioSegment.from_file("temp_audio_file")
                    #time.sleep(5)
                    # Play the audio
                    #st.audio(audio)
                st.audio(AUDIO_DIR + "/"+filename+".mp3" ,format="audio/mp3")
        except:
            st.write("Server is Busy!Sorry for the interruption :)")
            st.write("Try after sometime.")

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_filename_without_extension(file_path):
    file_basename = os.path.basename(file_path)
    filename_without_extension = file_basename.split('.')[0]
    return filename_without_extension

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("/content/faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def createDir(dirName):
  if not os.path.exists(dirName):

    os.makedirs(dirName)
    print("Directory " , dirName ,  " Created ")

  else:
    print("Directory " , dirName ,  " already exists")


def percentage(curr,total):
  return curr/total * 100


def pdf2Image(fn):
  file = get_filename_without_extension(fn)
  #image_fn_dir= IMAGES_DIR + file
  image_fn_dir= file
  createDir(image_fn_dir)
  print(fn)
  pages = convert_from_path(fn, 500)
  print(pages)
  filenames = []

  # Counter to store images of each page of PDF to image
  image_counter = 1
  total = len(pages)

  # Iterate through all the pages stored above
  for page in pages:

      # Declaring filename for each page of PDF as JPG
      # For each page, filename will be:
      # PDF page 1 -> page_1.jpg
      # PDF page 2 -> page_2.jpg
      # PDF page 3 -> page_3.jpg
      # ....
      # PDF page n -> page_n.jpg
      filename = image_fn_dir+ "/page_"+str(image_counter)+".jpg"
      filenames.append(filename)
      # Save the image of the page in system
      page.save(filename, 'JPEG')

      print("Conversion - "+str(percentage(image_counter,total)) +" % work done yay!")

      # Increment the counter to update filename
      image_counter = image_counter + 1

  return filenames
translator = Translator()
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("/content/faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    translation = translator.translate(response["output_text"], src="en", dest="hi")
    trans_text = translation.text
    if st.button("English"):
        ta_tts = gTTS(response["output_text"])
        ta_tts.save(AUDIO_DIR +"/"+ "REPLY" + '.mp3')
        st.audio(AUDIO_DIR + "/"+"REPLY"+".mp3" ,format="audio/mp3")
        st.write("Reply: ", response["output_text"])
    if st.button("Hindi"):
        ta_tts = gTTS(trans_text)
        ta_tts.save(AUDIO_DIR +"/"+ "REPLY" + '.mp3')
        st.audio(AUDIO_DIR + "/"+"REPLY"+".mp3" ,format="audio/mp3")
        st.write(trans_text)
    #st.write("Reply: ", response["output_text"])


IMAGES_DIR = "Images"
AUDIO_DIR = "Audio"
createDir(IMAGES_DIR)
createDir(AUDIO_DIR)

def convertImagesToText(filenames, ofilename):
  # Creating a text file to write the output
  outfile = "out_text_"+ofilename+".txt"

  # Open the file in append mode so that
  # All contents of all images are added to the same file
  f = open(outfile, "w")
  for file in filenames:
    print(file)
    text = str(((pytesseract.image_to_string(Image.open(file)))))
    text = text.replace('-\n', '')
    text = text + '\n'
    f.write(text)
  f.close()
  print(outfile + " prepared :) ")
  return outfile

import shutil

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        print(len(pdf_docs))
        print(pdf_docs)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
        text=get_pdf_text(pdf_docs)
        upl={}
        c=0
        print("---------------\n\n\n")
        for i in pdf_docs:
            destination_dir="files"
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)
            with open(os.path.join(destination_dir, i.name), "wb") as f:
                f.write(i.getbuffer())
            destination_file = os.path.join(destination_dir, i.name)
            print(destination_file)
            #shutil.copy(i.name, destination_file)
            upl[i.name]=text
            #print("Main      ",c,i.name) 
            c+=1
            play(i.name,destination_dir)
        if st.button("Exit"):
            shutil.rmtree(AUDIO_DIR)
            shutil.rmtree(IMAGES_DIR)


if __name__ == "__main__":
    main()