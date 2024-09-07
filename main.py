import logging
import random
import string
from io import BytesIO

import PyPDF3
import requests
import spacy
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

logging.basicConfig(filename='debug.log', encoding='utf-8', level=logging.DEBUG)

app = FastAPI()

chat_stores = {}


def make_chat_id():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))


class UrlRequestBody(BaseModel):
    url: str


class ChatIdQuestionBody(BaseModel):
    chat_id: str
    question: str


@app.post("/process_url")
async def process_url(url_request: UrlRequestBody):
    url = url_request.url

    # Associate it with unique Chat ID
    chat_id = make_chat_id()
    logger.info("URL Processing. Chat-id=" + chat_id)

    # Scrape content from URL
    try:
        response = requests.get(url)

        if response.status_code == 200:

            # Clean data
            soup = BeautifulSoup(response.content, 'html.parser')

            for script_styles in soup(['script', 'style']):
                script_styles.decompose()

            text = soup.get_text()

            text = ' '.join(text.split())

            chat_stores[chat_id] = text

            logger.info('Cleaned text')

            return {"chat_id": chat_id, "message": 'URL content processed and stored successfully.'}

        else:
            logger.error(f'Data collection failed HTTP {response.status_code}')
            return {"chat_id": chat_id,
                    "message": f'Collection from URL failed. Status code: {response.status_code}'}

    except Exception as e:
        logger.error("Exception", exc_info=e)
        return {"chat_id": chat_id, "message": f'Collection from URL failed. Error: {e}'}


@app.post("/process_pdf")
async def process_pdf(file: UploadFile = File(...)):
    chat_id = make_chat_id()
    logger.info(f'PDF Processing. Chat-id: {chat_id}')

    file_content = await file.read()
    file_stream = BytesIO(file_content)
    logger.info('File converted to stream.')

    try:
        pdf_reader = PyPDF3.PdfFileReader(file_stream)
        logger.info('Created reader instance successfully.')

        extracted_text = ''

        for page_no in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_no)
            extracted_text += page.extractText()

        text = ' '.join([x for x in extracted_text.split()])

        logger.info('Extracted and cleaned text')

        chat_stores[chat_id] = text

        return {"chat_id": chat_id, "message": 'PDF content processed and stored successfully.'}

    except Exception as e:
        logger.error(f'Exception encountered when processing PDF. Details: {e}')

        return {"chat_id": chat_id, "message": f'An error occured during PDF processing: {e}'}


@app.post("/chat")
async def chat(question_body: ChatIdQuestionBody):
    logger.info(f"Chat question asked. {question_body.chat_id}")

    nlp = spacy.load('en_core_web_sm')
    logger.info('NLP Loaded Successfully.')

    question_vec = nlp(question_body.question).vector

    try:
        document = chat_stores[question_body.chat_id]
    except KeyError:
        logger.error(f"Requested document not found. Chat-id requested: {question_body.chat_id}")
        return {"response": f"No document found with provided chat ID {question_body.chat_id}"}

    # Assuming section refers to sentences
    sentences = list(nlp(document).sents)
    logger.info('Document sentences retrieved.')

    relevant_sentence = ""
    highest_similarity = -1

    for sentence in sentences:
        sentence_vec = sentence.vector

        similarity = cosine_similarity([question_vec], [sentence_vec])[0][0]

        if similarity > highest_similarity:
            highest_similarity = similarity
            relevant_sentence = sentence.text

    logger.info(f'Found relevant sentence: {relevant_sentence}')
    return {"response": relevant_sentence}
