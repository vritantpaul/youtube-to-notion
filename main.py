import openai
import requests
import json
import time
import pytz
import datetime
import logging
from selectolax.parser import HTMLParser
from text_to_paragraph import split_to_paragraphs
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import NoTranscriptFound


def log_config():
    # CREATING A LOGGER OBJECT ---
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # setting logging level to "INFO"
    # CREATING A FORMATTER OBJECT ---
    custom_format = logging.Formatter(
        "%(asctime)s: %(levelname)s: %(message)s", datefmt="%d-%b %H:%M"
    )
    # CREATING A HANDLER OBJECT ---
    handler = logging.StreamHandler()
    handler.setFormatter(custom_format)  # setting up format rules
    # ADDING THE FILE HANDLER TO THE LOGGER ---
    logger.addHandler(handler)
    return logger


def get_transcript(url):
    """Getting the transcript of the video using 'youtube_transcript_api' module."""
    video_id = url.split("/")[-1]
    # getting a list of all transcripts
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    try:
        # we only need the manually generated transcripts and not the auto-generated ones
        response = transcript_list.find_manually_created_transcript(
            ["en", "en-US", "en-GB"]  # possible language codes for english transcript
        ).fetch()
        return " ".join(lines["text"].replace("\n", " ") for lines in response)
    except NoTranscriptFound as e:
        logger.info("Oops! No transcript found for this video.")
        exit()


def get_title(url):
    """Getting the title of the video by scraping its youtube page."""
    response = requests.get(url)
    tree = HTMLParser(response.text)
    name = tree.css_first("title").text()  # getting the text from the title tag
    return name.split("-")[0].strip()


def get_time():
    """Coverting utc time to ist."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    ist_tz = pytz.timezone("Asia/Kolkata")  # convert to IST timezone
    now_ist = now_utc.astimezone(ist_tz)
    return now_ist.strftime("%Y-%m-%dT%H:%M:%S%z")  # format the datetime as a string


def summarize(prompt):
    """Summarizing using open AI API."""
    openai.api_key = "Your OpenAI API key"

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response["choices"][0]["text"].strip()


def write_notion(url=None, title=None, notes=None, time=None):
    """Writing the summary in Notion using Notion API."""

    # Authorization Credentials ---
    token = "Your Notion Token"
    database_id = "Your Database ID"

    # Link & Headers for the POST request ---
    link = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2022-06-28",
        "content-type": "application/json",
        "accept": "application/json",
    }

    # Data to POST (send) to the Notion API ---
    create_page = {
        "parent": {"database_id": database_id},  # the database which contains the page
        "properties": {  # properties of the page
            "Title": {  # column name
                "title": [  # title text type - "title" (bold)
                    {"text": {"content": title}}  # title text
                ]
            },
            "URL": {"url": url},  # url of the video
            "Published": {"date": {"start": time, "end": None}},  # publishing date
        },
        "children": [],  # content of the page
    }

    for note in notes:
        paragraph = {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": f"{note}"}}]
            },
        }
        create_page["children"].append(paragraph)

    # Converting Dictionary object to String ---
    data = json.dumps(create_page)

    # Making the request ---
    response = requests.post(link, headers=headers, data=data)
    return response


def main():
    # INPUT VIDEO URL ---
    url = input("Enter Video URL: ")

    # GETTING VIDEO TRANSCRIPT, TITLE AND CURRENT TIME ---
    transcript = get_transcript(url)
    title = get_title(url)
    logger.info(f"Successfully retrieved transcript for the Video: {title}.")

    current_time = get_time()

    # SPLITTING THE TRANSCRIPT INTO PARAGRAPHS ---
    logger.info("Splitting text into paragraphs...")
    start = time.time()
    paragraphs = split_to_paragraphs(transcript).split("\n\n")
    end = time.time()
    duration = round((end - start) / 60, 2)
    logger.info(
        f"Successfully split the text into {len(paragraphs)} paragraphs in {duration} minute(s)."
    )

    # SUMMARIZING THE PARAGRAPHS USING OPEN AI API ---
    with open("Prompt.txt", "r") as file:
        prompt = file.read()
    notes = []
    to_summarize = input("Do you want to summarize the text? [Y/n]: ")
    if to_summarize == "Y":
        logger.info("Summarizing individual paragraphs with ChatGPT...")
        for paragraph in paragraphs:
            final_prompt = prompt + paragraph
            summary = summarize(final_prompt)
            notes.append(summary)
        logger.info("All paragraphs summarized.")
    else:
        logger.warning("Not summarizing the paragraphs.")
        notes.extend(iter(paragraphs))

    # WRITING THE PARAGRAPHS IN NOTION USING NOTION API ---
    logger.info("Writing notes in Notion...")
    response = write_notion(url, title=title, notes=notes, time=current_time)
    if response.ok:
        logger.info("Notes successfully written.")


if __name__ == "__main__":
    logger = log_config()
    main()
