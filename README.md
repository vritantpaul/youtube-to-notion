# How it works:
This program will take any youtube video with subtitles, and summarize the transcript into notes in notion.

The whole process goes through the following steps:
- Input video link. Upon receiving the link it will check whether the video has any subtitles. If yes, retrieve the transcript using **Youtube Transcript API**.
- After getting the transcript, it will scrape the video title from its youtube page using **selectolax**.
- The transcript is then tokenized using **spacy's "en_core_web_sm"**. *The goal is to make sentence length as consistent as possible by concatenating shorter sentences and shortening longer ones.*
- Each sentence is then embedded using **Sentence Transformers**. Specifically using **"all-mpnet-base-v2"**.
- Based on each sentence's cosine similarity, the algorithm identifies split points.
- It then splits the text based on those split points. And finally concatenates the paragraphs.
- Each of those paragraph is passed onto **Open AI API**, which then summarizes them.
- Finally all the paragraphs are written as bulleted list in Notion using **Notion API**.
