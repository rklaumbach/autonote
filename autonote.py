import os
import logging
import whisper
from openai import OpenAI
import datetime
import argparse

# Fixed directories
TRANSCRIBE_DIR = "transcribed_text"
CLEAN_DIR = "cleaned_text"
# SUMMARIZE_DIR = "summarized_text"  # Removed as summarization is no longer needed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("autonote.log"),
        logging.StreamHandler()
    ]
)

def load_api_key():
    """Load OpenAI API key from environment variable or file."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        try:
            with open('openai_key.txt', 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            logging.error("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable or provide 'openai_key.txt'.")
            exit(1)
        except Exception as e:
            logging.error(f"An error occurred while reading the API key: {e}")
            exit(1)
    return api_key

def load_client():
    """Instantiate and return the OpenAI client."""
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)
    return client

def transcribe_audio(audio_file: str):
    """Transcribe the given audio file using Whisper."""
    if not os.path.isfile(audio_file):
        logging.error(f"The audio file '{audio_file}' does not exist.")
        return

    # Load the client (not used in transcription but kept for consistency)
    client = load_client()

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Step 1: Transcribe Audio
    logging.info("Loading Whisper model...")
    try:
        model = whisper.load_model("medium")
    except Exception as e:
        logging.error(f"Error loading Whisper model: {e}")
        return

    logging.info(f"Transcribing audio file '{audio_file}'...")
    try:
        result = model.transcribe(audio_file)
        transcribed_text = result["text"]
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return

    # Ensure directory exists and save transcription to file
    os.makedirs(TRANSCRIBE_DIR, exist_ok=True)
    transcribed_filename = os.path.join(TRANSCRIBE_DIR, f"transcribed_text_{timestamp}.txt")
    try:
        with open(transcribed_filename, "w", encoding="utf-8") as f:
            f.write(transcribed_text)
        logging.info(f"Transcribed text saved to '{transcribed_filename}'.")
    except Exception as e:
        logging.error(f"Error saving transcribed text: {e}")
        return

    # Step 2: Clean Text
    clean_text(transcribed_text, timestamp)

def clean_text(transcribed_text: str, timestamp: str):
    """Clean the transcribed text using OpenAI's GPT model."""
    client = load_client()

    # Define the prompt with the added instruction for personal frustrations or reflections
    prompt = (
        "Please clean the following text and provide a well-structured synopsis by performing the following tasks:\n"
        "1. Remove filler words and correct any grammatical or typographical errors.\n"
        "2. Ensure that all essential information from the original text is retained accurately.\n"
        "3. Organize the text into clear paragraphs.\n"
        "4. Summarize key points without omitting any relevant details.\n"
        "5. Maintain a formal and organized tone, avoiding profanity and slang.\n"
        "6. While retaining a formal tone, also include the speaker's personal frustrations or reflections.\n"
        "7. Wrap the text so line breaks make it read like a typical essay in any text editor.\n"
        "8. Incorporate subheadings to categorize different sections/topics within the text.\n\n"
        "Do not add any additional commentary or separators.\n\n"
        "Please format the output in plain text.\n\n"
        "### Example Output\n\n"
        "### Main Heading\n\n"
        "**Subheading**\n\n"
        "Text content goes here. This section should provide detailed information relevant to the subheading.\n\n"
        "### Main Heading\n\n"
        "**Subheading**\n\n"
        "Text content goes here. This section should provide detailed information relevant to the subheading.\n\n"
        "### Main Heading\n\n"
        "**Subheading**\n\n"
        "Text content goes here. This section should provide detailed information relevant to the subheading.\n\n"
        "### Conclusion\n\n"
        "Final thoughts and summary of the discussed topics.\n\n"
        + transcribed_text
    )

    logging.info("Cleaning text...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=4096,
            temperature=0.5,
        )
        cleaned_text = response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error during text cleaning: {e}")
        return

    # Ensure directory exists and save cleaned text to file
    os.makedirs(CLEAN_DIR, exist_ok=True)
    cleaned_filename = os.path.join(CLEAN_DIR, f"cleaned_text_{timestamp}.txt")
    try:
        with open(cleaned_filename, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        logging.info(f"Cleaned text saved to '{cleaned_filename}'.")
    except Exception as e:
        logging.error(f"Error saving cleaned text: {e}")
        return

def run_all(audio_file: str):
    """Run all steps: transcribe and clean."""
    transcribe_audio(audio_file)

    # Since summarization is removed, no need to handle it here

def main():
    parser = argparse.ArgumentParser(description="Transcribe and clean audio files.")
    subparsers = parser.add_subparsers(dest='command', help='Sub-commands help')

    # Subparser for transcribe
    parser_transcribe = subparsers.add_parser('transcribe', help='Transcribe an audio file')
    parser_transcribe.add_argument('audio_file', type=str, help='Path to the audio file (e.g., test_audio.mp3)')

    # Subparser for clean (optional since cleaning is now part of transcribe)
    # You can remove this if cleaning is always done after transcription
    # Alternatively, keep it if you want to clean existing transcription files
    parser_clean = subparsers.add_parser('clean', help='Clean a transcription file')
    parser_clean.add_argument('transcription_file', type=str, help='Filename of the transcription file (e.g., transcribed_text_20230924123000.txt)')

    # Removed summarize subparser since summarization is no longer needed

    # Subparser for all
    parser_all = subparsers.add_parser('all', help='Run all steps: transcribe and clean')
    parser_all.add_argument('audio_file', type=str, help='Path to the audio file (e.g., test_audio.mp3)')

    args = parser.parse_args()

    if args.command == 'transcribe':
        transcribe_audio(args.audio_file)
    elif args.command == 'clean':
        # For cleaning existing transcription files
        transcription_path = os.path.join(TRANSCRIBE_DIR, args.transcription_file)
        if not os.path.isfile(transcription_path):
            logging.error(f"The transcription file '{transcription_path}' does not exist.")
        else:
            # Read transcribed text
            try:
                with open(transcription_path, "r", encoding="utf-8") as f:
                    transcribed_text = f.read()
                timestamp = args.transcription_file.replace("transcribed_text_", "").replace(".txt", "")
                clean_text(transcribed_text, timestamp)
            except Exception as e:
                logging.error(f"Error reading transcription file: {e}")
    elif args.command == 'all':
        run_all(args.audio_file)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
