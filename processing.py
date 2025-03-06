import subprocess
import os
import assemblyai as aai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from gtts import gTTS
from pydub import AudioSegment
import ffmpeg
from dotenv import load_dotenv
import time
import shutil

# Set up the environment
load_dotenv(dotenv_path=r"C:\coding\python\INFOSYS\.env")
api_key = os.getenv("API_KEY")
aai.settings.api_key = "7e188095c594495d94011a0eebe1c47f"

def extract_audio(video_path, audio_path):
    """
    Extracts audio from the video file using FFmpeg.
    """
    if not audio_path.endswith(".wav"):
        audio_path += ".wav"
    video_path = video_path.strip('"')
    
    command = [
        "ffmpeg",
        "-i", video_path, 
        "-vn",              
        "-acodec", "pcm_s16le", 
        "-ar", "44100",     
        "-ac", "2",         
        audio_path          
    ]
    subprocess.run(command, check=True)

def transcribe_audio(input_file, output_file):
    """
    Transcribes audio and saves it as a text file.
    """
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(input_file)
    
    if transcript.status == aai.TranscriptStatus.error:
        print(f"Transcription error: {transcript.error}")
    else:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(transcript.text)
        print(f"Transcript saved to {output_file}")

def translate_text(transcribed_text, target_language):
    """
    Translates transcribed text into the target language.
    """
    template = """Translate the following text into {language}: {sentence}"""
    prompt = PromptTemplate(
        input_variables=["sentence", "language"],
        template=template
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=api_key,
        timeout=600
    )

    chain = prompt | llm | StrOutputParser()
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = chain.invoke({"sentence": transcribed_text, "language": target_language})
            return res
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2 ** attempt) 
            else:
                raise RuntimeError("Max retries reached. Unable to complete the translation.")

def text_to_speech(input_file, output_file, language_code="hi", target_duration=None):
    """
    Converts text to speech and adjusts audio duration.
    """
    with open(input_file, "r", encoding="utf-8") as file:
        text = file.read()

    tts = gTTS(text=text, lang=language_code, slow=False)
    temp_audio_file = "temp_audio.mp3"
    tts.save(temp_audio_file)
    print("Generated audio file:", temp_audio_file)

    if target_duration:
        audio = AudioSegment.from_mp3(temp_audio_file)
        current_duration = len(audio)
        
        if current_duration != target_duration:
            speed_factor = current_duration / target_duration
            adjusted_audio = audio.speedup(playback_speed=speed_factor)
            adjusted_audio.export(output_file, format="mp3")
            print(f"Adjusted audio saved to {output_file} with duration: {target_duration}ms")
        else:
            audio.export(output_file, format="mp3")
            print(f"Audio saved to {output_file}.")
    else:
        os.rename(temp_audio_file, output_file)
        print(f"Audio saved to {output_file}")

    if os.path.exists(temp_audio_file):
        os.remove(temp_audio_file)

def replace_audio_in_video():
    """
    Replaces audio in the video with a new audio file.
    """
    video_file = input("Enter the path to the video file: ").strip()
    audio_file = input("Enter the path to the audio file: ").strip()
    output_file = input("Enter the desired output file name: ").strip()

    try:
        video_stream = ffmpeg.input(video_file)
        audio_stream = ffmpeg.input(audio_file)

        print(f"Processing... Output will be saved to: {output_file}")
        ffmpeg.output(
            video_stream.video,
            audio_stream.audio,
            output_file,
            vcodec='copy',
            acodec='aac'
        ).run()
        print(f"Audio successfully replaced! Output saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

def process_video(video_path, language_choice, output_path):
    """
    Main processing function that extracts audio, transcribes, translates, and replaces audio.
    """
    audio_path = "extracted_audio.wav"
    transcript_path = "transcript.txt"
    translated_text_path = "translated_text.txt"
    final_audio_path = "final_audio.mp3"
    
    # Extract audio from video
    extract_audio(video_path, audio_path)

    # Transcribe audio to text
    transcribe_audio(audio_path, transcript_path)

    # Translate transcribed text to chosen language
    with open(transcript_path, "r", encoding="utf-8") as file:
        transcribed_text = file.read()

    translated_text = translate_text(transcribed_text, language_choice)

    with open(translated_text_path, "w", encoding="utf-8") as file:
        file.write(translated_text)

    # Convert translated text to speech
    text_to_speech(translated_text_path, final_audio_path, language_code=language_choice)

    # Replace audio in the video
    replace_audio_in_video()

if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ").strip()
    language_choice = input("Enter the language code (e.g., 'hi' for Hindi): ").strip().lower()
    output_path = input("Enter the output file path (e.g., output_video.mp4): ").strip()

    process_video(video_path, language_choice, output_path)
