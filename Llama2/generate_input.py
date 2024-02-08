import os
import json
from sklearn.model_selection import train_test_split
import cv2

def convert_video_to_frames(video_file):
    video_path = os.path.join("data","videos",video_file)
    video_name = video_file.split('.')[0]
    images_save_folder = os.path.join("input", "images", video_name)
    
    # Extract images if not already extracted from video for utt
    if not os.path.exists(images_save_folder):
        os.makedirs(images_save_folder)
    
        # Read the video
        cam = cv2.VideoCapture(video_path)

        # Count frames
        total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        print("The number of frames in the video: {}".format(total_frames))

        # Skip every 0.5 sec
        time_skips = float(500)

        curr_frame = 0
        ret, frame = cam.read()
        while(True):
            if ret:
                name = os.path.join(images_save_folder, "frame"+str(curr_frame)+".jpg")
                print("Creating: " + name)
                cv2.imwrite(name, frame)
                cam.set(cv2.CAP_PROP_POS_MSEC, (curr_frame*time_skips))

                ret, frame = cam.read()
                curr_frame += 1
                # Max 20 frames for a video
                if curr_frame == 20:
                    break
            else:
                break
        
        cam.release()
        cv2.destroyAllWindows()

def videoCaption(video_file, max_captions=5):
    """Returns a list of captions for the video."""
    video_captions = []
    
    images_folder = os.path.join("input", "images", video_file.split(".")[0])
    for image in os.listdir(images_folder):
        caption = get_image_captions(os.path.join(images_folder, image))
        video_captions.append(caption)
    return video_captions

def audioCaptions(videoName, max_captions=1):
    """Returns a set of captions for the audio within the video."""
    audio_captions = []
    return audio_captions
    
def create_json(data, save_folder, speaker=False, video=False, audio=False):
    # Check if folder and input data already exist
    if speaker and not video and not audio:
        save_folder = os.path.join(save_folder, "speaker")
    elif speaker and video and not audio:
        save_folder = os.path.join(save_folder, "video")
    elif speaker and audio and not video:
        save_folder = os.path.join(save_folder, "audio")
    elif speaker and audio and video:
        save_folder = os.path.join(save_folder, "audio_video")
    else:
        save_folder = os.path.join(save_folder, "context")
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        print("Input already present at {}\n".format(save_folder))
        conv_file = os.path.join(save_folder, "conversations.json")
        return conv_file
    
    for conversation in data:
        for utt in conversation["conversation"]:
            video_name = "dia"+str(conversation["conversation_ID"])+"utt"+str(utt["utterance_ID"])+".mp4"
            utt["video_name"] = video_name
            # Include video context in input
            # if video:
            #     convert_video_to_frames(utt["video_name"])
            #     utt["video_caption"] = videoCaption(utt["video_name"])
            # Include audio context in input
            # if audio:
            #     utt["audio_captions"] = audioCaptions(utt["video_name"])
    
    print("Number of conversations in dataset: {}".format(len(data)))
    
    conv_file = os.path.join(save_folder, "conversations.json")
    with open(conv_file, 'w') as f:
        json.dump(data, f)
    return conv_file

def generate_input(args):
    text_folder = os.path.join("../data", "text")
    save_dir = os.path.join("input")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(text_folder, "Subtask_2_train.json")
    file = open(file_path)
    data = json.load(file)

    data_trainval, data_test = train_test_split(data, test_size=0.1, random_state=args.seed)
    data_train, data_val = train_test_split(data_trainval, test_size=0.1, random_state=args.seed)
    
    save_folder_test = os.path.join(save_dir, 'test')
    save_folder_train = os.path.join(save_dir, 'train')
    save_folder_val = os.path.join(save_dir, 'val')

    test_file = create_json(data_test, save_folder_test, args.speaker, args.video, args.audio)
    train_file = create_json(data_train, save_folder_train, args.speaker, args.video, args.audio)
    val_file = create_json(data_val, save_folder_val, args.speaker, args.video, args.audio)
    return test_file, train_file, val_file
