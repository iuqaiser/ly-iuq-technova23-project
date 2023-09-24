# Guiding Light - A new way to interact with your world

**Inspiration** 💡 \
We wanted to try working with computer vision, but also wanted to develop a tool to aid a marginalized community of people. Object detection projects are the norm for computer vision projects but we added speech commands/text-to-speech functionality for greater accessibility.

**What it does** ✨ \

Once started and after being voice-activated by the user, Guiding Light uses your webcam to capture real-time views of the area around you, then it both visually and verbally identifies what objects are in your proximity. Users are supposed to point the camera in front of them as they walk in order to detect obstacles.

**How we built it** 💻 \ 
Guiding Light uses the Ultralytics YOLOv8 object detection model, and retrieves video footage using OpenCV. It uses the pyttsx3 and SpeechRecognition libraries for text-to-speech and speech recognition functionalities respectively.

**Challenges we ran into** 🌩️ \ 
We thought the hardest part would be training the model, but that was actually one of the easiest. One of the harder parts was working on the speech recognition. As we're writing this, we haven't figured out how to use voice commands to end the program using just audio input, so that should speak for itself (literally). We're also had (/am having) difficulty figuring out the colour detection aspect of the project. I (Iman) have used a method that takes the average of the RGB values from the footage and returns the most common colour from red, green, and blue, but the applications of this are limited and being able to detect the colour of a specific object would be much better. Moreover, the original service we wanted to deploy on didn't support webcam functionality, plus I overtrained the model, plus we were both sleep-deprived the entire time -- but I've heard if you love something, you'll learn to accept it, flaws and all. And aren't flaws what make us unique?

**Accomplishments that we're proud of** 🎖️ \ 
First off, this was the most complex project Iman has worked on for a hackathon, so the fact that it can run without getting a traceback error every second is enough of a miracle for me. But in all seriousness, I'm glad I was able to experience working with and training an ML model, and implementing at least a rudimentary form of colour detection. Lavanya is also proud of the work she has done with speech recognition and text to speech, which greatly bolstered the accessibility of the project and made for an interesting challenge. 
For this hackathon, we wanted to really experiment with technologies we have not worked with before and I think we can positively say that we're proud of what we could make of this project.

**What we learned **🔖 \ 
Iman knows a lot more know than she did before about Python and its functionalities, and about an example of how data models are used in real life. Lavanya learned about some interesting python libraries out there and also computer vision like she wanted to and was able to use it to build a project. (and how certain operating systems are not to her liking 🙃)

**What's next for Guiding Light** 🌟 \ 
- Deployment as a mobile app to make it easily accessible
- Improved versatility on voice commands, including synonyms for commands
- Expansion of features into personal assistant application, with extended conversational capabilities
- Specified orientations/locations of objects with respect to frame, or list all objects in frame at a given moment (e.g. bowl on left, spoon on right), as well as depth perception
- Training on more object classes, namely on everyday items such as clothing, utensils, and more
- Refinement of colour detection feature to identify colours of specific objects, to be combined with above feature and assist users with identifying and choosing clothes
- Facial recognition to notify presence of notable individuals to user, including the user themselves, their family members, etc.
- Multilanguage support, which can allow it to be used as a language learning tool as well
- We may or may not have a lot more ideas, stay tuned!
