from tkinter import *
from PIL import Image, ImageTk
import cv2
import time

class MoodCamApp:
    def __init__(self, root, analyze_function):
        self.root = root
        self.root.title("MoodCam")
        self.root.geometry("640x520")

        # --- Start / Camera Pages ---
        self.start_page = Frame(root)
        self.camera_page = Frame(root)
        for frame in (self.start_page, self.camera_page):
            frame.place(x=0, y=0, relwidth=1, relheight=1)

        # --- Start Page ---
        start_label = Label(self.start_page, text="Welcome to MoodCam!", font=("Arial", 20))
        start_label.pack(pady=100)
        start_button = Button(self.start_page, text="Start", font=("Arial", 16),
                              command=self.show_camera_page)
        start_button.pack()
        self.start_page.tkraise()

        # --- Camera Page ---
        self.canvas = Canvas(self.camera_page, width=640, height=480)
        self.canvas.pack()
        self.mood_label = Label(self.camera_page, text="Your mood will appear here", font=("Arial", 16))
        self.mood_label.place(relx=0.5, rely=0.95, anchor="s")

        # Webcam
        self.video_capture = cv2.VideoCapture(0)
        self.current_image = None

        # Analysis
        self.analyze_function = analyze_function
        self.last_analysis_time = 0
        self.analysis_interval = 0.5  # seconds

    def show_camera_page(self):
        self.camera_page.tkraise()
        self.update_frame()

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            # --- Display Frame ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(frame_rgb)
            self.photo = ImageTk.PhotoImage(image=self.current_image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
            self.canvas.image = self.photo

            # --- Analyze frame ---
            current_time = time.time()
            if current_time - self.last_analysis_time >= self.analysis_interval:
                self.process_frame_for_analysis(frame)
                self.last_analysis_time = current_time

        self.root.after(15, self.update_frame)

    def process_frame_for_analysis(self, frame):
        # Convert frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))

        # If your algorithm expects a numpy array:
        mood = self.analyze_function(resized)

        # Update the label
        self.mood_label.config(text=f"Mood: {mood}")

    def close(self):
        self.video_capture.release()
        self.root.destroy()

# --- Dummy algorithm for demonstration ---
def dummy_mood_algorithm(image):
    # image is 48x48 grayscale numpy array
    # Replace with your real algorithm
    return "Happy"


root = Tk()
app = MoodCamApp(root, analyze_function=dummy_mood_algorithm)
root.protocol("WM_DELETE_WINDOW", app.close)
root.mainloop()
