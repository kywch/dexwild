import tkinter as tk
from tkinter import font as tkFont
import cv2
from PIL import Image, ImageTk
from tkinter import messagebox 
import numpy as np

class RecordUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("800x480")
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.root.title("Record UI")

        # Example variables
        self.hertz = 0
        self.prev_hertz = 0
        self.zed_is_recording = False
        self.start_recording = False
        self.episode_counter = 0
        self.cam_img = np.zeros((320, 240, 3))
        
        # Camera toggle flag (True => show camera feed; False => no feed)
        self.check_camera = False
        self.delete_last_episode = False

        # Use a nice big default font
        self.default_font = tkFont.Font(family="Helvetica", size=20, weight="bold")

        # -----------
        # MAIN FRAME
        # -----------
        self.main_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.main_frame.pack(fill="both", expand=True)

        # 2x2 grid inside main_frame
        self.grid_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.grid_frame.pack(pady=20)

        # 1) Hertz & Episode square
        self.hertz_episode_square = tk.Label(
            self.grid_frame,
            text=f"Hertz: {self.hertz:.3f}\nEpisode: {self.episode_counter}",
            font=self.default_font,
            width=20,
            height=5,
            bg="red"  # default
        )
        self.hertz_episode_square.grid(row=0, column=0, padx=10, pady=10)

        # ---------------------------------------------------
        # 2) SUB-FRAME for BOTH "Toggle Camera" & "Delete" btn
        # ---------------------------------------------------
        self.button_frame = tk.Frame(self.grid_frame, bg="#f0f0f0")
        self.button_frame.grid(row=0, column=1, padx=10, pady=10)

        # Toggle Camera Button in the sub-frame
        self.toggle_camera_button = tk.Button(
            self.button_frame,
            text="Show Camera",
            command=self.show_camera_frame,
            font=self.default_font
        )
        self.toggle_camera_button.pack(fill='x', pady=5)

        # Delete Episode Button in the same sub-frame
        self.delete_episode_button = tk.Button(
            self.button_frame,
            text="Delete Episode",
            font=self.default_font,
            command=self.delete_episode
        )
        self.delete_episode_button.pack(fill='x', pady=5)

        # 3) Zed Recording Block
        self.zed_block = tk.Label(
            self.grid_frame,
            text="SLAM Recording: No",
            fg="white",
            width=20,
            height=5,
            font=self.default_font,
            bg="#e74c3c"
        )
        self.zed_block.grid(row=1, column=0, padx=10, pady=10)

        # 4) Start Recording Block
        self.start_block = tk.Label(
            self.grid_frame,
            text="Start Recording: No",
            fg="white",
            width=20,
            height=5,
            font=self.default_font,
            bg="#e74c3c"
        )
        self.start_block.grid(row=1, column=1, padx=10, pady=10)

        # ------------
        # CAMERA FRAME
        # ------------
        self.camera_frame = tk.Frame(self.root, bg="black")  # hidden initially

        # Camera Label to display frames
        self.camera_label = tk.Label(self.camera_frame, bg="black")
        self.camera_label.pack(pady=10)

        # Button to close camera feed
        self.close_camera_button = tk.Button(
            self.camera_frame,
            text="Close Camera",
            font=self.default_font,
            command=self.hide_camera_frame
        )
        self.close_camera_button.pack(pady=10)

        # For real camera capture: create a VideoCapture object
        # If your environment is headless or you feed images differently, adapt this
        self.cap = cv2.VideoCapture(0)  # open default camera index=0

        # Initial UI update
        self.update_ui()
    
    def delete_episode(self):
        #show popup that says it deleted the episode
        self.delete_last_episode = True
        messagebox.showinfo("Delete Episode", "The episode has been deleted.")

    def show_camera_frame(self):
        """
        Hides main_frame and shows camera_frame, starts updating camera feed.
        """
        self.main_frame.pack_forget()         # hide main UI
        self.camera_frame.pack(fill="both", expand=True)  # show camera UI
        self.check_camera = True

    def hide_camera_frame(self):
        """
        Hides camera_frame and shows main_frame, stops updating camera feed.
        """
        self.check_camera = False
        self.camera_frame.pack_forget()       # hide camera UI
        self.main_frame.pack(fill="both", expand=True)  # show main UI

    def update_camera(self):
        """
        Continuously fetch frames from OpenCV (if check_camera is True)
        and update self.camera_label.
        """
        if self.check_camera:
            # Convert BGR -> RGB for displaying in Tkinter
            frame_rgb = cv2.cvtColor(self.cam_img, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)

            # Convert to ImageTk
            imgtk = ImageTk.PhotoImage(image=pil_image)

            # Update the label
            self.camera_label.config(image=imgtk)
            self.camera_label.image = imgtk

            # Schedule next frame read
        else:
            # Optionally, clear out the last frame if you want a blank camera_label
            self.camera_label.config(image="")

    def toggle_camera(self):
        """
        (Optional if you prefer just one Show/Close button)
        Currently unused in this example, but you can repurpose.
        """
        self.check_camera = not self.check_camera
        print(f"check_camera is now: {self.check_camera}")

    def update_ui(self):
        """
        Update the main UI blocks.
        """
        # Hertz color logic
        if self.hertz != self.prev_hertz:
            self.hertz_episode_square.config(bg="green")
        else:
            self.hertz_episode_square.config(bg="red")

        # Update text
        self.hertz_episode_square.config(
            text=f"Hertz: {self.hertz:.3f}\nEpisode: {self.episode_counter}"
        )
        
        # Keep track of previous Hertz
        self.prev_hertz = self.hertz
        
        # Zed Recording
        if self.zed_is_recording:
            self.zed_block.config(
                bg="#2ecc71",
                text="SLAM Recording: Yes"
            )
        else:
            self.zed_block.config(
                bg="#e74c3c",
                text="SLAM Recording: No"
            )
        
        # Start Recording
        if self.start_recording:
            self.start_block.config(
                bg="#2ecc71",
                text="Start Recording: Yes"
            )
        else:
            self.start_block.config(
                bg="#e74c3c",
                text="Start Recording: No"
            )

    def on_close(self):
        """
        Called when the main window is closed.
        """
        # Release camera if in use
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

def main():
    app = RecordUI()
    app.root.protocol("WM_DELETE_WINDOW", app.on_close)  # Proper cleanup on close
    app.root.mainloop()

if __name__ == "__main__":
    main()
