import tkinter as tk
from tkinter import ttk
import time

class LoadingWindow:

    def __init__(self, title="Loading...", message="Please wait while the application loads..."):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("400x200")
        self.root.resizable(False, False)
        
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.root.winfo_screenheight() // 2) - (200 // 2)
        self.root.geometry(f"400x200+{x}+{y}")
        
        style = ttk.Style()
        style.theme_use('clam')
        
        bg_color = '#fdf6e3'  # Light background
        accent_color = '#000080'  # Blue accent
        
        self.root.configure(bg=bg_color)

        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        title_label = ttk.Label(main_frame, text=title, 
                               font=('Segoe UI', 14, 'bold'),
                               foreground=accent_color)
        title_label.pack(pady=(0, 20))
        
        message_label = ttk.Label(main_frame, text=message,
                                 font=('Segoe UI', 10),
                                 wraplength=350)
        message_label.pack(pady=(0, 20))
        self.progress = ttk.Progressbar(main_frame, mode='determinate', length=300, maximum=100)
        self.progress.pack(pady=(0, 20))
        
        self.status_label = ttk.Label(main_frame, text="Initializing...",
                                     font=('Segoe UI', 9),
                                     foreground='#586e75')
        self.status_label.pack()
        self.progress['value'] = 0
        self.completion_callback = None      

        self.root.protocol("WM_DELETE_WINDOW", lambda: None)
        
    def animate_progress(self, duration=2.5, completion_callback=None):
        
        self.completion_callback = completion_callback
        steps = 100
        step_delay = int((duration * 1000) / steps)  # milliseconds per step
        
        status_messages = [
            "Initializing components...",
            "Loading dependencies...",
            "Setting up interface...",
            "Almost ready...",
            "Complete!"
        ]
        
        def update_progress(step):
            if step <= 100:
                self.progress['value'] = step
                
                if step < 25:
                    self.status_label.config(text=status_messages[0])
                elif step < 50:
                    self.status_label.config(text=status_messages[1])
                elif step < 75:
                    self.status_label.config(text=status_messages[2])
                elif step < 95:
                    self.status_label.config(text=status_messages[3])
                else:
                    self.status_label.config(text=status_messages[4])
                
                self.root.update()
                
                if step < 100:
                    self.root.after(step_delay, lambda: update_progress(step + 1))
                else:

                    self.root.after(300, self._complete_loading)

        update_progress(0)
        
    def _complete_loading(self):
        self.root.destroy()
        if self.completion_callback:
            self.completion_callback()
        
    def show_and_animate(self, duration=2.5, completion_callback=None):
        self.animate_progress(duration, completion_callback)
        self.root.mainloop()