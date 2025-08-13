import tkinter as tk
from tkinter import ttk
import time

class LoadingWindow:

    def __init__(self, title="Loading...", message="Please wait while the application loads..."):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("400x240")
        self.root.resizable(False, False)
        
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.root.winfo_screenheight() // 2) - (240 // 2)
        self.root.geometry(f"400x240+{x}+{y}")
        
        bg_color = '#fdf6e3'  
        accent_color = '#000080'  
        block_color = '#000080'  
        empty_color = '#e8dcc0'  
        
        self.root.configure(bg=bg_color)

        main_frame = tk.Frame(self.root, bg=bg_color, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(main_frame, text=title, 
                              font=('Segoe UI', 14, 'bold'),
                              fg=accent_color, bg=bg_color)
        title_label.pack(pady=(0, 20))
        
        message_label = tk.Label(main_frame, text=message,
                                font=('Segoe UI', 10),
                                wraplength=350, bg=bg_color)
        message_label.pack(pady=(0, 20))
        
        self.progress_frame = tk.Frame(main_frame, bg=bg_color)
        self.progress_frame.pack(pady=(0, 20))
        
        self.blocks = []
        self.total_blocks = 20
        block_width = 12
        block_height = 20
        block_spacing = 2
        
        for i in range(self.total_blocks):
            block = tk.Frame(self.progress_frame, 
                           width=block_width, 
                           height=block_height,
                           bg=empty_color,
                           relief='solid',
                           bd=1)
            block.pack(side=tk.LEFT, padx=(block_spacing if i > 0 else 0, 0))
            block.pack_propagate(False)  
            self.blocks.append(block)
        
        self.status_label = tk.Label(main_frame, text="Initializing...",
                                   font=('Segoe UI', 9),
                                   fg='#586e75', bg=bg_color)
        self.status_label.pack()

        footer_label = tk.Label(main_frame, text="Developed by Falah Sheikh",
                               font=('Segoe UI', 10, 'italic'),
                               fg='#93a1a1', bg=bg_color)
        footer_label.pack(pady=(20, 0))

        self.current_block = 0
        self.completion_callback = None
        self.block_color = block_color
        self.empty_color = empty_color

        self.root.protocol("WM_DELETE_WINDOW", lambda: None)
        
    def animate_progress(self, duration=2.5, completion_callback=None):
        self.completion_callback = completion_callback
        step_delay = int((duration * 1000) / self.total_blocks)  # milliseconds per block
        
        status_messages = [
            "Initializing components...",
            "Loading dependencies...",
            "Setting up interface...",
            "Almost ready...",
            "Complete!"
        ]
        
        def update_progress(block_index):
            if block_index < self.total_blocks:
                self.blocks[block_index].config(bg=self.block_color)
                
                progress_percent = (block_index + 1) / self.total_blocks * 100
                
                if progress_percent < 25:
                    self.status_label.config(text=status_messages[0])
                elif progress_percent < 50:
                    self.status_label.config(text=status_messages[1])
                elif progress_percent < 75:
                    self.status_label.config(text=status_messages[2])
                elif progress_percent < 95:
                    self.status_label.config(text=status_messages[3])
                else:
                    self.status_label.config(text=status_messages[4])
                
                self.root.update()
                
                if block_index < self.total_blocks - 1:
                    self.root.after(step_delay, lambda: update_progress(block_index + 1))
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