        if not self.model:
            messagebox.showwarning("No Model Loaded", 
                "Please load a tri-class AI model (CN/EMCI/LMCI) first.\n\n"
                "The model must have exactly 3 output classes and contain convolutional layers.")
            return