"""The annotation lists of the Slice Viewer, and undo and redo."""

import tkinter as tk
from tkinter import messagebox, simpledialog

from ..annotations import DRAWING, summary

HIGHLIGHT_MS = 1500


def _iid(kind, item_id):
    return f"{kind}_{item_id}"


class AnnotationPanelMixin:
    """Keeps the two lists (drawings and measurements) the same as the annotation store."""

    def _after_annotation_change(self):
        self._refresh_trees()
        self._update_history_buttons()
        self.update_views()

    def _refresh_trees(self):
        # The selection is not restored: selecting an item moves the views to its slice.
        for kind, tree in self.trees.items():
            tree.delete(*tree.get_children())
            for item in sorted(self.store.items(kind), key=lambda item: item["id"]):
                text = summary(kind, item)
                tree.insert("", tk.END, iid=_iid(kind, item["id"]), text=text, values=(text,))

    def _update_history_buttons(self):
        self.undo_button.config(state=tk.NORMAL if self.store.can_undo else tk.DISABLED)
        self.redo_button.config(state=tk.NORMAL if self.store.can_redo else tk.DISABLED)

    def _selected(self, kind):
        """The selected annotation of the list, or None."""
        selection = self.trees[kind].selection()
        if not selection:
            return None
        return self.store.find(kind, int(selection[0].rsplit("_", 1)[1]))

    def _on_tree_select(self, kind, _event=None):
        """Go to the slice of the selected annotation and highlight it for a short time."""
        item = self._selected(kind)
        if item is None:
            return
        self.highlight = (kind, item["id"])
        self.current_slices[item["plane"]] = item["slice"]
        self._sync_controls()
        self.update_views()
        self.root.after(HIGHLIGHT_MS, self._clear_highlight, (kind, item["id"]))

    def _clear_highlight(self, highlight):
        if self.highlight == highlight:
            self.highlight = None
            self.update_views()

    def comment_annotation(self, kind):
        """Add or change the comment of the selected annotation, or of the newest one if none is selected."""
        item = self._selected(kind) or self.store.latest(kind)
        name = "drawing" if kind == DRAWING else "measurement"
        if item is None:
            messagebox.showwarning("Comment", f"Make a {name} first.", parent=self.root)
            return
        comment = simpledialog.askstring(
            "Comment",
            f"Comment for {summary(kind, {**item, 'comment': None})}:",
            initialvalue=item.get("comment", ""),
            parent=self.root,
        )
        if comment is not None and self.store.set_comment(kind, item["id"], comment):
            self._after_annotation_change()
            self.set_status(f"Comment saved for {name} {item['id']}")

    def delete_annotation(self, kind):
        item = self._selected(kind)
        if item is None:
            messagebox.showwarning("Delete", "Select an item in the list first.", parent=self.root)
            return
        if not messagebox.askyesno("Delete", f"Delete {summary(kind, item)}?", parent=self.root):
            return
        self.store.delete(kind, item["id"])
        if self.highlight == (kind, item["id"]):
            self.highlight = None
        self._after_annotation_change()
        self.set_status(f"Deleted {summary(kind, item)}")

    def clear_annotations(self, kind, confirm=True):
        name = "drawings" if kind == DRAWING else "measurements"
        if not self.store.items(kind):
            if confirm:
                messagebox.showinfo("Clear", f"There are no {name}.", parent=self.root)
            return
        if confirm and not messagebox.askyesno("Clear", f"Clear all {name}?", parent=self.root):
            return
        self.store.clear(kind)
        self._after_annotation_change()
        self.set_status(f"All {name} cleared")

    def undo(self):
        description = self.store.undo()
        self.set_status(f"Undone: {description}" if description else "Nothing to undo")
        self._after_annotation_change()

    def redo(self):
        description = self.store.redo()
        self.set_status(f"Redone: {description}" if description else "Nothing to redo")
        self._after_annotation_change()
