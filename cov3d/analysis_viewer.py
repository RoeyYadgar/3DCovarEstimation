import os
import pickle
import sys
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from cov3d.analyze import create_pc_figure, create_umap_figure


class AnalyzeViewer:
    """Interactive GUI for visualizing and analyzing coordinate data with UMAP and PCA projections.

    This class provides a tkinter-based interface for exploring coordinate data through
    various visualization modes including UMAP projections and principal component analysis.
    Users can interact with the plots to select cluster coordinates and save/load them.

    Attributes:
        master: The tkinter root window
        data: Dictionary containing coordinate data and analysis results
        dir: Directory path for file operations
        coords: Principal component coordinates
        umap_coords: UMAP-reduced coordinates
        cluster_coords: Selected cluster center coordinates
        umap_cluster_coords: UMAP coordinates of cluster centers
        selected_cluster_coords: List of selected cluster coordinates
        figures: Dictionary storing matplotlib figures
        color_by: StringVar for color coding selection
        figure_type: StringVar for figure type selection
    """

    def __init__(self, master: tk.Tk, data: Dict[str, Any], dir: str) -> None:
        """Initialize the AnalyzeViewer with data and directory.

        Args:
            master: The tkinter root window
            data: Dictionary containing coordinate data with keys 'coords', 'umap_coords',
                  and optionally 'cluster_coords' and 'umap_cluster_coords'
            dir: Directory path for file operations
        """
        self.master = master
        menubar = tk.Menu(self.master)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self._open_new_file)
        filemenu.add_command(label="Load latent coords", command=self._load_latent_coords)
        filemenu.add_command(label="Save", command=self._save_cluster_coords)
        menubar.add_cascade(label="File", menu=filemenu)
        self.master.config(menu=menubar)
        self.data = data
        self.dir = dir
        self.coords = data["coords"]
        self.umap_coords = data["umap_coords"]
        self.cluster_coords = data.get("cluster_coords", None)
        self.umap_cluster_coords = data.get("umap_cluster_coords", None)
        self.selected_cluster_coords = []
        self.figures = {}
        self.color_by = tk.StringVar(value="None")
        self.figure_type = tk.StringVar(value="umap")
        self._setup_gui()
        self._draw_figure()

    def _open_new_file(self) -> None:
        """Open a new analysis data file and update the viewer.

        Prompts user to select a pickle file containing analysis coordinates, loads the data, and
        refreshes the display.
        """
        path = filedialog.askopenfilename(
            title="Select analyze_coordinates pkl", filetypes=[("Pickle files", "*.pkl")], initialdir=self.dir
        )
        if not path:
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.dir = os.path.split(path)[0]
        self.data = data
        self.coords = data["coords"]
        self.umap_coords = data["umap_coords"]
        self.cluster_coords = data.get("cluster_coords", None)
        self.umap_cluster_coords = data.get("umap_cluster_coords", None)
        self.selected_cluster_coords = []
        self._draw_figure()

    def _setup_gui(self) -> None:
        """Set up the GUI components including controls and matplotlib canvas.

        Creates control frame with dropdown menus for figure type and color coding, buttons for
        cluster operations, and embeds a matplotlib figure for plotting.
        """
        # Top frame for controls (buttons and dropdowns)
        control_frame = ttk.Frame(self.master)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Dropdown for figure type
        figure_types = ["umap"]
        pc_dim = self.coords.shape[1]
        for i in range(min(5, pc_dim)):
            for j in range(i + 1, min(5, pc_dim)):
                figure_types.append(f"pc_{i}_{j}")
        ttk.Label(control_frame, text="Figure:").pack(side=tk.LEFT, padx=5)
        figure_menu = ttk.OptionMenu(
            control_frame,
            self.figure_type,
            self.figure_type.get(),
            *figure_types,
            command=lambda _: self._draw_figure(),
        )
        figure_menu.pack(side=tk.LEFT, padx=5)

        # Dropdown for color by
        color_options = ["None", "Density"] + [f"PC {i}" for i in range(self.coords.shape[1])]
        ttk.Label(control_frame, text="Color by:").pack(side=tk.LEFT, padx=5)
        color_menu = ttk.OptionMenu(
            control_frame, self.color_by, self.color_by.get(), *color_options, command=lambda _: self._draw_figure()
        )
        color_menu.pack(side=tk.LEFT, padx=5)

        # Buttons for cluster selection
        ttk.Button(control_frame, text="Reset Cluster Coords", command=self._reset_cluster_coords).pack(
            side=tk.LEFT, padx=5
        )

        # Matplotlib figure
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect("button_press_event", self._on_click)

    def _draw_figure(self) -> None:
        """Draw the current figure based on selected type and color scheme.

        Updates the matplotlib figure based on the current figure type (UMAP or PCA) and color
        coding selection. Handles both scatter plots and density histograms.
        """
        fig_type = self.figure_type.get()
        color_by = self.color_by.get()
        # Prepare labels for coloring
        if color_by == "None" or color_by == "Density":
            labels = None
        else:
            idx = int(color_by.split(" ")[1])
            labels = self.coords[:, idx]

        type_fig = "hist" if color_by == "Density" else "scatter"
        # Use analyze.py plotting functions
        if fig_type == "umap":
            fig_dict = create_umap_figure(self.umap_coords, self.umap_cluster_coords, labels, fig_type=type_fig)
            fig = fig_dict["umap"]
            fig.savefig("test.jpg")
        else:
            i, j = map(int, fig_type.split("_")[1:])
            fig_dict = create_pc_figure(
                self.coords, self.cluster_coords, labels, num_pcs=max(i, j) + 1, fig_type=type_fig
            )
            fig = fig_dict.get(fig_type)
        # Remove the old canvas and create a new one
        self.canvas.get_tk_widget().pack_forget()
        plt.close(self.fig)  # Close the old figure to avoid memory leaks
        self.fig = fig
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.draw()

    def _on_click(self, event: Any) -> None:
        """Handle mouse click events on the plot to select cluster coordinates.

        When user clicks on the plot, finds the nearest data point and adds it
        to the cluster coordinates. Updates both UMAP and PCA coordinate sets.

        Args:
            event: Matplotlib mouse event containing click coordinates
        """
        # Only respond to clicks inside the axes
        if event.xdata is None or event.ydata is None:
            return

        fig_type = self.figure_type.get()
        # Get the current visible coordinates
        if fig_type == "umap":
            coords = self.umap_coords
        else:
            i, j = map(int, fig_type.split("_")[1:])
            coords = self.coords[:, [i, j]]

        # Find the closest point in coords to the click
        click_point = np.array([event.xdata, event.ydata])
        dists = np.linalg.norm(coords - click_point, axis=1)
        idx = np.argmin(dists)

        # vstack to umap_cluster_coords or cluster_coords
        if self.umap_cluster_coords is None:
            self.umap_cluster_coords = np.array([self.umap_coords[idx]])
            self.cluster_coords = np.array([self.coords[idx]])
        else:
            self.umap_cluster_coords = np.vstack([self.umap_cluster_coords, self.umap_coords[idx][None, :]])
            self.cluster_coords = np.vstack([self.cluster_coords, self.coords[idx][None, :]])

        self._draw_figure()

    def _reset_cluster_coords(self) -> None:
        """Reset all selected cluster coordinates and refresh the display.

        Clears both UMAP and PCA cluster coordinate arrays and redraws the figure.
        """
        self.umap_cluster_coords = None
        self.cluster_coords = None
        self._draw_figure()

    def _save_cluster_coords(self) -> None:
        """Save the current cluster coordinates to a pickle file.

        Prompts user to select a save location and saves the cluster coordinates as a pickle file
        for later use.
        """
        path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl")],
            initialdir=self.dir,
            initialfile="latent_coords.pkl",
        )
        if path:
            with open(path, "wb") as f:
                pickle.dump(self.cluster_coords, f)

    def _load_latent_coords(self) -> None:
        """Load cluster coordinates from a pickle file.

        Prompts user to select a pickle file containing cluster coordinates, loads them, and maps
        them to UMAP coordinates for display.
        """
        path = filedialog.askopenfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl")],
            initialdir=self.dir,
            initialfile="latent_coords.pkl",
        )
        if path:
            with open(path, "rb") as f:
                self.cluster_coords = pickle.load(f)

            self.umap_cluster_coords = np.zeros((len(self.cluster_coords), self.umap_coords.shape[1]))
            for i, cluster_center in enumerate(self.cluster_coords):
                idx = np.where(np.all(self.coords == cluster_center, axis=1))[0][0]
                self.umap_cluster_coords[i] = self.umap_coords[idx]
            self._draw_figure()

    def _on_close(self) -> None:
        """Handle window close event.

        Properly closes the tkinter window and quits the application.
        """
        self.master.quit()
        self.master.destroy()


def main() -> None:
    """Main function to launch the AnalyzeViewer application.

    Creates the main tkinter window and loads analysis data either from command line argument or
    file dialog. Starts the interactive viewer application.
    """
    root = tk.Tk()
    root.title("Analyze Coordinates Viewer")
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        path = filedialog.askopenfilename(title="Select analyze_coordinates pkl", filetypes=[("Pickle files", "*.pkl")])
        if not path:
            print("No file selected.")
            return
    with open(path, "rb") as f:
        data = pickle.load(f)
    AnalyzeViewer(root, data, dir=os.path.split(path)[0])
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()


if __name__ == "__main__":
    main()
